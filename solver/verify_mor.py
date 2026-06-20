"""MOR verification: modal truncation accuracy vs full model / Frýba + speedup.

Run directly:
    python solver/verify_mor.py

  [1] Accuracy vs the number of retained modes r: peak mid-span deflection
      vs the Frýba closed form. A few modes should already nail it.
  [2] Speed-up: the full model's explicit step is set by its stiffest mode; the
      reduced model's by the retained one -> far fewer DOFs AND a bigger step.

Core stays numpy-only; matplotlib optional.
"""

import os

import numpy as np

from beam_fem import Beam, free_dofs, natural_frequencies
from moving_load import fryba_deflection, integrate_moving_force
from mor import integrate_modal_moving_force, modal_basis

L, E, I, M_BAR, P = 20.0, 2.1e11, 0.02, 2000.0, 1.0e5
V = 30.0


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    nf = len(free_dofs(beam))

    print("=" * 70)
    print("MODEL ORDER REDUCTION (modal truncation) VERIFICATION")
    print(f"  full model: {nf} free DOFs;  v={V} m/s")
    print("=" * 70)

    # reference peak from Frýba
    full = integrate_moving_force(beam, P, V)
    w_fry = fryba_deflection(beam, P, V, full.t)
    peak_fry = w_fry[np.argmax(np.abs(w_fry))]

    print("\n[1] Accuracy vs retained modes r (peak deflection vs Frýba)")
    print(f"    {'r':>3} {'peak [m]':>13} {'rel.err':>10} {'DAF':>8}")
    res = {}
    for r in [1, 2, 3, 5, 10]:
        rm = integrate_modal_moving_force(beam, P, V, r)
        res[r] = rm
        pk = rm.w_mid[np.argmax(np.abs(rm.w_mid))]
        print(f"    {r:>3} {pk:>13.6e} {abs(pk-peak_fry)/abs(peak_fry):>10.2e} "
              f"{rm.daf:>8.4f}")
    print(f"    (full {nf}-DOF / Frýba DAF = {full.daf:.4f}; ~3 modes already match)")
    pk3 = res[3].w_mid[np.argmax(np.abs(res[3].w_mid))]
    assert abs(pk3 - peak_fry) / abs(peak_fry) < 5e-3, \
        "3-mode reduced model must reproduce the full/Fryba peak"

    # --- [2] speed-up ------------------------------------------------------
    om_full = 2 * np.pi * natural_frequencies(beam, nf)[-1]    # full omega_max
    om_3 = float(modal_basis(beam, 3)[0][-1])
    print("\n[2] Speed-up (reduced keeps r=3 modes)")
    print(f"    full model : {nf} DOFs, dt limited by omega_max ~ {om_full:.0f} rad/s")
    print(f"    reduced    : 3 DOFs,  dt limited by omega_3   ~ {om_3:.0f} rad/s")
    print(f"    -> ~{nf/3:.0f}x fewer DOFs and ~{om_full/om_3:.0f}x larger stable "
          f"step; the response is unchanged.")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        rs = [1, 2, 3, 5, 10]
        errs = []
        for r in rs:
            pk = res[r].w_mid[np.argmax(np.abs(res[r].w_mid))]
            errs.append(abs(pk - peak_fry) / abs(peak_fry))
        ax1.semilogy(rs, errs, "m-o")
        ax1.set_xlabel("retained modes r"); ax1.set_ylabel("peak rel. error vs Frýba")
        ax1.set_title("Modal truncation convergence"); ax1.grid(alpha=0.3, which="both")

        ax2.plot(full.t, w_fry * 1e3, "k-", lw=2.5, label="Frýba (exact)")
        ax2.plot(res[1].t, res[1].w_mid * 1e3, ":", lw=1.5, label="r=1 mode")
        ax2.plot(res[3].t, res[3].w_mid * 1e3, "c--", lw=1.5, label="r=3 modes")
        ax2.invert_yaxis(); ax2.set_xlabel("time [s]")
        ax2.set_ylabel("mid-span deflection [mm]")
        ax2.set_title("Reduced vs exact (r=3 already matches)")
        ax2.legend(); ax2.grid(alpha=0.3)
        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output"); os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "mor_verify.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
