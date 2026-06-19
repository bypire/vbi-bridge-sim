"""Newmark-beta verification: accuracy vs Frýba, and the stability advantage.

Run directly:
    python solver/verify_newmark.py

  [1] Accuracy: Newmark mid-span history & DAF vs the Frýba closed form
      (the same ground truth V2 used for RK4).
  [2] Convergence: peak-deflection error vs time step dt (should fall ~dt^2).
  [3] The point: on the realistic DAMPED 40 m bridge, explicit RK4 needs
      dt ~ 4e-6 s to stay stable, while Newmark is accurate AND stable at a step
      ~1000x larger — why implicit integrators are standard in structural codes
      (and why our batch exports were slow with RK4 + damping).

Core stays numpy-only; matplotlib optional.
"""

import os

import numpy as np

from beam_fem import Beam
from moving_load import fryba_deflection, integrate_moving_force
from newmark import integrate_newmark_moving_force

# 20 m verification beam (undamped) — has the Frýba ground truth
L, E, I, M_BAR, P = 20.0, 2.1e11, 0.02, 2000.0, 1.0e5
# 40 m realistic damped overpass — for the stability comparison
L2, E2, I2, MB2 = 40.0, 2.1e11, 0.40, 12000.0


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    v = 30.0

    print("=" * 70)
    print("NEWMARK-beta VERIFICATION (average-acceleration, gamma=1/2, beta=1/4)")
    print("=" * 70)

    # --- [1] accuracy vs Frýba --------------------------------------------
    dt = 1.0e-3
    rn = integrate_newmark_moving_force(beam, P, v, dt=dt)
    w_fry = fryba_deflection(beam, P, v, rn.t)
    peak_nm = rn.w_mid[np.argmax(np.abs(rn.w_mid))]
    peak_fr = w_fry[np.argmax(np.abs(w_fry))]
    print(f"\n[1] Accuracy vs Frýba (v={v} m/s, dt={dt} s)")
    print(f"    peak Newmark = {peak_nm:.6e} m")
    print(f"    peak Frýba   = {peak_fr:.6e} m")
    print(f"    rel. error   = {abs(peak_nm-peak_fr)/abs(peak_fr):.2e}")
    print(f"    DAF Newmark={rn.daf:.4f}")

    # --- [2] convergence ---------------------------------------------------
    print("\n[2] Convergence: peak-deflection error vs dt (expect ~ dt^2)")
    print(f"    {'dt [s]':>10} {'rel.err':>10} {'ratio':>8}")
    prev = None
    for dt in [4e-3, 2e-3, 1e-3, 5e-4]:
        r = integrate_newmark_moving_force(beam, P, v, dt=dt)
        wf = fryba_deflection(beam, P, v, r.t)
        pk = r.w_mid[np.argmax(np.abs(r.w_mid))]
        pf = wf[np.argmax(np.abs(wf))]
        err = abs(pk - pf) / abs(pf)
        ratio = "" if prev is None else f"{prev/err:.1f}x"
        print(f"    {dt:>10.0e} {err:>10.2e} {ratio:>8}")
        prev = err

    # --- [3] stability on the damped 40 m bridge ---------------------------
    beam2 = Beam(L=L2, E=E2, I=I2, mass_per_length=MB2, n_elements=20)
    rk4 = integrate_moving_force(beam2, 3.0e5, 25.0, damping_ratio=0.02)
    nm = integrate_newmark_moving_force(beam2, 3.0e5, 25.0, dt=4e-3,
                                        damping_ratio=0.02)
    print("\n[3] Damped 40 m bridge (zeta=2%): explicit RK4 vs implicit Newmark")
    print(f"    RK4      : dt = {rk4.dt:.2e} s  (forced tiny by stability), "
          f"DAF={rk4.daf:.4f}")
    print(f"    Newmark  : dt = {nm.dt:.2e} s  ({nm.dt/rk4.dt:.0f}x larger), "
          f"DAF={nm.daf:.4f}")
    print(f"    DAF agree to {abs(rk4.daf-nm.daf)/rk4.daf:.1e}; Newmark uses "
          f"~{rk4.dt and int(nm.dt/rk4.dt)}x fewer steps -> implicit wins on "
          "stiff damped systems.")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        rn = integrate_newmark_moving_force(beam, P, v, dt=1e-3)
        wf = fryba_deflection(beam, P, v, rn.t)
        ax1.plot(rn.t, wf * 1e3, "k-", lw=2.5, label="Frýba (exact)")
        ax1.plot(rn.t, rn.w_mid * 1e3, "c--", lw=1.5, label="Newmark-β")
        ax1.invert_yaxis(); ax1.set_xlabel("time [s]")
        ax1.set_ylabel("mid-span deflection [mm]")
        ax1.set_title("Newmark vs Frýba (20 m beam)"); ax1.legend(); ax1.grid(alpha=0.3)

        mask = nm.t <= nm.crossing_time
        ax2.plot(rk4.t, rk4.w_mid * 1e3, "r-", lw=1.0,
                 label=f"RK4 (dt={rk4.dt:.0e}s)")
        ax2.plot(nm.t[mask], nm.w_mid[mask] * 1e3, "c--", lw=1.8,
                 label=f"Newmark (dt={nm.dt:.0e}s)")
        ax2.invert_yaxis(); ax2.set_xlabel("time [s]")
        ax2.set_ylabel("mid-span deflection [mm]")
        ax2.set_title("Damped 40 m bridge: same answer, 1000× bigger step")
        ax2.legend(); ax2.grid(alpha=0.3)
        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output"); os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "newmark_verify.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
