"""P2 verification: regularization fixes the ill-posed tandem-axle B-WIM split.

Run directly:
    python -u solver/verify_regularize.py

Three checks against ground truth:
  [1] CONDITIONING vs axle spacing -- show cond(C^T C) blows up as a tandem closes,
      quantifying WHY Moses cannot split closely-spaced axles.
  [2] MOSES vs TIKHONOV (Monte Carlo) -- a true 3-axle truck with a close tandem;
      recover the loads under measurement noise with plain least squares (Moses)
      and with L-curve-chosen Tikhonov. Report per-axle bias & scatter and the
      gross weight. Expect: Moses has huge individual-axle scatter but good gross;
      Tikhonov trades a little bias for a large scatter reduction (bias-variance).
  [3] L-CURVE -- show the corner that picks lambda automatically.

The forward signal here is the static influence-line superposition (we are
isolating the INVERSE conditioning, not the dynamics -- the dynamic/model error is
characterised separately in verify_coverage.py). numpy-only core; matplotlib optional.
"""

import os

import numpy as np

from beam_fem import Beam
from bwim import moment_influence_line, moses_recover
from regularize import condition_number, l_curve, tikhonov_recover

L, E, I, M_BAR = 40.0, 2.1e11, 0.40, 12000.0
X_GAUGE = L / 2.0
SPEED = 20.0
N_SAMPLES = 400
NOISE_FRAC = 0.03
SEED = 7


def build_C(beam, offsets, n=N_SAMPLES):
    """Influence matrix for a truck whose first axle sweeps the span."""
    crossing = (beam.L + float(np.max(offsets))) / SPEED
    front = SPEED * np.linspace(0.0, crossing, n)
    cols = [moment_influence_line(beam, X_GAUGE, front - o) for o in offsets]
    return np.column_stack(cols), front


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    rng = np.random.default_rng(SEED)

    print("=" * 74)
    print("REGULARIZED B-WIM - taming the ill-posed multi-axle split")
    print(f"  bridge L={L} m, gauge mid-span, v={SPEED} m/s")
    print("=" * 74)

    # ---- [1] conditioning vs tandem spacing ----------------------------------
    print("\n[1] CONDITIONING: cond(C^T C) vs tandem axle spacing (2 axles)")
    print(f"    {'spacing[m]':>10} {'cond(C^T C)':>14}")
    spacings = [0.5, 1.0, 1.3, 2.0, 3.0, 5.0, 8.0]
    conds = []
    for d in spacings:
        C, _ = build_C(beam, np.array([0.0, d]))
        c = condition_number(C)
        conds.append(c)
        print(f"    {d:>10.1f} {c:>14.3e}")
    print("    -> closer axles => columns of C nearly parallel => cond explodes =>\n"
          "       the load SPLIT is what blows up, while the SUM (gross) stays stable.")

    # ---- [2] Moses vs Tikhonov on a real tandem (Monte Carlo) ----------------
    offsets = np.array([0.0, 4.0, 5.3])          # steer + close tandem (1.3 m)
    true_loads = np.array([60e3, 95e3, 95e3])    # N  (tandem nominally equal)
    gross_true = true_loads.sum()
    C, front = build_C(beam, offsets)
    m_clean = C @ true_loads
    noise_sd = NOISE_FRAC * np.max(np.abs(m_clean))
    cond_C = condition_number(C)

    # choose lambda once, from the L-curve of a representative noisy measurement
    lams = np.logspace(-2, 4, 80) * np.sqrt(C.shape[0])  # scale to problem size
    meas0 = m_clean + rng.normal(0, noise_sd, size=m_clean.shape)
    res, sol, lam_arr, i_corner, _ = l_curve(C, meas0, lams)
    lam_star = lam_arr[i_corner]

    print(f"\n[2] MOSES vs TIKHONOV  (3-axle, tandem spacing 1.3 m, cond={cond_C:.2e})")
    print(f"    true loads = {true_loads/1e3} kN, gross = {gross_true/1e3:.0f} kN")
    print(f"    L-curve corner lambda* = {lam_star:.3g}")

    trials = 500
    moses_P = np.empty((trials, 3))
    tikh_P = np.empty((trials, 3))
    for k in range(trials):
        meas = m_clean + rng.normal(0, noise_sd, size=m_clean.shape)
        moses_P[k], _ = moses_recover(beam, X_GAUGE, front, meas, axle_offsets=offsets)
        tikh_P[k] = tikhonov_recover(C, meas, lam_star)

    def report(name, P):
        mean, std = P.mean(0), P.std(0)
        gross = P.sum(1)
        print(f"\n    [{name}] per-axle recovery over {trials} noisy passes:")
        for i in range(3):
            bias = 100 * (mean[i] - true_loads[i]) / true_loads[i]
            scat = 100 * std[i] / true_loads[i]
            print(f"      axle {i+1}: mean {mean[i]/1e3:6.1f} kN  "
                  f"bias {bias:+6.1f}%  scatter {scat:5.1f}%")
        gb = 100 * (gross.mean() - gross_true) / gross_true
        gs = 100 * gross.std() / gross_true
        print(f"      GROSS : mean {gross.mean()/1e3:6.1f} kN  "
              f"bias {gb:+6.1f}%  scatter {gs:5.1f}%")
        return std / true_loads

    moses_scat = report("MOSES   ", moses_P)
    tikh_scat = report("TIKHONOV", tikh_P)
    print(f"\n    tandem-axle scatter cut by regularization: "
          f"{moses_scat[1]/tikh_scat[1]:.1f}x (axle 2), "
          f"{moses_scat[2]/tikh_scat[2]:.1f}x (axle 3).")
    print("    Gross weight is essentially unchanged: regularization fixes the\n"
          "    SPLIT (the ill-posed part) without touching the well-posed SUM.")

    # ---- visualization -------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

        ax1.semilogy(spacings, conds, "k-o")
        ax1.axvline(1.3, color="r", ls="--", label="tandem (1.3 m)")
        ax1.set_xlabel("axle spacing [m]"); ax1.set_ylabel("cond(C$^T$C)")
        ax1.set_title("Why the split is ill-posed"); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.loglog(res, sol, "b-")
        ax2.loglog(res[i_corner], sol[i_corner], "ro", ms=10,
                   label=f"corner $\\lambda$*={lam_star:.2g}")
        ax2.set_xlabel("residual ||C P - m||"); ax2.set_ylabel("solution ||P||")
        ax2.set_title("L-curve picks $\\lambda$"); ax2.legend(); ax2.grid(alpha=0.3, which="both")

        x = np.arange(3)
        ax3.bar(x - 0.2, moses_P.std(0) / 1e3, 0.38, color="tomato",
                label="Moses (lstsq)")
        ax3.bar(x + 0.2, tikh_P.std(0) / 1e3, 0.38, color="seagreen",
                label="Tikhonov")
        ax3.set_xticks(x); ax3.set_xticklabels(["steer", "drive", "trailer"])
        ax3.set_ylabel("per-axle scatter (1$\\sigma$) [kN]")
        ax3.set_title("Regularization tames the tandem split")
        ax3.legend(); ax3.grid(alpha=0.3, axis="y")

        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output"); os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "regularize.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
