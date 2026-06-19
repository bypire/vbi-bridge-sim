"""V4.5 verification: B-WIM weight recovery (Moses' algorithm).

Run directly:
    python solver/verify_bwim.py

Round-trip ground truth: put a KNOWN axle weight into the
forward simulator, read the bridge's bending-moment history at a gauge section,
add measurement noise, recover the weight with Moses' least squares, and compare
to the known input.

  [0] Influence-line + sign check: FEM static moment at the gauge vs the
      analytical triangle (verifies the basis Moses fits to).
  [1] Noise-free recovery: dynamic quarter-car signal -> recovered W vs true W.
  [2] Noise robustness: 3% strain noise, Monte-Carlo -> mean & spread of error.
  [3] Recovery error vs speed: higher speed -> more dynamics -> larger B-WIM error.

Core stays numpy-only; matplotlib is optional tooling here.
"""

import os

import numpy as np

from beam_fem import Beam, static_solve
from moving_load import bending_moment_at
from vehicle import QuarterCar, integrate_coupled
from bwim import add_measurement_noise, moment_influence_line, moses_recover

# --- same bridge + car as V3 ------------------------------------------------
L, E, I, M_BAR = 20.0, 2.1e11, 0.02, 2000.0
CAR = QuarterCar(m_s=9000.0, m_u=1000.0, k_s=2.0e6, k_t=1.0e7, c_s=6.0e4, g=9.81)

X_GAUGE = L / 2.0          # strain gauge at mid-span
DEMO_SPEED = 20.0
NOISE_FRAC = 0.03          # 3% of peak signal
N_MONTE_CARLO = 300
SEED = 12345


def recover_at_speed(beam, speed, noise_frac=0.0, rng=None):
    """Forward-simulate a crossing, then recover the axle weight with Moses."""
    res = integrate_coupled(beam, CAR, speed, moment_section=X_GAUGE)
    front = speed * res.t
    moment = res.moment
    if noise_frac > 0.0:
        moment = add_measurement_noise(moment, noise_frac, rng)
    P, fitted = moses_recover(beam, X_GAUGE, front, moment, axle_offsets=(0.0,))
    return res, front, moment, fitted, float(P[0])


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    W = CAR.weight

    print("=" * 70)
    print("V4.5 VERIFICATION - B-WIM axle-weight recovery (Moses' algorithm)")
    print(f"  bridge L={L} m;  gauge at x={X_GAUGE} m (mid-span)")
    print(f"  TRUE static axle weight W = {W:.1f} N  ({W/1e3:.2f} kN)")
    print("=" * 70)

    # --- [0] influence line + sign ----------------------------------------
    print("\n[0] MOMENT INFLUENCE LINE at the gauge: FEM static vs analytical")
    print(f"    {'load x[m]':>9} {'FEM M[N m]':>12} {'analytic[N m]':>14} "
          f"{'rel.err':>9}")
    worst = 0.0
    for n in [2, 5, 8, 10, 12, 15, 18]:
        x_load = beam.node_x(n)
        u = static_solve(beam, loads={n: 1.0})       # unit load at that node
        m_fem = bending_moment_at(beam, u, X_GAUGE)
        m_ana = float(moment_influence_line(beam, X_GAUGE, np.array([x_load]))[0])
        rel = abs(m_fem - m_ana) / abs(m_ana)
        worst = max(worst, rel)
        print(f"    {x_load:>9.2f} {m_fem:>12.5f} {m_ana:>14.5f} {rel:>9.2e}")
    print(f"    peak (analytical) = L/4 = {L/4:.3f} N m under a unit load at "
          f"mid-span; worst rel.err = {worst:.2e}")

    # --- [1] noise-free recovery ------------------------------------------
    res, front, moment, fitted, W_rec = recover_at_speed(beam, DEMO_SPEED)
    err = (W_rec - W) / W
    print(f"\n[1] NOISE-FREE recovery at v = {DEMO_SPEED} m/s "
          f"({DEMO_SPEED*3.6:.0f} km/h), DAF = {res.daf:.3f}")
    print(f"    peak measured moment   = {np.max(np.abs(moment)):.1f} N m")
    print(f"    recovered weight W_rec = {W_rec:.1f} N  ({W_rec/1e3:.2f} kN)")
    print(f"    true weight W          = {W:.1f} N")
    print(f"    error                  = {100*err:+.2f} %")

    # --- [2] noise robustness (Monte Carlo) -------------------------------
    rng = np.random.default_rng(SEED)
    # reuse one deterministic dynamic signal; only the added noise varies
    res2 = integrate_coupled(beam, CAR, DEMO_SPEED, moment_section=X_GAUGE)
    front2 = DEMO_SPEED * res2.t
    recovered = np.empty(N_MONTE_CARLO)
    for k in range(N_MONTE_CARLO):
        noisy = add_measurement_noise(res2.moment, NOISE_FRAC, rng)
        P, _ = moses_recover(beam, X_GAUGE, front2, noisy, axle_offsets=(0.0,))
        recovered[k] = P[0]
    mean_err = 100 * (recovered.mean() - W) / W
    std_pct = 100 * recovered.std() / W
    print(f"\n[2] NOISE ROBUSTNESS: {NOISE_FRAC*100:.0f}% peak noise, "
          f"{N_MONTE_CARLO} runs")
    print(f"    recovered W: mean = {recovered.mean():.1f} N "
          f"({recovered.mean()/1e3:.2f} kN), std = {recovered.std():.1f} N")
    print(f"    mean error = {mean_err:+.2f} %,  scatter (1 sigma) = "
          f"{std_pct:.2f} %")
    print("    (Noise is zero-mean, so averaging over the whole crossing keeps\n"
          "     the bias tiny; the scatter is the per-pass uncertainty.)")

    # --- [3] error vs speed -----------------------------------------------
    print("\n[3] RECOVERY ERROR vs SPEED (noise-free; the pure dynamic error)")
    print(f"    {'v[m/s]':>7} {'km/h':>6} {'DAF':>7} {'W_rec[kN]':>10} "
          f"{'error[%]':>9}")
    sweep = []
    for v in [10, 20, 30, 50, 80]:
        r, fr, mo, fi, wr = recover_at_speed(beam, v)
        e = 100 * (wr - W) / W
        sweep.append((v, r.daf, wr, e))
        print(f"    {v:>7.0f} {v*3.6:>6.0f} {r.daf:>7.3f} {wr/1e3:>10.2f} "
              f"{e:>9.2f}")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # one noisy realisation for the fit panel
        rng_plot = np.random.default_rng(SEED)
        noisy_demo = add_measurement_noise(res.moment, NOISE_FRAC, rng_plot)
        Pn, fit_noisy = moses_recover(beam, X_GAUGE, front, noisy_demo,
                                      axle_offsets=(0.0,))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

        ax1.plot(res.t, noisy_demo / 1e3, color="0.6", lw=0.8,
                 label=f"measured (+{NOISE_FRAC*100:.0f}% noise)")
        ax1.plot(res.t, moment / 1e3, "b-", lw=1.3, label="true dynamic signal")
        ax1.plot(res.t, fit_noisy / 1e3, "r--", lw=1.8,
                 label=f"Moses fit -> {Pn[0]/1e3:.1f} kN")
        ax1.set_xlabel("time t [s]")
        ax1.set_ylabel("bending moment at gauge [kN·m]")
        ax1.set_title("Measured moment vs Moses fit")
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.hist(recovered / 1e3, bins=30, color="steelblue", edgecolor="k",
                 alpha=0.8)
        ax2.axvline(W / 1e3, color="r", lw=2, label=f"true W = {W/1e3:.1f} kN")
        ax2.axvline(recovered.mean() / 1e3, color="k", ls="--",
                    label=f"mean = {recovered.mean()/1e3:.1f} kN")
        ax2.set_xlabel("recovered weight [kN]")
        ax2.set_ylabel("count")
        ax2.set_title(f"Recovery under {NOISE_FRAC*100:.0f}% noise "
                      f"({N_MONTE_CARLO} passes)")
        ax2.legend(); ax2.grid(alpha=0.3)

        vs = [s[0] for s in sweep]
        ax3.axhline(0.0, color="gray", lw=1)
        ax3.plot(vs, [s[3] for s in sweep], "m-o")
        ax3.set_xlabel("vehicle speed v [m/s]")
        ax3.set_ylabel("weight error [%]")
        ax3.set_title("B-WIM dynamic error vs speed")
        ax3.grid(alpha=0.3)

        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "v4_5_bwim.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
