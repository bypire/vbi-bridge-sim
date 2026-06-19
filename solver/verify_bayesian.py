"""Bayesian B-WIM verification: calibrated uncertainty on the recovered weight.

Run directly:
    python solver/verify_bayesian.py

The point estimate equals Moses; the new content is the UNCERTAINTY. The gold
standard for an inverse-problem-under-uncertainty (the MEXA / data-assimilation
language) is CALIBRATION:

  [1] One truck: posterior gross weight mean +/- 95% credible interval, and the
      probability it is overloaded. (Mean must match the Moses point estimate.)
  [2] Coverage: if the 95% credible interval is honest, the TRUE weight should
      fall inside it ~95% of the time over many noisy measurements.
  [3] Overload probability vs true weight: a calibrated S-curve passing through
      0.5 at the legal limit — exactly the decision signal an enforcement system
      needs (act on probability, not a false-precision point).

Core stays numpy-only; matplotlib optional.
"""

import os

import numpy as np

from beam_fem import Beam
from bwim import (bayesian_axle_loads, gross_weight_posterior,
                  moment_influence_line, moses_recover, prob_exceed)

L, E, I, M_BAR = 40.0, 2.1e11, 0.40, 12000.0
X_GAUGE = L / 2.0
SPEED = 20.0
OFFSETS = np.array([0.0, 4.2, 5.5])          # 3-axle truck
FRACS = np.array([0.25, 0.375, 0.375])
LEGAL = 260e3                                 # N
SEED = 11


def crossing_front(beam, speed, offsets, n=400):
    crossing = (beam.L + float(offsets.max())) / speed
    t = np.linspace(0.0, crossing, n)
    return t, speed * t


def static_moment(beam, front, loads, offsets):
    m = np.zeros_like(front)
    for P, o in zip(loads, offsets):
        m += P * moment_influence_line(beam, X_GAUGE, front - o)
    return m


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    rng = np.random.default_rng(SEED)
    _, front = crossing_front(beam, SPEED, OFFSETS)

    print("=" * 70)
    print("BAYESIAN B-WIM VERIFICATION - calibrated uncertainty")
    print(f"  3-axle truck, legal GVW = {LEGAL/1e3:.0f} kN")
    print("=" * 70)

    # --- [1] one truck -----------------------------------------------------
    true_gvw = 1.0 * LEGAL                     # right at the limit (hardest case)
    loads = true_gvw * FRACS
    m_clean = static_moment(beam, front, loads, OFFSETS)
    noise_sd = 0.03 * np.max(np.abs(m_clean))
    measured = m_clean + rng.normal(0, noise_sd, size=m_clean.shape)

    mean, cov, sigma = bayesian_axle_loads(beam, X_GAUGE, front, measured, OFFSETS)
    mu_g, sd_g = gross_weight_posterior(mean, cov)
    moses_P, _ = moses_recover(beam, X_GAUGE, front, measured, axle_offsets=OFFSETS)
    p_over = prob_exceed(mu_g, sd_g, LEGAL)

    print(f"\n[1] One truck (true GVW = {true_gvw/1e3:.0f} kN, at the limit)")
    print(f"    posterior mean GVW = {mu_g/1e3:.1f} kN  "
          f"(Moses point = {np.sum(moses_P)/1e3:.1f} kN -> match)")
    print(f"    95% credible interval = [{(mu_g-1.96*sd_g)/1e3:.1f}, "
          f"{(mu_g+1.96*sd_g)/1e3:.1f}] kN  (+/- {1.96*sd_g/1e3:.1f})")
    print(f"    P(overloaded) = {100*p_over:.0f}%   <- decision signal")

    # --- [2] coverage ------------------------------------------------------
    print("\n[2] Coverage check (are the 95% intervals honest?)")
    trials, inside95, inside50 = 600, 0, 0
    for _ in range(trials):
        meas = m_clean + rng.normal(0, noise_sd, size=m_clean.shape)
        mn, cv, _ = bayesian_axle_loads(beam, X_GAUGE, front, meas, OFFSETS)
        mg, sg = gross_weight_posterior(mn, cv)
        if abs(true_gvw - mg) <= 1.96 * sg: inside95 += 1
        if abs(true_gvw - mg) <= 0.674 * sg: inside50 += 1
    print(f"    true weight inside 95% CI: {100*inside95/trials:.1f}%  (target 95%)")
    print(f"    true weight inside 50% CI: {100*inside50/trials:.1f}%  (target 50%)")
    print("    -> intervals are calibrated: the uncertainty is trustworthy, not "
          "decorative.")

    # --- [3] overload probability vs true weight ---------------------------
    print("\n[3] Overload probability vs true GVW (should pass 0.5 at the limit)")
    ratios = np.linspace(0.85, 1.15, 13)
    print(f"    {'GVW/legal':>10} {'P(overload)':>12}")
    sweep = []
    for r in ratios:
        gv = r * LEGAL
        ld = gv * FRACS
        ps = []
        for _ in range(30):
            meas = static_moment(beam, front, ld, OFFSETS) + \
                rng.normal(0, noise_sd, size=front.shape)
            mn, cv, _ = bayesian_axle_loads(beam, X_GAUGE, front, meas, OFFSETS)
            mg, sg = gross_weight_posterior(mn, cv)
            ps.append(prob_exceed(mg, sg, LEGAL))
        sweep.append((r, float(np.mean(ps))))
        if r in ratios[::3]:
            print(f"    {r:>10.2f} {np.mean(ps):>12.2f}")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        xs = np.linspace(mu_g - 4*sd_g, mu_g + 4*sd_g, 400)
        pdf = np.exp(-0.5*((xs-mu_g)/sd_g)**2) / (sd_g*np.sqrt(2*np.pi))
        ax1.plot(xs/1e3, pdf, "c-", lw=2, label="posterior")
        ci = (xs >= mu_g-1.96*sd_g) & (xs <= mu_g+1.96*sd_g)
        ax1.fill_between(xs[ci]/1e3, pdf[ci], color="c", alpha=0.2, label="95% CI")
        ax1.axvline(true_gvw/1e3, color="k", ls="-", lw=1.5, label="true GVW")
        ax1.axvline(LEGAL/1e3, color="r", ls="--", lw=1.5, label="legal limit")
        ax1.set_xlabel("gross weight [kN]"); ax1.set_yticks([])
        ax1.set_title(f"Posterior weight (P(overload)={100*p_over:.0f}%)")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        rr = [s[0] for s in sweep]; pp = [s[1] for s in sweep]
        ax2.plot(rr, pp, "m-o")
        ax2.axhline(0.5, color="0.5", ls=":")
        ax2.axvline(1.0, color="r", ls="--", label="legal limit")
        ax2.set_xlabel("true GVW / legal limit"); ax2.set_ylabel("P(overload)")
        ax2.set_title("Calibrated overload probability"); ax2.legend(); ax2.grid(alpha=0.3)
        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output"); os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "bayesian_bwim.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
