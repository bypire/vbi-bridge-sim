"""Free-flow gate, hardened: full DYNAMIC multi-vehicle round-trip with noise.

Run:
    python -u solver/verify_gate.py

verify_multi.py made the multiple-presence point with a clean STATIC influence-line
superposition. This hardens it into a real round-trip:

  * forward model = the full coupled DYNAMIC convoy sim (integrate_convoy), so the
    gauge moment carries dynamic amplification, not just the static triangle;
  * + strain-gauge measurement noise, Monte-Carlo'd -> error BARS, not point values;
  * recover the target truck with single-vehicle B-WIM and measure the gross-weight
    error vs the following headway (mean +/- 1 sigma over noise);
  * then a traffic-stream model: ungated single-vehicle B-WIM vs a free-flow GATE
    (weigh only when the span carries one vehicle), reporting accuracy AND throughput.

The clean (single-presence) baseline error is the honest dynamic+noise floor; the
overlap error is what the gate removes. numpy-only core; matplotlib optional.
"""

import os
import numpy as np

from beam_fem import Beam
from bwim import add_measurement_noise, moment_influence_line, moses_recover
from multi_vehicle import integrate_convoy

L, E, I, M_BAR = 40.0, 2.1e11, 0.40, 12000.0
X_GAUGE = L / 2.0
V = 20.0
W_T, W_F = 300e3, 440e3          # target, follower gross [N]
NOISE_FRAC = 0.03
N_MC = 200
SEED = 7
CROSS = L / V


def recover_target_mc(beam, moment_t, res_t, t_win, x_target, rng):
    """Monte-Carlo single-vehicle B-WIM of the target over its window; return
    (mean_err%, std_err%) of the recovered gross weight under measurement noise."""
    m_win = np.interp(t_win, res_t, moment_t)
    errs = np.empty(N_MC)
    for k in range(N_MC):
        noisy = add_measurement_noise(m_win, NOISE_FRAC, rng)
        P, _ = moses_recover(beam, X_GAUGE, x_target, noisy, axle_offsets=(0.0,))
        errs[k] = 100 * (P[0] - W_T) / W_T
    return float(errs.mean()), float(errs.std())


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    rng = np.random.default_rng(SEED)
    t_win = np.linspace(0.0, CROSS, 250)
    x_target = V * t_win

    print("=" * 72)
    print("FREE-FLOW GATE (hardened) -- dynamic convoy + measurement noise + MC")
    print(f"  L={L} m, v={V} m/s, crossing={CROSS:.1f} s; target {W_T/1e3:.0f} kN, "
          f"follower {W_F/1e3:.0f} kN; {N_MC} noise draws/point")
    print("=" * 72)

    # ---- [1] clean dynamic+noise baseline (target alone) ------------------
    base = integrate_convoy(beam, [{"P": W_T, "speed": V, "enter": 0.0}], X_GAUGE)
    b_mean, b_std = recover_target_mc(beam, base.moment, base.t, t_win, x_target, rng)
    print(f"\n[1] CLEAN baseline (single presence, dynamic + {NOISE_FRAC*100:.0f}% noise):")
    print(f"    gross-weight error = {b_mean:+.1f}% +/- {b_std:.1f}%  "
          "(the honest dynamic+noise floor)")

    # ---- [2] dynamic error vs following headway ---------------------------
    print("\n[2] WITH a follower, error vs headway (mean +/- 1 sigma over noise):")
    print(f"    {'headway[s]':>10} {'overlap':>8} {'error%':>16}")
    headways = np.array([0.3, 0.5, 0.7, 1.0, 1.4, 1.8, 2.2, 2.8])
    curve = []
    for h in headways:
        conv = integrate_convoy(beam, [{"P": W_T, "speed": V, "enter": 0.0},
                                       {"P": W_F, "speed": V, "enter": float(h)}], X_GAUGE)
        mean, std = recover_target_mc(beam, conv.moment, conv.t, t_win, x_target, rng)
        curve.append((float(h), mean, std))
        ov = "yes" if h < CROSS else "no"
        print(f"    {h:>10.1f} {ov:>8} {mean:>+10.1f} +/- {std:.1f}")
    hs = np.array([c[0] for c in curve]); errc = np.array([c[1] for c in curve])

    def err_at_gap(g):
        """|error| a target suffers from its nearest neighbour at gap g [s]."""
        if g >= CROSS:
            return abs(b_mean)
        return abs(float(np.interp(g, hs, errc)))

    # ---- [3] traffic-stream model: ungated vs free-flow gate --------------
    print("\n[3] TRAFFIC STREAM: ungated single-vehicle B-WIM vs the free-flow gate")
    print(f"    {'trucks/min':>10} {'ungated|err|':>13} {'gated|err|':>11} "
          f"{'kept':>6}")
    rng2 = np.random.default_rng(SEED + 1)
    for q in [6, 12, 20, 30]:
        lam = q / 60.0
        n = 4000
        gaps = rng2.exponential(1.0 / lam, n + 1)        # inter-arrival gaps [s]
        nb = np.minimum(gaps[:-1], gaps[1:])             # each truck's nearest gap
        ung = np.array([err_at_gap(g) for g in nb])
        clean = nb >= CROSS
        ungated = ung.mean()
        gated = ung[clean].mean() if clean.any() else float("nan")
        kept = 100 * clean.mean()
        print(f"    {q:>10} {ungated:>12.1f}% {gated:>10.1f}% {kept:>5.0f}%")
    print("    -> the gate cuts the average error to the clean floor, keeping only the")
    print("       single-presence events; heavier traffic keeps fewer of them.")

    # ---- figure ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        means = [c[1] for c in curve]; stds = [c[2] for c in curve]
        cols = ["#ff5a5a" if h < CROSS else "#43e0a0" for h in hs]
        ax1.axhline(b_mean, color="#43e0a0", ls="--", lw=1.3, label=f"clean floor {b_mean:+.1f}%")
        ax1.axvline(CROSS, color="0.5", ls=":", label=f"crossing {CROSS:.0f} s")
        ax1.errorbar(hs, means, yerr=stds, fmt="none", ecolor="#888", capsize=3, zorder=1)
        ax1.scatter(hs, means, c=cols, s=55, zorder=2)
        ax1.set_xlabel("following headway [s]"); ax1.set_ylabel("B-WIM gross error [%]")
        ax1.set_title("Dynamic round-trip: error vs headway (+/-1σ noise)")
        ax1.legend(); ax1.grid(alpha=0.3)

        qs = np.linspace(2, 40, 60)
        ung_q, gat_q, kept_q = [], [], []
        rng3 = np.random.default_rng(SEED + 2)
        for q in qs:
            lam = q / 60.0; gaps = rng3.exponential(1.0/lam, 3001)
            nb = np.minimum(gaps[:-1], gaps[1:]); ung = np.array([err_at_gap(g) for g in nb])
            cl = nb >= CROSS
            ung_q.append(ung.mean()); gat_q.append(ung[cl].mean() if cl.any() else np.nan)
            kept_q.append(100*cl.mean())
        ax2.plot(qs, ung_q, "r-", lw=2, label="ungated mean |error|")
        ax2.plot(qs, gat_q, "g-", lw=2, label="gated mean |error|")
        ax2.plot(qs, kept_q, "b--", lw=1.5, label="% events kept (gate)")
        ax2.set_xlabel("traffic intensity [trucks/min]"); ax2.set_ylabel("% ")
        ax2.set_title("Free-flow gate: accuracy vs throughput")
        ax2.legend(); ax2.grid(alpha=0.3)

        fig.tight_layout()
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "gate_dynamic.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
