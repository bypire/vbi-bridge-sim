"""Multiple-vehicle physics + the multiple-PRESENCE B-WIM problem.

Run:
    python -u solver/verify_multi.py

Two parts:

  [1] CORRECTNESS (assumption-free): the convoy march is linear, so N vehicles =
      exact superposition of the single-vehicle responses. We confirm it to machine
      precision (a free, rigorous check that the multi-vehicle integrator is right).

  [2] Research question (branches off standard B-WIM): classic
      B-WIM assumes ONE vehicle on the span. When a second truck is simultaneously
      present, the gauge sees the SUM of both bending effects, and naive
      single-vehicle recovery mis-weighs the target. We quantify that error vs the
      following headway, and show a simple **free-flow gate** (weigh only when the
      span carries one vehicle) trades throughput for accuracy. Multiple presence is
      a known primary B-WIM error source (COST323 accuracy classes; O'Brien et al.);
      here it is cleanly quantified on an open model. [refs to verify before README]

numpy-only core; matplotlib optional.
"""

import os
import numpy as np

from beam_fem import Beam
from bwim import moment_influence_line, moses_recover
from multi_vehicle import integrate_convoy

L, E, I, M_BAR = 40.0, 2.1e11, 0.40, 12000.0
X_GAUGE = L / 2.0
V = 20.0                       # m/s (72 km/h)
SEED = 7


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    print("=" * 72)
    print("MULTIPLE VEHICLES -- superposition check + the multiple-presence problem")
    print(f"  bridge L={L} m, gauge mid-span, v={V} m/s, crossing={L/V:.1f} s")
    print("=" * 72)

    # ---- [1] superposition correctness -----------------------------------
    # Keep an identical time grid by always passing 2 vehicles; zero one out.
    v1 = {"P": 300e3, "speed": V, "enter": 0.0}
    v2 = {"P": 400e3, "speed": V, "enter": 0.6}
    both = integrate_convoy(beam, [v1, v2], X_GAUGE)
    only1 = integrate_convoy(beam, [v1, {**v2, "P": 0.0}], X_GAUGE)
    only2 = integrate_convoy(beam, [{**v1, "P": 0.0}, v2], X_GAUGE)
    err = np.max(np.abs(both.moment - (only1.moment + only2.moment))) / \
        np.max(np.abs(both.moment))
    print("\n[1] SUPERPOSITION (linear beam): moment(v1+v2) vs moment(v1)+moment(v2)")
    print(f"    max rel. difference = {err:.2e}   (expected ~machine zero)")
    print(f"    up to {both.n_on_max} vehicles were on the span at once.")

    # ---- [2] multiple-presence B-WIM error vs headway --------------------
    # Target truck T crosses from t=0; a heavier follower F enters `h` seconds later.
    # Gauge sees BOTH; naive single-vehicle Moses attributes it all to T.
    W_T, W_F = 300e3, 440e3
    n = 400
    t = np.linspace(0.0, L / V, n)               # window while T is on the span
    xT = V * t
    print(f"\n[2] MULTIPLE-PRESENCE ERROR (target {W_T/1e3:.0f} kN, "
          f"follower {W_F/1e3:.0f} kN)")
    print(f"    {'headway[s]':>10} {'gap[m]':>7} {'overlap':>8} {'W_rec[kN]':>10} {'error%':>8}")
    headways = [0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 2.5]
    rows = []
    for h in headways:
        xF = V * (t - h)
        m_meas = W_T * moment_influence_line(beam, X_GAUGE, xT) + \
            W_F * moment_influence_line(beam, X_GAUGE, xF)
        P, _ = moses_recover(beam, X_GAUGE, xT, m_meas, axle_offsets=(0.0,))
        err_pct = 100 * (P[0] - W_T) / W_T
        overlap = "yes" if h < L / V else "no"
        rows.append((h, P[0], err_pct, h < L / V))
        print(f"    {h:>10.1f} {h*V:>7.0f} {overlap:>8} {P[0]/1e3:>10.0f} {err_pct:>+8.1f}")
    print(f"    -> overlap (headway < crossing {L/V:.0f} s) inflates the weight badly;")
    print("       once the follower clears the span the recovery is clean again.")

    # ---- free-flow gate: throughput vs accuracy --------------------------
    # Poisson traffic: mean headway 1/lambda. A target is 'clean' if no other vehicle
    # is on the span during its crossing -> headway > crossing time L/V both sides.
    print("\n    FREE-FLOW GATE (weigh only single-presence events):")
    cross = L / V
    for q_per_min in [6, 12, 20, 30]:
        lam = q_per_min / 60.0                    # trucks per second
        # P(clean) ~ P(gap before AND after > crossing) = exp(-2 lam * cross)
        p_clean = float(np.exp(-2 * lam * cross))
        print(f"      {q_per_min:>2} trucks/min ->  {100*p_clean:4.0f}% of events are "
              f"clean (gated), {100*(1-p_clean):4.0f}% rejected")
    print("      The gate restores accuracy by spending throughput -- the core")
    print("      multiple-presence trade-off, quantified on this model.")

    # ---- figure ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        hs = [r[0] for r in rows]; es = [r[2] for r in rows]
        cols = ["#ff5a5a" if r[3] else "#43e0a0" for r in rows]
        ax1.axhline(0, color="0.6", lw=1)
        ax1.axvline(cross, color="0.5", ls="--", label=f"crossing time = {cross:.0f} s")
        ax1.plot(hs, es, "-", color="#888", lw=1.2, zorder=1)
        ax1.scatter(hs, es, c=cols, s=60, zorder=2)
        ax1.set_xlabel("following headway [s]"); ax1.set_ylabel("B-WIM weight error [%]")
        ax1.set_title("Multiple presence corrupts single-vehicle B-WIM")
        ax1.legend(); ax1.grid(alpha=0.3)

        qs = np.linspace(2, 40, 100); lam = qs / 60.0
        pc = np.exp(-2 * lam * cross)
        ax2.plot(qs, 100 * pc, "b-", lw=2, label="clean (weighable) events")
        ax2.plot(qs, 100 * (1 - pc), "r-", lw=2, label="rejected (multiple presence)")
        ax2.set_xlabel("traffic intensity [trucks/min]")
        ax2.set_ylabel("share of events [%]")
        ax2.set_title("Free-flow gate: throughput vs accuracy")
        ax2.legend(); ax2.grid(alpha=0.3)

        fig.tight_layout()
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "multi_presence.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
