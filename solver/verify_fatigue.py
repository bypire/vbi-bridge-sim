"""Fatigue & economics verification: what an overloaded truck actually costs.

Run directly:
    python -u solver/verify_fatigue.py

This closes the loop on the project's cost story. B-WIM gives a weight; fatigue
turns that weight into bridge life spent. Three checks:

  [1] THE POWER LAW (assumption-free): doubling the load multiplies fatigue damage
      by 2^m -- 8x for steel (Eurocode m=3), 16x for pavement (AASHTO m=4). Exact.
  [2] PARETO (assumption-free, from the simulated traffic stream): because damage
      grows with the cube/4th power of load, the heaviest few percent of trucks
      cause most of the fatigue consumption -- and those are exactly the trucks
      B-WIM flags. We report the concentration and a Lorenz curve.
  [3] EURO ILLUSTRATION (explicit inputs): treating the bridge as a depreciating
      asset, what one legal pass vs one overloaded pass costs, and the annual
      fatigue bill the overweight minority imposes.

numpy-only core; matplotlib optional.
"""

import os

import numpy as np

from fatigue import (M_PAVEMENT, M_STEEL, cost_per_pass, lorenz,
                     load_equivalency, pareto_concentration, stream_damage)
from traffic import generate_population

# --- explicit economic inputs (order-of-magnitude, stated so they're auditable) ---
W_REF = 260e3            # reference legal gross weight [N] (~26.5 t, EU 3-axle limit)
N_DESIGN = 50e6          # equivalent legal passes the detail is designed to survive
                         # (~1370 heavy trucks/day * 365 * 100 yr)
REPLACEMENT_COST = 40e6  # euro to replace a 40 m highway overpass (order of magnitude)
N_TRUCKS = 400
SEED = 7


def main():
    print("=" * 72)
    print("FATIGUE & ECONOMICS - turning a recovered weight into bridge life spent")
    print(f"  reference legal gross = {W_REF/1e3:.0f} kN;  S-N slope m = {M_STEEL} "
          f"(steel, Eurocode)")
    print("=" * 72)

    # ---- [1] the power law (exact) ---------------------------------------
    r_steel = load_equivalency(2 * W_REF, W_REF, M_STEEL) / load_equivalency(W_REF, W_REF, M_STEEL)
    r_pave = load_equivalency(2 * W_REF, W_REF, M_PAVEMENT) / load_equivalency(W_REF, W_REF, M_PAVEMENT)
    print("\n[1] POWER LAW: double the load -> how much more damage?")
    print(f"    steel detail (m=3): {r_steel:.1f}x   (expected 2^3 = 8)")
    print(f"    pavement     (m=4): {r_pave:.1f}x   (expected 2^4 = 16)")
    assert abs(r_steel - 8) < 1e-9 and abs(r_pave - 16) < 1e-9
    print("    -> exact. A modest overload does wildly disproportionate damage.")

    # ---- [2] Pareto over the simulated traffic ---------------------------
    trucks = generate_population(N_TRUCKS, seed=SEED)
    gvw = np.array([t["gvw"] for t in trucks])
    legal = np.array([t["legal"] for t in trucks])
    overloaded = gvw > legal
    dmg = stream_damage(gvw, W_REF, N_DESIGN, M_STEEL)

    share10, _ = pareto_concentration(dmg, 0.10)
    share20, _ = pareto_concentration(dmg, 0.20)
    ov_count = 100 * overloaded.mean()
    ov_damage = 100 * dmg[overloaded].sum() / dmg.sum()
    print(f"\n[2] PARETO over {N_TRUCKS} simulated trucks "
          f"({ov_count:.0f}% are overloaded)")
    print(f"    heaviest 10% of trucks  -> {100*share10:.0f}% of fatigue damage")
    print(f"    heaviest 20% of trucks  -> {100*share20:.0f}% of fatigue damage")
    print(f"    the {ov_count:.0f}% OVERLOADED trucks alone -> {ov_damage:.0f}% of "
          "all fatigue damage")
    print("    -> damage is concentrated in the heavy tail; B-WIM flags that tail.")

    # ---- [3] euro illustration -------------------------------------------
    c_legal = cost_per_pass(W_REF, W_REF, N_DESIGN, REPLACEMENT_COST, M_STEEL)
    c_2x = cost_per_pass(2 * W_REF, W_REF, N_DESIGN, REPLACEMENT_COST, M_STEEL)
    heaviest = trucks[int(np.argmax(gvw))]
    c_heavy = cost_per_pass(heaviest["gvw"], W_REF, N_DESIGN, REPLACEMENT_COST, M_STEEL)
    # annualise: scale the sampled stream to a year of heavy traffic
    heavy_per_day = 1370
    ann_factor = heavy_per_day * 365 / N_TRUCKS
    ann_cost_total = cost_per_pass(gvw, W_REF, N_DESIGN, REPLACEMENT_COST, M_STEEL).sum() * ann_factor
    ann_cost_overload = cost_per_pass(gvw[overloaded], W_REF, N_DESIGN, REPLACEMENT_COST,
                                      M_STEEL).sum() * ann_factor
    print("\n[3] EURO ILLUSTRATION  (replacement EUR {:.0f}M, design life {:.0f}M legal "
          "passes)".format(REPLACEMENT_COST/1e6, N_DESIGN/1e6))
    print(f"    one LEGAL pass         consumes EUR {c_legal:6.2f} of bridge life")
    print(f"    one 2x-OVERLOAD pass   consumes EUR {c_2x:6.2f}   ({c_2x/c_legal:.0f}x more)")
    print(f"    heaviest truck seen ({heaviest['gvw']/1e3:.0f} kN) consumes "
          f"EUR {c_heavy:6.2f} in a single crossing")
    print(f"    annual fatigue bill (this traffic mix): EUR {ann_cost_total/1e3:,.0f}k, "
          f"of which EUR {ann_cost_overload/1e3:,.0f}k\n      ({100*ann_cost_overload/ann_cost_total:.0f}%) "
          "is caused by the overloaded minority -- the revenue case for B-WIM enforcement.")

    # ---- figure ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        x = np.linspace(0.5, 2.5, 100)
        ax1.plot(x, x**M_STEEL, label="steel detail  (m=3)", lw=2.2, color="#1f77b4")
        ax1.plot(x, x**M_PAVEMENT, label="pavement  (m=4)", lw=2.2, color="#d62728")
        ax1.axvline(1, color="0.6", ls="--"); ax1.axhline(1, color="0.6", ls="--")
        ax1.scatter([2], [8], color="#1f77b4", zorder=5)
        ax1.scatter([2], [16], color="#d62728", zorder=5)
        ax1.annotate("2x load = 8x", (2, 8), textcoords="offset points", xytext=(-70, 6))
        ax1.annotate("2x load = 16x", (2, 16), textcoords="offset points", xytext=(-78, 4))
        ax1.set_xlabel("load / legal limit"); ax1.set_ylabel("fatigue damage per pass")
        ax1.set_title("Why overloads hurt: the power law"); ax1.legend(); ax1.grid(alpha=0.3)

        lx, ly = lorenz(dmg)
        ax2.plot([0, 1], [0, 1], "0.6", ls="--", label="if all trucks equal")
        ax2.plot(lx, ly, color="#d62728", lw=2.4, label="actual (cube law)")
        ax2.fill_between(lx, ly, lx, color="#d62728", alpha=0.12)
        ax2.set_xlabel("cumulative share of trucks (lightest → heaviest)")
        ax2.set_ylabel("cumulative share of fatigue damage")
        ax2.set_title(f"Damage concentrates in the heavy tail\n"
                      f"(top 10% of trucks = {100*share10:.0f}% of damage)")
        ax2.legend(loc="upper left"); ax2.grid(alpha=0.3)

        fig.tight_layout()
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "fatigue_economics.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
