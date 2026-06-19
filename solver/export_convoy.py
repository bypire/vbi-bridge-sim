"""Export a long multi-span VIADUCT under heavy mixed traffic — the 'from afar,
real bridge / real traffic' monitoring view (BridgeBeat Act 5).

A 500 m continuous viaduct (10 x 50 m spans over piers) carries a stream of mixed
vehicles (cars + trucks, many on the deck at once). For the wide monitoring view we
use the QUASI-STATIC deflection each frame (solve K u = F(t) for the instantaneous
load set) -- exact static physics, and at this zoom the dynamic ripple is invisible
(dynamics are the subject of Acts 1-2). A running ledger accumulates the inspector's
balance sheet over the run: vehicles passed, trucks, overloads, fatigue cost.

    python -u solver/export_convoy.py   ->  output/convoy_data.js
"""

import json
import os
import numpy as np

from beam_fem import assemble, free_dofs, make_viaduct
from moving_load import bending_moment_at, moving_force_vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPAN, N_SPANS = 50.0, 10                       # 500 m viaduct
E, I, M_BAR = 2.1e11, 0.5, 15000.0
W_REF, N_DESIGN, REPLACEMENT = 260e3, 50e6, 40e6   # fatigue economics (see fatigue.py)
N_FRAMES = 220
SEED = 12


def main():
    beam = make_viaduct(SPAN, N_SPANS, E, I, M_BAR, elems_per_span=6)
    L = beam.L
    M, K = assemble(beam)
    free = free_dofs(beam)
    K_ff_inv = np.linalg.inv(K[np.ix_(free, free)])    # one factorization, reuse
    node_x = np.array([beam.node_x(i) for i in range(beam.n_nodes)])
    rng = np.random.default_rng(SEED)

    # ---- build a mixed traffic stream (mostly cars, some trucks) ----------
    N = 46
    veh = []
    t_enter = 0.0
    for k in range(N):
        t_enter += float(rng.exponential(1.1))         # Poisson-ish arrivals [s]
        is_truck = rng.random() < 0.32
        if is_truck:
            cls = rng.choice(["2-axle", "3-axle", "5-axle semi"], p=[.3, .3, .4])
            legal = {"2-axle": 180e3, "3-axle": 260e3, "5-axle semi": 400e3}[cls]
            P = float(rng.uniform(0.55, 1.35) * legal)  # some overloaded
            speed = float(rng.uniform(18, 25))
        else:
            cls = "car"; legal = 1e9
            P = float(rng.uniform(11e3, 24e3)); speed = float(rng.uniform(27, 34))
        veh.append({"cls": cls, "P": P, "legal": legal, "enter": t_enter,
                    "speed": speed, "truck": is_truck,
                    "overload": bool(P > legal),
                    "exit": t_enter + L / speed,
                    "euro": (P / W_REF) ** 3 / N_DESIGN * REPLACEMENT})
    total_time = max(v["exit"] for v in veh) * 1.01

    # ---- quasi-static deflection field per frame --------------------------
    times = np.linspace(0.0, total_time, N_FRAMES)
    frames = []
    n_on_peak = 0
    m_max = 1.0
    for t in times:
        Ff = np.zeros(beam.n_dof)
        xs = []
        for v in veh:
            x = v["speed"] * (t - v["enter"])
            if 0.0 <= x <= L:
                Ff += moving_force_vector(beam, -v["P"], x)   # downward
                xs.append(round(x, 1))
            else:
                xs.append(None)
        u = np.zeros(beam.n_dof)
        u[free] = K_ff_inv @ Ff[free]
        # bending moment along the deck (the real "structural utilization" we colour by)
        mvals = [bending_moment_at(beam, u, float(node_x[i])) for i in range(beam.n_nodes)]
        m_max = max(m_max, max(abs(m) for m in mvals))
        n_on = sum(1 for x in xs if x is not None)
        n_on_peak = max(n_on_peak, n_on)
        frames.append({"t": round(float(t), 2),
                       "w": [round(float(w), 6) for w in u[0::2]],
                       "m": [round(float(m), 1) for m in mvals],
                       "x": xs})

    n_truck = sum(v["truck"] for v in veh)
    n_over = sum(v["overload"] for v in veh)
    print(f"viaduct {L:.0f} m ({N_SPANS}x{SPAN:.0f} m), {N} vehicles "
          f"({n_truck} trucks, {n_over} overloaded), up to {n_on_peak} on the deck at once")
    print(f"  run window {total_time:.0f} s; total fatigue EUR "
          f"{sum(v['euro'] for v in veh):.0f} (trucks dominate)")

    data = {
        "meta": {"L": L, "span": SPAN, "n_spans": N_SPANS, "disp_mag": 300,
                 "m_max": round(float(m_max), 1), "n_on_peak": int(n_on_peak),
                 "piers": [round(float(node_x[s]), 1) for s in beam.support_nodes],
                 "node_x": [round(float(x), 1) for x in node_x],
                 "W_ref_kN": W_REF / 1e3, "n_design": N_DESIGN, "replacement": REPLACEMENT},
        "vehicles": [{"cls": v["cls"], "truck": v["truck"], "gvw": round(v["P"]/1e3, 1),
                      "legal": v["legal"]/1e3 if v["legal"] < 1e8 else None,
                      "overload": v["overload"], "enter": round(v["enter"], 2),
                      "exit": round(v["exit"], 2), "speed_kmh": round(v["speed"]*3.6, 0),
                      "euro": round(v["euro"], 3)} for v in veh],
        "frames": frames,
    }
    out = os.path.join(ROOT, "output", "convoy_data.js")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("// generated by solver/export_convoy.py (multi-span viaduct)\n")
        fh.write("const CONVOY = " + json.dumps(data) + ";\n")
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} kB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
