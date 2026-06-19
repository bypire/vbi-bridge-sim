"""Traffic population generator for the BridgeTwin ops dashboard.

A realistic-ish stream of trucks crossing the bridge: a few standard axle
configurations, a gross-weight distribution with a genuine overloaded tail, and
randomised speeds. Each truck becomes a B-WIM "weighing event" downstream.

Deterministic (seeded) so the dashboard is reproducible. numpy only.
"""

from __future__ import annotations

import numpy as np

# Standard truck classes: axle spacings [m] behind the steer axle, the fraction
# of gross weight on each axle, and a legal gross-weight limit [kN].
TRUCK_CLASSES = {
    "2-axle rigid":  {"offsets": [0.0, 4.5],
                      "fracs": [0.35, 0.65], "legal_kN": 180.0, "prob": 0.30},
    "3-axle":        {"offsets": [0.0, 4.2, 5.5],
                      "fracs": [0.25, 0.375, 0.375], "legal_kN": 260.0, "prob": 0.30},
    "5-axle semi":   {"offsets": [0.0, 3.6, 4.9, 9.4, 10.7],
                      "fracs": [0.17, 0.205, 0.205, 0.21, 0.21],
                      "legal_kN": 400.0, "prob": 0.40},
}


def generate_population(n: int, seed: int = 7) -> list[dict]:
    """Generate `n` trucks. Each: class, axle loads/offsets [N], speed, GVW.

    Gross weight is a mixture: most trucks part-loaded around ~70% of the legal
    limit, plus a heavier cluster straddling the limit -> a realistic overloaded
    tail (a fraction genuinely exceed the legal GVW).
    """
    rng = np.random.default_rng(seed)
    names = list(TRUCK_CLASSES)
    probs = np.array([TRUCK_CLASSES[k]["prob"] for k in names])
    probs /= probs.sum()

    trucks = []
    t_clock = 0.0
    for i in range(n):
        cls = names[rng.choice(len(names), p=probs)]
        spec = TRUCK_CLASSES[cls]
        legal = spec["legal_kN"] * 1e3                  # N

        # load factor: 75% part-loaded cluster, 25% heavy cluster (some > legal)
        if rng.random() < 0.75:
            factor = rng.normal(0.72, 0.14)
        else:
            factor = rng.normal(1.04, 0.16)
        factor = float(np.clip(factor, 0.30, 1.7))
        gvw = legal * factor

        fracs = np.array(spec["fracs"])
        axle_loads = gvw * fracs
        speed = float(rng.uniform(16.0, 28.0))          # m/s (~58-100 km/h)
        t_clock += float(rng.uniform(2.0, 9.0))         # arrival gap [s]

        trucks.append({
            "id": i + 1,
            "cls": cls,
            "offsets": list(spec["offsets"]),
            "axle_loads": axle_loads,                   # N
            "gvw": gvw,                                 # N (true)
            "legal": legal,                             # N
            "speed": speed,
            "arrival": t_clock,
        })
    return trucks
