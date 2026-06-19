"""Export the BridgeTwin ops-dashboard data: a stream of B-WIM weighing events.

For each truck we form the bending-moment signal the gauge would see and recover
the gross weight with Moses' B-WIM least squares — exactly what an instrumented
bridge does. The dashboard (web/dashboard.html) plays the stream: live weighing,
overload flags, KPIs, weight histogram and accuracy scatter.

Forward model (FAST, vectorised): the measured moment is the sum of each axle's
load times the section's static influence line, with a small dynamic ripple at
the bridge frequency (the amplification the static model omits) plus measurement
noise. This is the standard B-WIM forward relation and lets us weigh 60 trucks in
under a second — the per-truck FULL dynamic simulation lives in the verify_*
scripts; here we want throughput. A wall-clock budget guard makes sure the export
can never run away.

    python -u solver/export_traffic.py   ->  output/traffic_data.js
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from beam_fem import Beam, natural_frequencies
from bwim import (bayesian_axle_loads, gross_weight_posterior,
                  moment_influence_line, prob_exceed)
from traffic import generate_population

# Dynamic/model error the static influence line cannot remove. This is NOT a
# free fudge factor: verify_coverage.py measures it directly by running the FULL
# coupled dynamic FEM through the same Moses/Bayes inverse and finds the RMS
# weight bias over the highway speed band (58-144 km/h) is ~3.1%. We use that
# calibrated value as the per-pass systematic error AND fold a slightly
# conservative 3.5% into the credible interval (MODEL_UNC) so the dashboard's
# stated uncertainty is honest about model error, not just electrical noise.
DYN_MODEL_ERR = 0.031      # calibrated from verify_coverage.py (RMS dynamic bias)
MODEL_UNC = 0.035          # CI inflation: DYN_MODEL_ERR rounded up for margin

# realistic 40 m overpass (Phase 3 scale)
L, E, I, M_BAR = 40.0, 2.1e11, 0.40, 12000.0
X_GAUGE = L / 2.0
N_TRUCKS = 60
NOISE_FRAC = 0.04          # measurement noise, fraction of peak moment
RIPPLE = 0.05              # dynamic ripple amplitude (static-model error)
N_SAMPLES = 400            # moment samples per crossing
SEED = 7
TIME_BUDGET_S = 30.0       # hard guard: never run longer than this


def measured_moment(beam, loads, offsets, speed, f1, rng):
    """Gauge bending-moment history for a multi-axle truck (fast, analytic).

    Static influence-line superposition + a bridge-frequency ripple (the dynamic
    amplification the static model leaves out) + measurement noise.
    """
    max_off = float(np.max(offsets))
    crossing = (beam.L + max_off) / speed
    t = np.linspace(0.0, crossing, N_SAMPLES)
    front = speed * t
    m = np.zeros_like(t)
    for P, off in zip(loads, offsets):
        m += P * moment_influence_line(beam, X_GAUGE, front - off)
    ripple = 1.0 + RIPPLE * np.sin(2 * np.pi * f1 * t + rng.uniform(0, 2 * np.pi))
    m = m * ripple
    # per-pass systematic error the static influence line cannot remove (dynamic
    # amplification, calibration drift, temperature). Std = DYN_MODEL_ERR, the
    # value CALIBRATED in verify_coverage.py from the full coupled dynamic sim --
    # not a hand-tuned knob. This is the dominant real-world B-WIM error and the
    # reason detection is imperfect near the limit.
    m = m * (1.0 + rng.normal(0.0, DYN_MODEL_ERR))
    m = m + rng.normal(0.0, NOISE_FRAC * np.max(np.abs(m)), size=m.shape)
    return t, front, m


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    f1 = float(natural_frequencies(beam, 1)[0])
    rng = np.random.default_rng(SEED + 1)
    trucks = generate_population(N_TRUCKS, seed=SEED)

    t0 = time.time()
    vehicles, abs_errs = [], []
    for tk in trucks:
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"  [guard] time budget {TIME_BUDGET_S}s hit at truck "
                  f"{tk['id']}; writing the {len(vehicles)} done so far.", flush=True)
            break
        offs = np.array(tk["offsets"])
        _, front, m = measured_moment(beam, tk["axle_loads"], offs, tk["speed"], f1, rng)
        # Bayesian recovery: posterior gross weight + uncertainty
        mean, cov, _ = bayesian_axle_loads(beam, X_GAUGE, front, m, axle_offsets=offs)
        mu_g, sd_g = gross_weight_posterior(mean, cov)
        sd_tot = float(np.sqrt(sd_g**2 + (MODEL_UNC * mu_g)**2))   # + model error
        p_over = prob_exceed(mu_g, sd_tot, tk["legal"])
        rec_gvw = mu_g
        err = 100.0 * (rec_gvw - tk["gvw"]) / tk["gvw"]
        abs_errs.append(abs(err))
        flagged = p_over > 0.5
        vehicles.append({
            "id": tk["id"], "cls": tk["cls"], "n_axles": len(offs),
            "arrival": tk["arrival"], "speed_kmh": tk["speed"] * 3.6,
            "true_gvw": tk["gvw"] / 1e3, "rec_gvw": rec_gvw / 1e3,
            "ci_kN": 1.96 * sd_tot / 1e3,            # 95% half-width
            "p_over": p_over,
            "legal": tk["legal"] / 1e3, "err_pct": err,
            "overload": bool(tk["gvw"] > tk["legal"]),
            "flagged": bool(flagged),
            "uncertain": bool(0.15 < p_over < 0.85),
        })
        print(f"  truck {tk['id']:>2}/{N_TRUCKS} {tk['cls']:<13} "
              f"true {tk['gvw']/1e3:6.1f} rec {rec_gvw/1e3:6.1f}±{1.96*sd_tot/1e3:4.1f} kN "
              f"P(over)={p_over:.2f}{'  OVERLOAD' if tk['gvw'] > tk['legal'] else ''}",
              flush=True)

    n = len(vehicles)
    n_over = sum(v["overload"] for v in vehicles)
    n_flag = sum(v["flagged"] for v in vehicles)
    correct = sum(v["overload"] == v["flagged"] for v in vehicles)
    kpis = {"n_trucks": n, "pct_overloaded": 100.0 * n_over / max(n, 1),
            "n_overloaded": n_over,
            "heaviest_kN": max((v["true_gvw"] for v in vehicles), default=0),
            "mean_abs_err": float(np.mean(abs_errs)) if abs_errs else 0.0,
            "detection_acc": 100.0 * correct / max(n, 1)}

    data = {"meta": {"bridge_id": "BR-40N (demo)", "span_L": L, "gauge_x": X_GAUGE,
                     "E": E, "I": I, "f1": f1, "noise_pct": NOISE_FRAC * 100},
            "kpis": kpis, "vehicles": vehicles}

    def enc(o):
        if isinstance(o, np.ndarray): return [round(float(v), 5) for v in o]
        if isinstance(o, (np.floating, float)): return round(float(o), 5)
        if isinstance(o, (np.integer, int)): return int(o)
        if isinstance(o, dict): return {k: enc(v) for k, v in o.items()}
        if isinstance(o, list): return [enc(v) for v in o]
        return o

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "output"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "traffic_data.js")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("// generated by solver/export_traffic.py\n")
        fh.write("const TRAFFIC_DATA = " + json.dumps(enc(data)) + ";\n")

    print(f"\n  {n} trucks | {n_over} overloaded ({kpis['pct_overloaded']:.0f}%) | "
          f"flagged {n_flag} | detection {kpis['detection_acc']:.0f}% | "
          f"mean |err| {kpis['mean_abs_err']:.1f}% | {time.time()-t0:.1f}s", flush=True)
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} kB)", flush=True)


if __name__ == "__main__":
    main()
