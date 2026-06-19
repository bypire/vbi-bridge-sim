"""V4 — export a coupled crossing for the web viewer.

Runs one showcase quarter-car crossing, snapshots the deflected shape + nodal
bending moments + vehicle motion frame by frame, also runs the B-WIM recovery on
the same signal, and writes everything to output/sim_data.js as

    const SIM_DATA = { ... };

The web page (web/index.html) includes that file with a <script src> tag, so it
opens by double-click — no server, no fetch, no CORS. Deflections are exported in
real metres; the viewer applies (and prints) a magnification factor, because real
bridge deflections are millimetric.

    python solver/export.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from beam_fem import Beam
from moving_load import bending_moment_at
from vehicle import QuarterCar, integrate_coupled
from bwim import moses_recover

# --- showcase scenario (same verified bridge + car) -------------------------
L, E, I, M_BAR = 20.0, 2.1e11, 0.02, 2000.0
CAR = QuarterCar(m_s=9000.0, m_u=1000.0, k_s=2.0e6, k_t=1.0e7, c_s=6.0e4, g=9.81)
SPEED = 20.0          # m/s (72 km/h)
X_GAUGE = L / 2.0
N_FRAMES_TARGET = 400  # animation frames
N_SERIES_TARGET = 800  # points in the time-series plots


def deflection_at(node_x: np.ndarray, node_w: np.ndarray, x: float) -> float:
    """Linear-interpolate the bridge deflection at an arbitrary x (low-poly)."""
    return float(np.interp(x, node_x, node_w))


def main() -> None:
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    free = list(range(beam.n_dof))  # for reconstruction we use the mask below

    crossing = beam.L / SPEED
    # pick a frame stride that yields ~N_FRAMES_TARGET frames
    res = integrate_coupled(beam, CAR, SPEED, moment_section=X_GAUGE)
    n_steps = len(res.t) - 1
    stride = max(1, n_steps // N_FRAMES_TARGET)

    res = integrate_coupled(beam, CAR, SPEED, moment_section=X_GAUGE,
                            frame_stride=stride)

    from beam_fem import free_dofs
    free_idx = free_dofs(beam)
    node_x = np.array([beam.node_x(k) for k in range(beam.n_nodes)])

    z_s0, z_u0 = CAR.static_equilibrium()

    # --- per-frame geometry ------------------------------------------------
    frames = []
    u_full = np.zeros(beam.n_dof)
    max_w = 0.0
    for i in range(len(res.frames_t)):
        u_full[:] = 0.0
        u_full[free_idx] = res.frames_u[i]
        node_w = u_full[0::2].copy()                      # transverse DOFs
        node_M = np.array([bending_moment_at(beam, u_full, x) for x in node_x])
        max_w = max(max_w, float(np.max(np.abs(node_w))))

        x_v = min(SPEED * res.frames_t[i], beam.L)
        frames.append({
            "t": res.frames_t[i],
            "w": node_w,                                  # m (real)
            "M": node_M,                                  # N m
            "xv": x_v,
            "wc": deflection_at(node_x, node_w, x_v),     # deck under the wheel
            "body": res.frames_zs[i] - z_s0,              # dynamic body bounce [m]
            "axle": res.frames_zu[i] - z_u0,              # dynamic axle bounce [m]
            "fc": res.frames_fc[i],                       # contact force [N]
        })

    # magnification so the peak deflection draws as ~12% of the span
    mag = (0.12 * beam.L) / max_w if max_w > 0 else 1.0

    # --- B-WIM recovery on the same crossing (the capstone payoff) ---------
    front = SPEED * res.t
    W_rec, _ = moses_recover(beam, X_GAUGE, front, res.moment, axle_offsets=(0.0,))
    W_rec = float(W_rec[0])
    W_true = CAR.weight
    bwim_err = 100.0 * (W_rec - W_true) / W_true

    # --- time series for the plots (down-sampled) --------------------------
    s = max(1, n_steps // N_SERIES_TARGET)
    series = {
        "t": res.t[::s],
        "w_mid": res.w_mid[::s],
        "fc": res.contact_force[::s],
        "M_mid": res.moment[::s],
    }

    # global moment range for a stable colour scale
    M_abs_max = max(float(np.max(np.abs(f["M"]))) for f in frames)

    data = {
        "meta": {
            "L": beam.L, "n_nodes": beam.n_nodes,
            "E": beam.E, "I": beam.I, "m_bar": beam.mass_per_length,
            "speed": SPEED, "speed_kmh": SPEED * 3.6,
            "weight": W_true, "daf": res.daf,
            "static_midspan": res.static_midspan,
            "mag": mag, "dt": res.dt, "crossing_time": crossing,
            "n_frames": len(frames), "gauge_x": X_GAUGE,
            "M_abs_max": M_abs_max,
            "W_true": W_true, "W_rec": W_rec, "bwim_error_pct": bwim_err,
            "car": {"m_s": CAR.m_s, "m_u": CAR.m_u, "k_s": CAR.k_s,
                    "k_t": CAR.k_t, "c_s": CAR.c_s},
        },
        "node_x": node_x,
        "frames": frames,
        "series": series,
    }

    # --- write output/sim_data.js -----------------------------------------
    def encode(o):
        """Compact JSON: round arrays/floats to keep the file small & readable."""
        if isinstance(o, np.ndarray):
            return [round(float(v), 8) for v in o]
        if isinstance(o, (np.floating, float)):
            return round(float(o), 8)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, dict):
            return {k: encode(v) for k, v in o.items()}
        if isinstance(o, list):
            return [encode(v) for v in o]
        return o

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "sim_data.js")
    payload = json.dumps(encode(data))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("// generated by solver/export.py — do not edit by hand\n")
        fh.write("const SIM_DATA = " + payload + ";\n")

    print(f"wrote {out}")
    print(f"  {len(frames)} frames, magnification x{mag:.0f}, "
          f"DAF={res.daf:.3f}")
    print(f"  B-WIM: true {W_true/1e3:.2f} kN -> recovered "
          f"{W_rec/1e3:.2f} kN ({bwim_err:+.2f}%)")
    print(f"  size: {os.path.getsize(out)/1024:.1f} kB")


if __name__ == "__main__":
    main()
