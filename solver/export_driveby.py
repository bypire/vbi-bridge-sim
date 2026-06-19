"""D5 — export the drive-by damage-detection demo for the web viewer.

Runs several slow scanning crossings at increasing damage severity (smooth road),
and for each one records:
  - animation frames (deflected deck + vehicle),
  - the contact-point acceleration spectrum (with its bridge peak),
  - the detected bridge frequency vs the true (FEM) frequency.

Writes output/driveby_data.js as `const DRIVEBY_DATA = {...}`. The web page
(web/driveby.html) lets you slide the damage severity and watch the spectral
peak march left while a health panel flips from HEALTHY to DAMAGE DETECTED.

    python solver/export_driveby.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from beam_fem import Beam, free_dofs, make_damaged_beam, natural_frequencies
from vehicle import QuarterCar, integrate_coupled
from driveby import acceleration_spectrum, contact_point_response, peak_in_band

# realistic 40 m overpass + 2% damping (Phase 3 scale)
L, E, I, M_BAR = 40.0, 2.1e11, 0.40, 12000.0
ZETA = 0.02
# soft-suspension scanning vehicle: body bounce ~1.2 Hz sits BELOW the bridge's
# ~2.6 Hz, leaving the bridge peak in a clean gap (heavier truck modes would
# overlap the bridge at this scale).
CAR = QuarterCar(m_s=9000.0, m_u=1000.0, k_s=5.0e5, k_t=1.0e7, c_s=3.0e4)
V_SCAN = 8.0
ZONE = (16.0, 24.0)
SEVERITIES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
N_FRAMES = 120
SPEC_BAND = (1.5, 5.0)        # Hz window shown in the spectrum panel (40 m bridge)
N_SPEC = 240                  # points in the displayed spectrum


def main():
    healthy = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    f0 = float(natural_frequencies(healthy, 1)[0])
    # search band isolates the bridge peak from the quasi-static bowl (<1 Hz)
    # and the soft vehicle body bounce (~1.2 Hz); damaged f drops toward ~2.2 Hz
    band = (1.8, f0 + 0.6)
    free_idx = free_dofs(healthy)
    node_x = np.array([healthy.node_x(k) for k in range(healthy.n_nodes)])
    z_s0, z_u0 = CAR.static_equilibrium()

    # damaged elements (for highlighting the zone in the viewer)
    damaged_elems = [e for e in range(healthy.n_elements)
                     if ZONE[0] <= (e + 0.5) * healthy.le <= ZONE[1]]

    scenarios = []
    max_w = 0.0
    spec_grid = np.linspace(*SPEC_BAND, N_SPEC)
    for sev in SEVERITIES:
        beam = healthy if sev == 0 else make_damaged_beam(healthy, *ZONE, sev)
        f_true = float(natural_frequencies(beam, 1)[0])

        # single run: full-resolution signal + frames in one pass (stride derived
        # internally from n_frames). Undamped here: stiffness-proportional damping
        # would shrink the RK4 step ~13x for negligible effect on the frequency
        # extraction (the bridge peak is just as clear). Damping realism is shown
        # in the main explorer demo; here the focus is the spectral peak shift.
        res = integrate_coupled(beam, CAR, V_SCAN, n_frames=N_FRAMES)

        # contact-point response -> spectrum -> detected frequency
        rc = contact_point_response(CAR, res.t, res.a_s, res.a_u)
        f_axis, sp = acceleration_spectrum(res.t, rc)
        f_det, _ = peak_in_band(f_axis, sp, *band)
        # resample the spectrum onto a fixed grid, normalised to its own peak
        amp = np.interp(spec_grid, f_axis, sp)
        amp = amp / (amp.max() + 1e-30)

        # animation frames
        frames = []
        u_full = np.zeros(healthy.n_dof)
        for i in range(len(res.frames_t)):
            u_full[:] = 0.0
            u_full[free_idx] = res.frames_u[i]
            node_w = u_full[0::2].copy()
            max_w = max(max_w, float(np.max(np.abs(node_w))))
            xv = min(V_SCAN * res.frames_t[i], L)
            frames.append({
                "t": res.frames_t[i],
                "w": node_w,
                "xv": xv,
                "wc": float(np.interp(xv, node_x, node_w)),
                "body": res.frames_zs[i] - z_s0,
                "axle": res.frames_zu[i] - z_u0,
            })

        scenarios.append({
            "severity": sev,
            "f_true": f_true,
            "f_detected": f_det,
            "drop_pct": 100.0 * (f0 - f_true) / f0,
            "spectrum": amp,
            "frames": frames,
        })
        print(f"  severity {sev:.0%}: f_true={f_true:.3f} Hz, "
              f"detected={f_det:.3f} Hz, drop={100*(f0-f_true)/f0:.1f}%")

    mag = (0.12 * L) / max_w if max_w > 0 else 1.0

    data = {
        "meta": {
            "L": L, "n_nodes": healthy.n_nodes, "mag": mag,
            "v_scan": V_SCAN, "v_kmh": V_SCAN * 3.6,
            "f_healthy": f0, "zone": list(ZONE), "damaged_elems": damaged_elems,
            "spec_band": list(SPEC_BAND), "road": "smooth",
            "n_scenarios": len(scenarios),
        },
        "node_x": node_x,
        "spec_freq": spec_grid,
        "scenarios": scenarios,
    }

    def enc(o):
        if isinstance(o, np.ndarray):
            return [round(float(v), 7) for v in o]
        if isinstance(o, (np.floating, float)):
            return round(float(o), 7)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, dict):
            return {k: enc(v) for k, v in o.items()}
        if isinstance(o, list):
            return [enc(v) for v in o]
        return o

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "driveby_data.js")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("// generated by solver/export_driveby.py — do not edit\n")
        fh.write("const DRIVEBY_DATA = " + json.dumps(enc(data)) + ";\n")

    print(f"\nwrote {out}  ({os.path.getsize(out)/1024:.0f} kB, "
          f"{len(scenarios)} severities, magnification x{mag:.0f})")


if __name__ == "__main__":
    main()
