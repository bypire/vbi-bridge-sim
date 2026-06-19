"""D2 verification: extract the bridge frequency from the vehicle's acceleration.

Run directly:
    python solver/verify_d2.py

Ground truth: the bridge's true first natural frequency from the FEM eigenproblem
(V1). We drive a quarter-car across at a slow "scanning" speed, FFT its axle
acceleration, and check that a peak appears at the bridge frequency.

  [1] SMOOTH road: the bridge peak is clean -> extracted f vs true f.
  [2] ROUGH roads (ISO 8608 A/B/C): the peak gets buried; we track how its
      detectability (prominence) collapses. This is the wall D3 must climb.

Why slow: the usable record is only one crossing, so frequency resolution is
1 / (L/v). Drive-by frequency extraction therefore needs LOW speed — a real,
cited constraint, not a modelling shortcut.

Core stays numpy-only; matplotlib optional.
"""

import os

import numpy as np

from beam_fem import Beam, natural_frequencies
from vehicle import QuarterCar, integrate_coupled
from road import generate_iso8608
from driveby import acceleration_spectrum, peak_in_band, peak_prominence

L, E, I, M_BAR = 20.0, 2.1e11, 0.02, 2000.0
CAR = QuarterCar(m_s=9000.0, m_u=1000.0, k_s=2.0e6, k_t=1.0e7, c_s=6.0e4)
V_SCAN = 5.0          # m/s (18 km/h) — slow scanning pass


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    f_bridge = float(natural_frequencies(beam, 1)[0])
    f_body = CAR.natural_frequencies_rigid()[0]
    f_hop = CAR.natural_frequencies_rigid()[1]

    print("=" * 70)
    print("D2 VERIFICATION - bridge frequency from vehicle acceleration")
    print(f"  true bridge f1 (FEM)   = {f_bridge:.3f} Hz   <- ground truth")
    print(f"  vehicle body bounce    = {f_body:.3f} Hz")
    print(f"  vehicle wheel hop      = {f_hop:.3f} Hz")
    print(f"  scanning speed         = {V_SCAN} m/s; crossing {L/V_SCAN:.1f} s "
          f"-> freq resolution {V_SCAN/L:.3f} Hz")
    print("=" * 70)

    band = (f_bridge - 1.5, f_bridge + 1.5)   # search window around true f

    # --- [1] smooth road ---------------------------------------------------
    rs = integrate_coupled(beam, CAR, V_SCAN)
    f_axis, sp = acceleration_spectrum(rs.t, rs.a_u)
    f_ext, _ = peak_in_band(f_axis, sp, *band)
    prom = peak_prominence(f_axis, sp, f_bridge)
    print("\n[1] SMOOTH road")
    print(f"    extracted bridge f = {f_ext:.3f} Hz  (true {f_bridge:.3f} Hz, "
          f"err {100*abs(f_ext-f_bridge)/f_bridge:.2f}%)")
    print(f"    peak prominence    = {prom:.1f}x background")

    # --- [2] rough roads ---------------------------------------------------
    print("\n[2] ROUGH roads (ISO 8608); same extraction")
    print(f"    {'class':>6} {'extracted f[Hz]':>16} {'err[%]':>8} "
          f"{'prominence':>11}")
    specs = {"smooth": (f_axis, sp)}
    for cls in ["A", "B", "C"]:
        prof = generate_iso8608(length=L, road_class=cls, dx=0.02, seed=3)
        r = integrate_coupled(beam, CAR, V_SCAN, road=prof)
        fa, s = acceleration_spectrum(r.t, r.a_u)
        fe, _ = peak_in_band(fa, s, *band)
        pr = peak_prominence(fa, s, f_bridge)
        specs[cls] = (fa, s)
        print(f"    {cls:>6} {fe:>16.3f} {100*abs(fe-f_bridge)/f_bridge:>8.2f} "
              f"{pr:>11.1f}")
    print("\n    On a smooth road the bridge frequency falls right out of the\n"
          "    vehicle signal. As roughness rises the peak's prominence drops\n"
          "    toward the background -> raw acceleration alone is not enough; the\n"
          "    contact-point response (D3) is designed to recover it.")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(rs.t, rs.a_u, lw=0.8)
        ax1.set_xlabel("time t [s]"); ax1.set_ylabel("axle acceleration [m/s²]")
        ax1.set_title(f"Axle acceleration, smooth road (v={V_SCAN} m/s)")
        ax1.grid(alpha=0.3)

        for name, (fa, s) in specs.items():
            m = fa <= 25
            ax2.plot(fa[m], s[m], lw=1.0, label=name)
        ax2.axvline(f_bridge, color="g", ls="--", lw=1.5,
                    label=f"bridge {f_bridge:.2f} Hz")
        ax2.axvline(f_body, color="0.5", ls=":", lw=1);
        ax2.axvline(f_hop, color="0.5", ls=":", lw=1, label="vehicle modes")
        ax2.set_xlabel("frequency [Hz]"); ax2.set_ylabel("accel amplitude")
        ax2.set_yscale("log")
        ax2.set_title("Acceleration spectrum: bridge peak vs roughness")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")

        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "d2_bridge_frequency.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
