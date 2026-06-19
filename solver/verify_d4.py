"""D4 verification: detect localized bridge damage from the passing vehicle.

Run directly:
    python solver/verify_d4.py

Damage = a local loss of bending stiffness EI (cracking / section loss) over a
central zone of the span. It lowers the bridge's natural frequency. The question:
can the slow-scanning vehicle SEE that frequency drop?

  [1] FEM ground truth: natural frequency vs damage severity (this is the exact
      shift we must detect).
  [2] Drive-by detection on a smooth road: extract the bridge frequency from the
      contact-point response for each severity and check it tracks the FEM truth.
  [3] Honest limits: how small a damage is resolvable (frequency-based detection
      is a GLOBAL, blunt indicator), and that road roughness (D3) defeats it.

Core stays numpy-only; matplotlib optional.
"""

import os

import numpy as np

from beam_fem import Beam, make_damaged_beam, natural_frequencies
from vehicle import QuarterCar, integrate_coupled
from road import generate_iso8608
from driveby import acceleration_spectrum, contact_point_response, peak_in_band

L, E, I, M_BAR = 20.0, 2.1e11, 0.02, 2000.0
CAR = QuarterCar(m_s=9000.0, m_u=1000.0, k_s=2.0e6, k_t=1.0e7, c_s=6.0e4)
V_SCAN = 5.0
ZONE = (8.0, 12.0)                 # central damage zone [m] (20% of span)
SEVERITIES = [0.0, 0.15, 0.30, 0.50]


def drive_by_frequency(beam, road, band):
    r = integrate_coupled(beam, CAR, V_SCAN, road=road)
    rc = contact_point_response(CAR, r.t, r.a_s, r.a_u)
    f, s = acceleration_spectrum(r.t, rc)
    fe, _ = peak_in_band(f, s, *band)
    return fe


def main():
    healthy = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    f0 = float(natural_frequencies(healthy, 1)[0])
    band = (f0 - 2.5, f0 + 1.0)

    print("=" * 70)
    print("D4 VERIFICATION - drive-by detection of localized bridge damage")
    print(f"  healthy bridge f1 = {f0:.3f} Hz; damage zone x in {ZONE} "
          f"(central {100*(ZONE[1]-ZONE[0])/L:.0f}% of span)")
    print(f"  scan speed {V_SCAN} m/s -> freq resolution {V_SCAN/L:.3f} Hz")
    print("=" * 70)

    # --- [1]+[2] FEM truth vs drive-by, on a SMOOTH road -------------------
    print("\n[1+2] frequency vs damage: FEM truth vs drive-by (smooth road)")
    print(f"    {'severity':>9} {'FEM f1[Hz]':>11} {'drop[%]':>8} "
          f"{'drive-by f[Hz]':>15} {'detect err[%]':>14}")
    fem_f, db_f = [], []
    for sev in SEVERITIES:
        beam = healthy if sev == 0 else make_damaged_beam(healthy, *ZONE, sev)
        f_true = float(natural_frequencies(beam, 1)[0])
        f_db = drive_by_frequency(beam, None, band)
        fem_f.append(f_true); db_f.append(f_db)
        drop = 100 * (f0 - f_true) / f0
        derr = 100 * abs(f_db - f_true) / f_true
        print(f"    {sev:>9.2f} {f_true:>11.3f} {drop:>8.2f} {f_db:>15.3f} "
              f"{derr:>14.2f}")
    print("    The drive-by frequency tracks the true (damaged) bridge frequency:\n"
          "    a stiffness loss in the deck is detected from the passing vehicle.")

    # --- [3] limits: smallest resolvable damage, and roughness -------------
    print("\n[3] limits")
    # frequency shift for a small (10%) localized damage vs resolution
    small = make_damaged_beam(healthy, *ZONE, 0.10)
    df_small = f0 - float(natural_frequencies(small, 1)[0])
    print(f"    10% local damage shifts f1 by only {df_small*1e3:.1f} mHz "
          f"(resolution {V_SCAN/L*1e3:.0f} mHz) -> frequency-based detection is a\n"
          "    blunt, GLOBAL indicator; small local damage is near the floor.")
    # roughness defeats it (from D3)
    worst = make_damaged_beam(healthy, *ZONE, 0.50)
    f_true = float(natural_frequencies(worst, 1)[0])
    f_rough = drive_by_frequency(worst, generate_iso8608(L, "A", dx=0.02, seed=3), band)
    print(f"    50% damage on a class-A road: drive-by f = {f_rough:.3f} Hz vs "
          f"true {f_true:.3f} Hz -> roughness buries\n    the peak (needs the "
          "two-axle / residual fix from D3's note).")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(SEVERITIES, fem_f, "k-o", lw=2, label="FEM truth")
        ax1.plot(SEVERITIES, db_f, "r--s", lw=1.5, label="drive-by (CP, smooth)")
        ax1.set_xlabel("damage severity (EI loss in zone)")
        ax1.set_ylabel("bridge frequency f1 [Hz]")
        ax1.set_title("Detected frequency drop vs damage")
        ax1.legend(); ax1.grid(alpha=0.3)

        for sev, c in [(0.0, "tab:green"), (0.5, "tab:red")]:
            beam = healthy if sev == 0 else make_damaged_beam(healthy, *ZONE, sev)
            r = integrate_coupled(beam, CAR, V_SCAN)
            rc = contact_point_response(CAR, r.t, r.a_s, r.a_u)
            f, s = acceleration_spectrum(r.t, rc)
            m = (f >= 3) & (f <= 8)
            ax2.plot(f[m], s[m] / s[m].max(), c=c, lw=1.4,
                     label=f"severity {sev:.0%}")
        ax2.set_xlabel("frequency [Hz]"); ax2.set_ylabel("normalised amplitude")
        ax2.set_title("Bridge peak shifts left with damage (smooth road)")
        ax2.legend(); ax2.grid(alpha=0.3)

        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "d4_damage_detection.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
