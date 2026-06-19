"""D3 verification: contact-point (CP) response vs raw vehicle acceleration.

Run directly:
    python solver/verify_d3.py

The raw vehicle response is the bridge motion seen THROUGH the vehicle's transfer
function, so its spectrum is contaminated by the vehicle's own modes (body bounce
~2.2 Hz, wheel hop ~17 Hz) and the bridge peak is distorted. The contact-point
response reconstructs the motion at the tyre contact, which is free of that
vehicle filtering — the bridge mode stands out and the vehicle modes are
suppressed (Yang et al. 2020).

  [1] Reconstruction accuracy: CP rebuilt from accelerations vs the exact contact
      motion from the simulation (ground truth).
  [2] Bridge-frequency extraction: raw body acceleration vs CP response, smooth
      then rough — does CP recover a cleaner / more accurate bridge peak?

Core stays numpy-only; matplotlib optional.
"""

import os

import numpy as np

from beam_fem import Beam, natural_frequencies
from vehicle import QuarterCar, integrate_coupled
from road import generate_iso8608
from driveby import (
    acceleration_spectrum,
    contact_point_response,
    peak_in_band,
    peak_prominence,
    true_contact_response,
)

L, E, I, M_BAR = 20.0, 2.1e11, 0.02, 2000.0
CAR = QuarterCar(m_s=9000.0, m_u=1000.0, k_s=2.0e6, k_t=1.0e7, c_s=6.0e4)
V_SCAN = 5.0


def extract(t, sig, band, f_bridge):
    f, s = acceleration_spectrum(t, sig)
    fe, _ = peak_in_band(f, s, *band)
    pr = peak_prominence(f, s, f_bridge)
    return fe, pr, (f, s)


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    f_bridge = float(natural_frequencies(beam, 1)[0])
    f_body, f_hop = CAR.natural_frequencies_rigid()
    band = (f_bridge - 1.5, f_bridge + 1.5)

    print("=" * 70)
    print("D3 VERIFICATION - contact-point response vs raw vehicle acceleration")
    print(f"  true bridge f1 = {f_bridge:.3f} Hz; vehicle modes "
          f"{f_body:.2f} / {f_hop:.2f} Hz; scan {V_SCAN} m/s")
    print("=" * 70)

    # --- [1] reconstruction accuracy (smooth) ------------------------------
    rs = integrate_coupled(beam, CAR, V_SCAN)
    rc_true = true_contact_response(CAR, rs.t, rs.z_u, rs.contact_force)
    rc_rec = contact_point_response(CAR, rs.t, rs.a_s, rs.a_u)
    rel_rms = np.sqrt(np.mean((rc_rec - rc_true) ** 2)) / np.std(rc_true)
    fe_true, _, _ = extract(rs.t, rc_true, band, f_bridge)
    fe_rec, _, _ = extract(rs.t, rc_rec, band, f_bridge)
    print("\n[1] CP RECONSTRUCTION from accelerations vs exact contact motion")
    print(f"    rel. RMS error          = {rel_rms:.2e}")
    print(f"    bridge f from true r_c  = {fe_true:.3f} Hz")
    print(f"    bridge f from rebuilt   = {fe_rec:.3f} Hz  "
          f"(err vs true bridge {100*abs(fe_rec-f_bridge)/f_bridge:.2f}%)")

    # --- [2] raw accel vs CP, smooth + rough -------------------------------
    print("\n[2] BRIDGE-FREQUENCY EXTRACTION: raw body accel vs CP response")
    print(f"    {'road':>8} | {'raw f':>7} {'raw prom':>9} {'raw@body':>9} | "
          f"{'CP f':>7} {'CP prom':>8} {'CP@body':>8}")
    cases = [("smooth", None)]
    for cls in ["A", "B"]:
        cases.append((f"class {cls}",
                      generate_iso8608(length=L, road_class=cls, dx=0.02, seed=3)))

    plot_specs = {}
    for name, road in cases:
        r = integrate_coupled(beam, CAR, V_SCAN, road=road)
        # raw body acceleration
        fe_raw, pr_raw, sp_raw = extract(r.t, r.a_s, band, f_bridge)
        body_raw = peak_prominence(*sp_raw, f_body, half_width=0.5)
        # contact-point response
        rc = contact_point_response(CAR, r.t, r.a_s, r.a_u)
        fe_cp, pr_cp, sp_cp = extract(r.t, rc, band, f_bridge)
        body_cp = peak_prominence(*sp_cp, f_body, half_width=0.5)
        plot_specs[name] = (sp_raw, sp_cp)
        print(f"    {name:>8} | {fe_raw:>7.3f} {pr_raw:>9.1f} {body_raw:>9.1f} | "
              f"{fe_cp:>7.3f} {pr_cp:>8.1f} {body_cp:>8.1f}")
    print("\n    'prom' = bridge-peak prominence; '@body' = how strong the spurious\n"
          "    vehicle body-mode peak is. CP suppresses the body mode (@body -> ~1)\n"
          "    and keeps the bridge peak, which is why it is preferred for modal ID.")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(rs.t, rc_true * 1e3, "k-", lw=1.6, label="true contact motion")
        ax1.plot(rs.t, rc_rec * 1e3, "r--", lw=1.2,
                 label="reconstructed from accel")
        ax1.set_xlabel("time t [s]"); ax1.set_ylabel("contact-point disp. [mm]")
        ax1.set_title(f"CP reconstruction (smooth), rel.RMS {rel_rms:.1e}")
        ax1.legend(); ax1.grid(alpha=0.3)

        (sp_raw, sp_cp) = plot_specs["class A"]
        for (f, s), lab, c in [(sp_raw, "raw body accel", "tab:blue"),
                               (sp_cp, "contact-point", "tab:red")]:
            m = f <= 25
            ax2.plot(f[m], s[m] / np.max(s[m]), c=c, lw=1.1, label=lab)
        ax2.axvline(f_bridge, color="g", ls="--", lw=1.5, label="bridge")
        ax2.axvline(f_body, color="0.5", ls=":", lw=1)
        ax2.axvline(f_hop, color="0.5", ls=":", lw=1, label="vehicle modes")
        ax2.set_yscale("log")
        ax2.set_xlabel("frequency [Hz]"); ax2.set_ylabel("normalised amplitude")
        ax2.set_title("Spectra on class-A road: raw vs CP")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")

        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "d3_contact_point.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
