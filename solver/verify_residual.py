"""D6 verification: two-axle RESIDUAL beats road roughness (climbs D3's wall).

Run directly:
    python solver/verify_residual.py

D3 showed a single-axle contact-point response cannot recover the bridge
frequency on a rough road. The two-axle residual cancels the common road profile
(rear axle rides the same road as the front, delayed by spacing/speed), leaving
the bridge response. Ground truth: the FEM bridge frequency.

  raw single-axle CP   vs   two-axle residual,   on ISO 8608 class B & C roads.

Core stays numpy-only; matplotlib optional.
"""

import os

import numpy as np

from beam_fem import Beam, natural_frequencies
from vehicle import QuarterCar, integrate_two_axle
from road import generate_iso8608
from driveby import (
    acceleration_spectrum,
    contact_point_response,
    peak_in_band,
    peak_prominence,
    residual_contact_response,
)

L, E, I, M_BAR = 20.0, 2.1e11, 0.02, 2000.0
CAR = QuarterCar(m_s=9000.0, m_u=1000.0, k_s=2.0e6, k_t=1.0e7, c_s=6.0e4)
V_SCAN = 5.0
SPACING = 4.0


def analyse(t, sig, band, f_bridge):
    f, s = acceleration_spectrum(t, sig)
    fe, _ = peak_in_band(f, s, *band)
    pr = peak_prominence(f, s, f_bridge)
    return fe, pr, (f, s)


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    f_bridge = float(natural_frequencies(beam, 1)[0])
    band = (f_bridge - 1.5, f_bridge + 1.5)

    print("=" * 70)
    print("D6 VERIFICATION - two-axle residual vs single-axle CP on rough roads")
    print(f"  true bridge f1 = {f_bridge:.3f} Hz; scan {V_SCAN} m/s, "
          f"axle spacing {SPACING} m (tau = {SPACING/V_SCAN:.2f} s)")
    print("=" * 70)
    print(f"\n  {'road':>8} | {'single-axle CP':>22} | {'two-axle residual':>24}")
    print(f"  {'':>8} | {'f[Hz]':>9} {'err%':>5} {'prom':>5} | "
          f"{'f[Hz]':>9} {'err%':>5} {'prom':>6}")

    specs = {}
    for cls in ["smooth", "B", "C"]:
        road = None if cls == "smooth" else generate_iso8608(L + SPACING, cls,
                                                             dx=0.02, seed=3)
        r = integrate_two_axle(beam, CAR, SPACING, V_SCAN, road=road)
        # single-axle CP (front axle only)
        rc1 = contact_point_response(CAR, r.t, r.a_s1, r.a_u1)
        fe1, pr1, sp1 = analyse(r.t, rc1, band, f_bridge)
        # two-axle residual
        tv, resid = residual_contact_response(CAR, r.t, r.a_s1, r.a_u1,
                                              r.a_s2, r.a_u2, SPACING, V_SCAN)
        fe2, pr2, sp2 = analyse(tv, resid, band, f_bridge)
        specs[cls] = (sp1, sp2)
        e1 = 100 * abs(fe1 - f_bridge) / f_bridge
        e2 = 100 * abs(fe2 - f_bridge) / f_bridge
        print(f"  {cls:>8} | {fe1:>9.3f} {e1:>5.1f} {pr1:>5.1f} | "
              f"{fe2:>9.3f} {e2:>5.1f} {pr2:>6.1f}")

    print("\n  On rough roads the single-axle CP peak is buried (prominence ~1,\n"
          "  wrong frequency); the residual cancels the road and the bridge peak\n"
          "  re-emerges at the right frequency. (Cancellation is ideal here -\n"
          "  identical road under both wheels; real tyres/tracks make it partial.)")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, cls in zip(axes, ["B", "C"]):
            (sp1, sp2) = specs[cls]
            for (f, s), lab, c in [(sp1, "single-axle CP", "tab:blue"),
                                   (sp2, "two-axle residual", "tab:red")]:
                m = (f >= 2.5) & (f <= 9)
                ax.plot(f[m], s[m] / s[m].max(), c=c, lw=1.3, label=lab)
            ax.axvline(f_bridge, color="g", ls="--", lw=1.5, label="true bridge")
            ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("normalised amplitude")
            ax.set_title(f"ISO 8608 class {cls} road")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "d6_residual.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
