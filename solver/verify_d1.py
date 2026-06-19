"""D1 verification: ISO 8608 road profiles + the roughness-dominates problem.

Run directly:
    python solver/verify_d1.py

Two things to establish before any drive-by extraction:

  [1] Our synthesised profiles really are ISO 8608: the displacement PSD must
      follow Gd(n) = Gd(n0) (n/n0)^-2, i.e. slope ~ -2 in log-log and the right
      level for each class. (Ground truth = the standard we generated to.)
  [2] The core difficulty: on a rough road the road-induced vehicle vibration
      dwarfs the bridge-induced part. We quantify it (RMS acceleration smooth vs
      rough) — this is exactly what D2/D3 must see through.

Core stays numpy-only; matplotlib optional.
"""

import os

import numpy as np

from beam_fem import Beam
from vehicle import QuarterCar, integrate_coupled
from road import N0, estimate_psd, generate_iso8608, iso8608_gd_n0

L, E, I, M_BAR = 20.0, 2.1e11, 0.02, 2000.0
CAR = QuarterCar(m_s=9000.0, m_u=1000.0, k_s=2.0e6, k_t=1.0e7, c_s=6.0e4)
SPEED = 20.0


def main():
    print("=" * 70)
    print("D1 VERIFICATION - ISO 8608 road profiles + roughness vs bridge signal")
    print("=" * 70)

    # --- [1] verify the profiles ARE ISO 8608 ------------------------------
    # Level check uses the EXACT closed-form variance (the profile is built so
    # that var(h) = integral of Gd over the band), which is a stronger test than
    # a noisy periodogram level. The periodogram still confirms the -2 SLOPE.
    n_min, n_max = 0.011, 2.83
    print("\n[1] Profiles vs ISO 8608  Gd(n)=Gd(n0)(n/n0)^-2  (slope + RMS level)")
    print(f"    {'class':>5} {'target Gd(n0)':>14} {'PSD slope':>10} "
          f"{'RMS act[mm]':>12} {'RMS theory[mm]':>15} {'rel.err':>9}")
    for cls in ["A", "B", "C"]:
        prof = generate_iso8608(length=500.0, road_class=cls, dx=0.05, seed=7)
        n, psd = estimate_psd(prof.h, dx=0.05)
        band = (n >= 0.02) & (n <= 2.0)               # fit within ISO band
        slope, _ = np.polyfit(np.log(n[band]), np.log(psd[band]), 1)
        gd0 = iso8608_gd_n0(cls)
        rms_theory = np.sqrt(gd0 * N0**2 * (1.0 / n_min - 1.0 / n_max))
        rms_act = prof.h.std()
        rel = abs(rms_act - rms_theory) / rms_theory
        print(f"    {cls:>5} {gd0:>14.2e} {slope:>10.2f} {rms_act*1e3:>12.2f} "
              f"{rms_theory*1e3:>15.2f} {rel:>9.2e}")
    print("    (slope ~ -2 AND RMS matching the closed-form integral confirm the "
          "profiles are ISO 8608.)")

    # --- [2] roughness vs bridge in the vehicle response -------------------
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    print(f"\n[2] Vehicle sprung-mass acceleration over the crossing "
          f"(v={SPEED} m/s)")
    print(f"    {'road':>10} {'RMS accel [m/s^2]':>18} {'peak [m/s^2]':>14}")
    res_smooth = integrate_coupled(beam, CAR, SPEED)
    print(f"    {'smooth':>10} {res_smooth.a_s.std():>18.4f} "
          f"{np.max(np.abs(res_smooth.a_s)):>14.4f}")
    runs = {"smooth": res_smooth}
    for cls in ["A", "B", "C"]:
        prof = generate_iso8608(length=L, road_class=cls, dx=0.02, seed=3)
        r = integrate_coupled(beam, CAR, SPEED, road=prof)
        runs[cls] = r
        print(f"    {('class '+cls):>10} {r.a_s.std():>18.4f} "
              f"{np.max(np.abs(r.a_s)):>14.4f}")
    ratio = runs["B"].a_s.std() / max(res_smooth.a_s.std(), 1e-12)
    print(f"\n    class-B roughness inflates the RMS vehicle vibration ~{ratio:.0f}x "
          "over\n    the smooth (bridge-only) case. THAT is the drive-by challenge: "
          "the\n    bridge's signature is buried in road-induced motion (D2/D3).")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

        for cls in ["A", "B", "C"]:
            p = generate_iso8608(length=L, road_class=cls, dx=0.02, seed=3)
            ax1.plot(p.x, p.h * 1e3, lw=0.9, label=f"class {cls}")
        ax1.set_xlabel("position x [m]"); ax1.set_ylabel("road elevation [mm]")
        ax1.set_title("ISO 8608 road profiles"); ax1.legend(); ax1.grid(alpha=0.3)

        for cls in ["A", "B", "C"]:
            p = generate_iso8608(length=500.0, road_class=cls, dx=0.05, seed=7)
            n, psd = estimate_psd(p.h, 0.05)
            ax2.loglog(n, psd, lw=0.6, alpha=0.5)
            nn = np.array([0.01, 3.0])
            ax2.loglog(nn, iso8608_gd_n0(cls) * (nn / N0) ** -2, "k--", lw=1.2)
        ax2.set_xlabel("spatial freq n [cycles/m]")
        ax2.set_ylabel("Gd(n) [m³]")
        ax2.set_title("PSD vs ISO 8608 target (dashed)"); ax2.grid(alpha=0.3, which="both")

        ax3.plot(res_smooth.t, res_smooth.a_s, "b-", lw=0.8, label="smooth road")
        ax3.plot(runs["B"].t, runs["B"].a_s, "r-", lw=0.6, alpha=0.8,
                 label="class B road")
        ax3.set_xlabel("time t [s]"); ax3.set_ylabel("sprung accel [m/s²]")
        ax3.set_title("Vehicle acceleration: bridge vs road"); ax3.legend()
        ax3.grid(alpha=0.3)

        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "d1_road_roughness.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
