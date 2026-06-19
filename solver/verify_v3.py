"""V3 verification: quarter-car coupled to the bridge.

Run directly:
    python solver/verify_v3.py

There is no simple closed form for the coupled problem (that was V2's Frýba),
so we pin V3 down with two checks that DO have ground truth, plus the headline
result:

  [1] Vehicle natural frequencies on a rigid road — the analytical 2x2
      eigenproblem vs frequencies recovered from a free-vibration time march
      (verifies the vehicle assembly AND the integrator).
  [2] Quasi-static limit — as speed -> 0 the coupled DAF must approach 1 and the
      peak mid-span deflection must approach the static W L^3/48EI.
  [3] DAF vs speed for the quarter-car, overlaid on the moving-constant-force
      (weight W) curve, to show what the vehicle dynamics add.

Core stays numpy-only; matplotlib is optional verification tooling here.
"""

import os

import numpy as np

from beam_fem import Beam
from moving_load import integrate_moving_force
from vehicle import QuarterCar, integrate_coupled

# --- same bridge as V1/V2 ---------------------------------------------------
L, E, I, M_BAR = 20.0, 2.1e11, 0.02, 2000.0

# --- quarter-car: one heavy axle (~98 kN), realistic truck-ish parameters ---
CAR = QuarterCar(
    m_s=9000.0,   # sprung (body share) [kg]
    m_u=1000.0,   # unsprung (axle+wheel) [kg]
    k_s=2.0e6,    # suspension stiffness [N/m]  -> body bounce ~2.4 Hz
    k_t=1.0e7,    # tyre stiffness [N/m]        -> wheel hop ~16 Hz
    c_s=6.0e4,    # suspension damping [N s/m]  (~25% critical)
    c_t=0.0,
    g=9.81,
)

DEMO_SPEED = 20.0
SWEEP_SPEEDS = [2, 5, 10, 20, 30, 50, 80, 120]


def free_vehicle_frequencies(car: QuarterCar):
    """Recover the vehicle's natural frequencies from an UNDAMPED free decay.

    Integrate the 2-DOF vehicle on a rigid road (w_c = 0, c_s = 0) from a small
    perturbation, FFT the response, and read off the two spectral peaks. This
    exercises the same equations the coupled integrator uses.
    """
    m_s, m_u, k_s, k_t = car.m_s, car.m_u, car.k_s, car.k_t
    z_s0, z_u0 = car.static_equilibrium()

    def deriv(y):
        zs, zu, vs, vu = y
        susp = k_s * (zs - zu)                 # undamped
        return np.array([vs, vu,
                         car.g - susp / m_s,
                         car.g + (susp - k_t * zu) / m_u])

    dt = 2.0e-4
    n = 40000  # 8 s
    # perturb both DOFs so both modes are excited
    y = np.array([z_s0 + 0.02, z_u0 - 0.01, 0.0, 0.0])
    zu = np.empty(n)
    for i in range(n):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        zu[i] = y[1] - z_u0                    # dynamic part of unsprung motion

    spec = np.abs(np.fft.rfft(zu * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, dt)
    # two largest local maxima
    peaks = [j for j in range(1, len(spec) - 1)
             if spec[j] > spec[j - 1] and spec[j] > spec[j + 1]]
    peaks.sort(key=lambda j: spec[j], reverse=True)
    top = sorted(freqs[peaks[:2]])
    return np.array(top)


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    W = CAR.weight

    print("=" * 70)
    print("V3 VERIFICATION - quarter-car coupled to the bridge")
    print(f"  bridge: L={L} m, E={E:.3g} Pa, I={I} m^4, m_bar={M_BAR} kg/m")
    print(f"  car: m_s={CAR.m_s} kg, m_u={CAR.m_u} kg, k_s={CAR.k_s:.2g} N/m, "
          f"k_t={CAR.k_t:.2g} N/m, c_s={CAR.c_s:.2g} N s/m")
    print(f"  static axle load W=(m_s+m_u)g = {W:.1f} N "
          f"({W/1e3:.1f} kN);  W L^3/48EI = {W*L**3/(48*E*I):.6e} m")
    print("=" * 70)

    # --- Check 1: vehicle natural frequencies ------------------------------
    f_eig = CAR.natural_frequencies_rigid()
    f_fft = free_vehicle_frequencies(CAR)
    print("\n[1] VEHICLE NATURAL FREQUENCIES on rigid road [Hz]")
    print(f"    {'mode':>12}  {'analytic(2x2)':>13}  {'from march(FFT)':>15}  "
          f"{'rel.err':>9}")
    names = ["body bounce", "wheel hop"]
    for i in range(2):
        rel = abs(f_fft[i] - f_eig[i]) / f_eig[i]
        print(f"    {names[i]:>12}  {f_eig[i]:>13.4f}  {f_fft[i]:>15.4f}  "
              f"{rel:>9.2e}")
    print("    (FFT resolution ~0.125 Hz over the 8 s record sets the floor.)")

    # --- demo crossing + sweep --------------------------------------------
    res = integrate_coupled(beam, CAR, DEMO_SPEED)
    print(f"\n[demo] crossing at v={DEMO_SPEED} m/s ({DEMO_SPEED*3.6:.0f} km/h): "
          f"dt={res.dt:.2e}s, f_max,FEM={res.f_max_hz:.0f} Hz, "
          f"steps={len(res.t)}")
    print(f"       peak mid-span = {np.max(np.abs(res.w_mid)):.6e} m, "
          f"DAF = {res.daf:.4f}")
    print(f"       contact force: static W={W:.0f} N, "
          f"range [{res.contact_force.min():.0f}, "
          f"{res.contact_force.max():.0f}] N "
          f"(+/-{100*(res.contact_force.max()-W)/W:.1f}% of W)")

    print("\n[2,3] DAF vs SPEED - quarter-car vs moving constant force (= W)")
    print(f"    {'v[m/s]':>7} {'km/h':>6} {'DAF_car':>8} {'DAF_force':>10} "
          f"{'peak_mid[m]':>12}")
    sweep = []
    for v in SWEEP_SPEEDS:
        rc = integrate_coupled(beam, CAR, v)
        rf = integrate_moving_force(beam, W, v)  # constant force = weight
        sweep.append((v, rc.daf, rf.daf, np.max(np.abs(rc.w_mid))))
        print(f"    {v:>7.0f} {v*3.6:>6.0f} {rc.daf:>8.4f} {rf.daf:>10.4f} "
              f"{np.max(np.abs(rc.w_mid)):>12.6e}")
    print(f"\n    Quasi-static check: at v={SWEEP_SPEEDS[0]} m/s DAF_car="
          f"{sweep[0][1]:.4f} -> 1 and peak_mid={sweep[0][3]:.6e} m vs static "
          f"{res.static_midspan:.6e} m.")
    print("    The vehicle's suspension/inertia make DAF_car differ from the\n"
          "    bare moving force; the gap is the vehicle-dynamics contribution.")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

        rf_demo = integrate_moving_force(beam, W, DEMO_SPEED)
        mask = res.t <= res.crossing_time + 1e-12
        ax1.plot(rf_demo.t[rf_demo.t <= res.crossing_time + 1e-12],
                 rf_demo.w_mid[rf_demo.t <= res.crossing_time + 1e-12] * 1e3,
                 "b-", lw=1.5, label="moving force W")
        ax1.plot(res.t[mask], res.w_mid[mask] * 1e3, "r-", lw=1.8,
                 label="quarter-car")
        ax1.axhline(res.static_midspan * 1e3, color="gray", ls=":",
                    label="static W L³/48EI")
        ax1.invert_yaxis()
        ax1.set_xlabel("time t [s]")
        ax1.set_ylabel("mid-span deflection [mm]")
        ax1.set_title(f"Mid-span (v={DEMO_SPEED} m/s, DAF={res.daf:.3f})")
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(res.t[mask], res.contact_force[mask] / 1e3, "g-", lw=1.5)
        ax2.axhline(W / 1e3, color="gray", ls=":", label="static weight W")
        ax2.set_xlabel("time t [s]")
        ax2.set_ylabel("contact force F_c [kN]")
        ax2.set_title("Wheel–bridge contact force\n(its variation is what B-WIM reads)")
        ax2.legend(); ax2.grid(alpha=0.3)

        vs = [s[0] for s in sweep]
        ax3.plot(vs, [s[2] for s in sweep], "b-o", label="moving force W")
        ax3.plot(vs, [s[1] for s in sweep], "r--s", label="quarter-car")
        ax3.set_xlabel("vehicle speed v [m/s]")
        ax3.set_ylabel("DAF")
        ax3.set_title("DAF vs speed")
        ax3.legend(); ax3.grid(alpha=0.3)

        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "v3_quarter_car.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
