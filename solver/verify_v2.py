"""V2 verification: moving constant force vs the Frýba closed form.

Run directly:
    python solver/verify_v2.py

Ground truth: the undamped Frýba moving-force solution. We
check the FEM mid-span deflection time-history and the dynamic amplification
factor (DAF) against it, for one demo speed and across a speed sweep.

The Python CORE stays numpy-only. matplotlib is used HERE only as optional
verification tooling — if it is missing the script still prints all numbers and
draws an ASCII sketch of the time-history instead.
"""

import numpy as np

from beam_fem import Beam, static_solve
from moving_load import (
    fryba_deflection,
    fryba_speed_parameter,
    integrate_moving_force,
)

# --- same beam as V1 --------------------------------------------------------
L = 20.0        # span [m]
E = 2.1e11      # Young's modulus [Pa]
I = 0.02        # second moment of area [m^4]
M_BAR = 2000.0  # mass per length [kg/m]
P = 1.0e5       # moving force magnitude [N] (100 kN, ~ one heavy axle)

DEMO_SPEED = 30.0                       # m/s for the detailed time-history (~108 km/h)
SWEEP_SPEEDS = [10, 20, 30, 50, 80, 120, 180, 250]  # m/s for the DAF curve


def ascii_history(t, w_fem, w_fry, rows=18, cols=70):
    """Tiny terminal plot: FEM ('o') vs Frýba ('.') mid-span deflection."""
    lo = min(w_fem.min(), w_fry.min())
    hi = max(w_fem.max(), w_fry.max())
    if hi == lo:
        hi = lo + 1.0
    grid = [[" "] * cols for _ in range(rows)]

    def place(series, ch):
        idx = np.linspace(0, len(series) - 1, cols).astype(int)
        for c, i in enumerate(idx):
            r = int((hi - series[i]) / (hi - lo) * (rows - 1))
            grid[r][c] = ch

    place(w_fry, ".")
    place(w_fem, "o")
    print(f"    deflection [m]   top={hi:+.3e}  bottom={lo:+.3e}")
    for row in grid:
        print("    |" + "".join(row))
    print("    +" + "-" * cols + f"  t: 0 .. {t[-1]:.3f} s")
    print("    legend:  o = FEM   . = Frýba")


def run_one_speed(beam, speed, n_steps_per_crossing=2000):
    """FEM march + Frýba on the same time stamps, restricted to the crossing."""
    res = integrate_moving_force(beam, P, speed,
                                 n_steps_per_crossing=n_steps_per_crossing)
    # Compare only on the crossing window (Frýba closed form is valid there).
    mask = res.t <= res.crossing_time + 1e-12
    t = res.t[mask]
    w_fem = res.w_mid[mask]
    w_fry = fryba_deflection(beam, P, speed, t)
    daf_fry = float(np.max(np.abs(w_fry)) / abs(res.static_midspan))
    return res, t, w_fem, w_fry, daf_fry


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)

    print("=" * 70)
    print("V2 VERIFICATION - moving constant force vs Frýba closed form")
    print(f"  L={L} m, E={E:.3g} Pa, I={I} m^4, m_bar={M_BAR} kg/m, "
          f"{beam.n_elements} elements")
    print(f"  P={P:.3g} N, static mid-span PL^3/48EI = "
          f"{P * L**3 / (48 * E * I):.6e} m")
    print("=" * 70)

    # --- detailed single-speed check ---------------------------------------
    res, t, w_fem, w_fry, daf_fry = run_one_speed(beam, DEMO_SPEED)
    alpha = fryba_speed_parameter(beam, DEMO_SPEED)

    peak_fem = w_fem[np.argmax(np.abs(w_fem))]
    peak_fry = w_fry[np.argmax(np.abs(w_fry))]
    peak_rel = abs(peak_fem - peak_fry) / abs(peak_fry)
    scale = np.max(np.abs(w_fry))
    rms_rel = np.sqrt(np.mean((w_fem - w_fry) ** 2)) / scale

    print(f"\n[1] TIME-HISTORY at speed v = {DEMO_SPEED} m/s "
          f"({DEMO_SPEED * 3.6:.0f} km/h)")
    print(f"    speed parameter alpha = Omega_1/omega_1 = {alpha:.4f} "
          f"(alpha=1 is critical)")
    print(f"    crossing time L/v     = {res.crossing_time:.4f} s")
    print(f"    RK4 dt                = {res.dt:.3e} s  "
          f"(f_max,FEM = {res.f_max_hz:.1f} Hz -> stability cap)")
    print(f"    steps over crossing   = {len(t)}")
    print(f"    peak deflection FEM   = {peak_fem:+.6e} m")
    print(f"    peak deflection Frýba = {peak_fry:+.6e} m")
    print(f"    peak rel. error       = {peak_rel:.2e}")
    print(f"    RMS rel. error (hist) = {rms_rel:.2e}")
    print(f"    DAF  FEM={res.daf:.4f}   Frýba={daf_fry:.4f}   "
          f"(rel. err {abs(res.daf - daf_fry) / daf_fry:.2e})")

    # --- DAF vs speed sweep ------------------------------------------------
    print("\n[2] DAF vs SPEED  (FEM vs Frýba)")
    print(f"    {'v[m/s]':>7} {'km/h':>6} {'alpha':>7} "
          f"{'DAF_FEM':>9} {'DAF_Fryba':>10} {'rel.err':>9}")
    sweep = []
    for v in SWEEP_SPEEDS:
        r, tt, wf, wy, dfry = run_one_speed(beam, v)
        a = fryba_speed_parameter(beam, v)
        rel = abs(r.daf - dfry) / dfry
        sweep.append((v, a, r.daf, dfry))
        print(f"    {v:>7.0f} {v * 3.6:>6.0f} {a:>7.3f} "
              f"{r.daf:>9.4f} {dfry:>10.4f} {rel:>9.2e}")

    print("\n    (DAF rises with speed; the spread peaks near alpha~0.5-0.7, then\n"
          "     the load crosses too fast for the beam to fully respond. Realistic\n"
          "     truck speeds (alpha<<1) give DAF only slightly above 1 — correct.)")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(t, w_fry * 1e3, "k-", lw=2.5, label="Frýba (closed form)")
        ax1.plot(t, w_fem * 1e3, "r--", lw=1.5, label="FEM (RK4)")
        ax1.axhline(res.static_midspan * 1e3, color="gray", ls=":",
                    label="static PL³/48EI")
        ax1.invert_yaxis()  # deflection is downward; show it dipping down
        ax1.set_xlabel("time t [s]")
        ax1.set_ylabel("mid-span deflection w [mm]")
        ax1.set_title(f"Mid-span history  (v={DEMO_SPEED} m/s, "
                      f"α={alpha:.3f}, DAF={res.daf:.3f})")
        ax1.legend()
        ax1.grid(alpha=0.3)

        vs = [s[0] for s in sweep]
        ax2.plot(vs, [s[3] for s in sweep], "k-o", label="Frýba")
        ax2.plot(vs, [s[2] for s in sweep], "r--s", label="FEM")
        ax2.set_xlabel("vehicle speed v [m/s]")
        ax2.set_ylabel("DAF  (max dynamic / static)")
        ax2.set_title("Dynamic amplification vs speed")
        ax2.legend()
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        # Write to <project root>/output regardless of the current directory.
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "v2_fryba_check.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}  (FEM vs Frýba time-history + DAF curve)")
    except Exception as exc:  # matplotlib missing or headless issue
        print(f"\n[plot] matplotlib unavailable ({exc!r}); ASCII sketch instead:\n")
        ascii_history(t, w_fem, w_fry)


if __name__ == "__main__":
    main()
