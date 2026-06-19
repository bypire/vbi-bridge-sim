"""P1 verification: realistic ~40 m highway overpass + Rayleigh damping.

Run directly:
    python solver/verify_p1.py

Phase 3 scales the toy 20 m beam up to a realistic simply-supported highway
overpass and adds energy dissipation (Rayleigh damping), which real bridges have
and our undamped model lacked.

  [1] Realistic sanity: first frequencies (vs the rule-of-thumb f1 ~ 100/L) and
      static deflection (vs the L/800 serviceability limit).
  [2] Damping ground truth: pluck the bridge and measure the decay's logarithmic
      decrement -> the damping ratio must come back equal to the target we set.
  [3] Damping reduces the dynamic amplification (DAF) — as expected.

Core stays numpy-only; matplotlib optional.
"""

import os

import numpy as np

from beam_fem import (Beam, assemble, free_dofs, natural_frequencies,
                      rayleigh_alpha_beta, static_solve)
from moving_load import integrate_moving_force

# --- realistic ~40 m steel-composite highway overpass -----------------------
L = 40.0          # span [m]
E = 2.1e11        # Pa (steel composite girder)
I = 0.40          # m^4 (large girder section)
M_BAR = 12000.0   # kg/m (deck + girders)
ZETA = 0.02       # target damping ratio (2%, typical highway bridge)
P_TRUCK = 3.0e5   # 300 kN test truck for static/DAF


def free_vibration_zeta(beam, zeta_target):
    """Pluck the bridge (static shape), let it decay, recover zeta by log-dec."""
    M, K = assemble(beam)
    free = free_dofs(beam)
    M_ff, K_ff = M[np.ix_(free, free)], K[np.ix_(free, free)]
    M_inv = np.linalg.inv(M_ff)
    A = M_inv @ K_ff
    a_ray, b_ray = rayleigh_alpha_beta(beam, zeta_target)
    omega_max = float(np.sqrt(np.max(np.linalg.eigvals(A).real)))
    dt = min(0.5 / omega_max, 2.2 / (b_ray * omega_max**2))
    nf = len(free)
    mid = free.index(2 * beam.mid_node)

    # initial shape = static deflection under a central load (mode-1 dominated)
    u0 = static_solve(beam, loads={beam.mid_node: P_TRUCK})[free]
    y = np.concatenate((u0, np.zeros(nf)))

    def deriv(y):
        u, v = y[:nf], y[nf:]
        return np.concatenate((v, -A @ u - (a_ray * v + b_ray * (A @ v))))

    T = 6.0
    n = int(T / dt)
    w = np.empty(n + 1)
    w[0] = y[mid]
    for i in range(1, n + 1):
        k1 = deriv(y); k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2); k4 = deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        w[i] = y[mid]

    # positive peaks -> logarithmic decrement
    peaks = [w[i] for i in range(1, n) if w[i] > w[i-1] and w[i] > w[i+1] and w[i] > 0]
    peaks = np.array(peaks)
    N = len(peaks) - 1
    delta = np.log(peaks[0] / peaks[N]) / N           # mean log decrement
    zeta_meas = delta / np.sqrt(4 * np.pi**2 + delta**2)
    return zeta_meas, dt, np.arange(n + 1) * dt, w


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)

    print("=" * 70)
    print("P1 VERIFICATION - realistic 40 m overpass + Rayleigh damping")
    print(f"  L={L} m, E={E:.2g} Pa, I={I} m^4, m_bar={M_BAR} kg/m, "
          f"target zeta={ZETA:.0%}")
    print("=" * 70)

    # --- [1] realistic sanity ---------------------------------------------
    f = natural_frequencies(beam, 3)
    static_mid = P_TRUCK * L**3 / (48 * E * I)
    print("\n[1] REALISTIC RANGES")
    print(f"    f1 = {f[0]:.2f} Hz  (rule of thumb 100/L = {100/L:.1f} Hz)")
    print(f"    f2 = {f[1]:.2f} Hz, f3 = {f[2]:.2f} Hz")
    print(f"    static mid-span under {P_TRUCK/1e3:.0f} kN = {static_mid*1e3:.1f} mm "
          f"(L/{L/static_mid:.0f}; serviceability ~ L/800 = {L/800*1e3:.0f} mm)")

    # --- [2] damping ground truth -----------------------------------------
    a_ray, b_ray = rayleigh_alpha_beta(beam, ZETA)
    zeta_meas, dt, tt, w = free_vibration_zeta(beam, ZETA)
    print("\n[2] RAYLEIGH DAMPING - free-vibration log-decrement check")
    print(f"    alpha={a_ray:.4f}, beta={b_ray:.3e}")
    print(f"    target zeta   = {ZETA:.4f}")
    print(f"    measured zeta = {zeta_meas:.4f}  "
          f"(rel.err {abs(zeta_meas-ZETA)/ZETA:.2%})")

    # --- [3] damping reduces DAF ------------------------------------------
    print("\n[3] DAMPING REDUCES DAF (300 kN moving force)")
    print(f"    {'v[m/s]':>7} {'DAF undamped':>13} {'DAF zeta=2%':>13}")
    for v in [15, 25, 40]:
        d0 = integrate_moving_force(beam, P_TRUCK, v).daf
        dz = integrate_moving_force(beam, P_TRUCK, v, damping_ratio=ZETA).daf
        print(f"    {v:>7.0f} {d0:>13.4f} {dz:>13.4f}")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(tt, w * 1e3, lw=1.0)
        env = w[0] * np.exp(-ZETA * 2 * np.pi * natural_frequencies(beam, 1)[0] * tt)
        ax.plot(tt, env * 1e3, "r--", lw=1.2, label=f"ζ={ZETA:.0%} envelope")
        ax.plot(tt, -env * 1e3, "r--", lw=1.2)
        ax.set_xlabel("time [s]"); ax.set_ylabel("mid-span deflection [mm]")
        ax.set_title(f"Free-vibration decay, 40 m overpass "
                     f"(measured ζ={zeta_meas:.3f})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output"); os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "p1_damping.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
