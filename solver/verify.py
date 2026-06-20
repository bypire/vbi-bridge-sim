"""V1 verification: dynamic beam core checked against analytical references.

Run directly:
    python solver/verify.py

Two ground-truth checks for a uniform simply-supported beam:

  1. Static central point load P -> mid-span deflection  w = P L^3 / (48 E I).
  2. Natural frequencies  f_n = (n*pi/L)^2 * sqrt(E I / m_bar) / (2*pi).

If the FEM disagrees with these, the FEM is wrong.
"""

from beam_fem import Beam, natural_frequencies, static_solve

# --- verification beam (toy-ish but realistic steel girder) -----------------
L = 20.0        # span [m]
E = 2.1e11      # Young's modulus [Pa]
I = 0.02        # second moment of area [m^4]
M_BAR = 2000.0  # mass per length [kg/m]
P = 1.0e5       # central point load [N] (100 kN)


def analytical_static_midspan() -> float:
    return P * L**3 / (48.0 * E * I)


def analytical_frequencies(n_modes: int = 5):
    import math
    return [
        ((n * math.pi / L) ** 2) * math.sqrt(E * I / M_BAR) / (2.0 * math.pi)
        for n in range(1, n_modes + 1)
    ]


def main() -> None:
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)

    print("=" * 64)
    print("V1 VERIFICATION - simply-supported Euler-Bernoulli beam")
    print(f"  L={L} m, E={E:.3g} Pa, I={I} m^4, m_bar={M_BAR} kg/m, "
          f"{beam.n_elements} elements")
    print("=" * 64)

    # --- Check 1: static central deflection ---------------------------------
    u = static_solve(beam, loads={beam.mid_node: P})
    w_mid_fem = u[2 * beam.mid_node]
    w_mid_exact = analytical_static_midspan()
    rel = abs(w_mid_fem - w_mid_exact) / abs(w_mid_exact)
    print("\n[1] STATIC mid-span deflection under central P = "
          f"{P:.3g} N")
    print(f"    FEM        = {w_mid_fem:.6e} m")
    print(f"    analytical = {w_mid_exact:.6e} m   (P L^3 / 48 E I)")
    print(f"    rel. error = {rel:.2e}")
    assert rel < 1e-6, "static mid-span deflection must match P L^3 / 48 E I"

    # --- Check 2: natural frequencies ---------------------------------------
    f_fem = natural_frequencies(beam, n_modes=5)
    f_exact = analytical_frequencies(5)
    print("\n[2] NATURAL FREQUENCIES [Hz]")
    print(f"    {'mode':>4}  {'FEM':>12}  {'analytical':>12}  {'rel.err':>10}")
    for n in range(5):
        rel_n = abs(f_fem[n] - f_exact[n]) / f_exact[n]
        print(f"    {n + 1:>4}  {f_fem[n]:>12.4f}  {f_exact[n]:>12.4f}  "
              f"{rel_n:>10.2e}")

    assert abs(f_fem[0] - f_exact[0]) / f_exact[0] < 1e-3, \
        "fundamental frequency must match (n pi / L)^2 sqrt(EI / m_bar)"

    print("\n(Higher modes drift more - expected: a 20-element mesh resolves "
          "low modes best. V2 adds the moving load + Fryba check.)")


if __name__ == "__main__":
    main()
