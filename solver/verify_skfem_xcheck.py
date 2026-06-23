"""Independent cross-check of the beam core against scikit-fem.

The static Euler-Bernoulli deflection from this project's own beam_fem (a 2-node
cubic-Hermite element, assembled by hand) is compared, on the same mesh and the
same load case, against scikit-fem's ElementLineHermite. scikit-fem is a
maintained finite-element library whose mesh I/O test suite I also fixed
(kinnala/scikit-fem PR #1213, merged). Both are cubic-Hermite formulations, so
for a central point load on a simply-supported span (deflection piecewise cubic,
with a node at mid-span) each is exact; they should agree with each other and
with the closed form w = P L^3 / (48 E I) to machine precision.

Run:
    python solver/verify_skfem_xcheck.py
"""

import numpy as np

from beam_fem import Beam, static_solve

from skfem import (Basis, BilinearForm, ElementLineHermite, MeshLine, condense,
                   solve)
from skfem.helpers import dd, ddot

# --- same realistic ~40 m overpass as verify_p1.py --------------------------
L = 40.0          # span [m]
E = 2.1e11        # Pa
I = 0.40          # m^4
M_BAR = 12000.0   # kg/m (irrelevant for the static check, kept for the Beam)
P = 3.0e5         # central point load [N]
N_EL = 20         # even, so mid-span lands on a node


def beam_fem_midspan() -> float:
    """Mid-span deflection from this project's hand-assembled beam core."""
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=N_EL)
    mid = beam.mid_node
    u = static_solve(beam, loads={mid: P})
    return float(u[2 * mid])          # DOF 2k = transverse deflection w_k


def skfem_midspan() -> float:
    """Mid-span deflection from scikit-fem's cubic-Hermite line element."""
    EI = E * I
    mesh = MeshLine(np.linspace(0.0, L, N_EL + 1))
    basis = Basis(mesh, ElementLineHermite())

    @BilinearForm
    def bending(u, v, w):
        # Euler-Bernoulli weak form: integral of EI * w'' * v'' dx.
        # In 1D, dd(.) is the (1x1) Hessian; ddot collapses it to the scalar.
        return EI * ddot(dd(u), dd(v))

    K = bending.assemble(basis)

    value_dofs = basis.nodal_dofs[0]          # function-value DOF at each node
    mid = N_EL // 2
    f = np.zeros(basis.N)
    f[value_dofs[mid]] = P                     # point load at mid-span node

    # simply supported: w = 0 at the two end nodes (rotations free)
    clamped = np.array([value_dofs[0], value_dofs[N_EL]], dtype=np.int64)
    u = solve(*condense(K, f, D=clamped))
    return float(u[value_dofs[mid]])


def main() -> None:
    w_analytic = P * L**3 / (48.0 * E * I)
    w_mine = beam_fem_midspan()
    w_skfem = skfem_midspan()

    def rel(a, b):
        return abs(a - b) / abs(b)

    print("Independent beam cross-check (simply-supported, central point load)")
    print(f"  span L            = {L} m,  EI = {E*I:.3e} N m^2,  P = {P:.1f} N")
    print(f"  closed form       w = P L^3 / 48 EI = {w_analytic:.9e} m")
    print(f"  beam_fem (mine)   w = {w_mine:.9e} m")
    print(f"  scikit-fem        w = {w_skfem:.9e} m")
    print(f"  rel. err  mine   vs closed form = {rel(w_mine, w_analytic):.3e}")
    print(f"  rel. err  skfem  vs closed form = {rel(w_skfem, w_analytic):.3e}")
    print(f"  rel. err  mine   vs scikit-fem  = {rel(w_mine, w_skfem):.3e}")

    tol = 1e-9
    ok = (rel(w_mine, w_skfem) < tol and rel(w_mine, w_analytic) < tol)
    print("  RESULT:", "PASS" if ok else "MISMATCH (report the difference honestly)")


if __name__ == "__main__":
    main()
