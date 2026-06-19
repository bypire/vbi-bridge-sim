"""Euler-Bernoulli beam finite-element core for the VBI simulator.

This is the dynamic step up from project #1's truss stiffness method: instead of
axial bar elements (1 DOF/node direction), we use 2-node beam elements with
2 DOF per node — transverse deflection w and rotation theta — and we now also
build a consistent MASS matrix, because the next milestones march the system in
time (M u'' + C u' + K u = F(t)).

Kept deliberately explicit and commented; numpy only.

DOF convention
--------------
Node k has DOFs [2k] = w_k (transverse), [2k+1] = theta_k (rotation).
Element DOF order: [w_i, theta_i, w_j, theta_j].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Beam:
    """A uniform, simply-supported Euler-Bernoulli beam. Consistent SI units.

    L : span [m]
    E : Young's modulus [Pa]
    I : second moment of area [m^4]
    mass_per_length : mass per unit length, rho*A [kg/m]
    n_elements : number of equal beam elements (use an even number so the
        mid-span lands on a node, which makes the analytical checks exact).
    """

    L: float
    E: float
    I: float
    mass_per_length: float
    n_elements: int = 20
    # per-element bending-stiffness multipliers (1.0 = healthy). A value < 1 in
    # some elements models LOCAL DAMAGE (loss of EI from cracking/section loss).
    # None means uniform/healthy. Mass is unchanged by damage.
    stiffness_factors: np.ndarray | None = None
    # node indices where transverse deflection w = 0 (a multi-span VIADUCT: the two
    # abutments plus the intermediate piers). None -> simply supported at the two
    # ends only. Rotations stay free everywhere (a continuous girder over piers).
    support_nodes: tuple | None = None

    @property
    def n_nodes(self) -> int:
        return self.n_elements + 1

    @property
    def n_dof(self) -> int:
        return 2 * self.n_nodes

    @property
    def le(self) -> float:
        return self.L / self.n_elements

    def node_x(self, k: int) -> float:
        return k * self.le

    @property
    def mid_node(self) -> int:
        return self.n_elements // 2


def _element_stiffness(EI: float, le: float) -> np.ndarray:
    """4x4 beam-bending element stiffness, DOF order [w_i, th_i, w_j, th_j]."""
    return (EI / le**3) * np.array([
        [ 12,     6 * le,   -12,     6 * le],
        [ 6 * le, 4 * le**2, -6 * le, 2 * le**2],
        [-12,    -6 * le,    12,    -6 * le],
        [ 6 * le, 2 * le**2, -6 * le, 4 * le**2],
    ])


def _element_mass(m_bar: float, le: float) -> np.ndarray:
    """4x4 CONSISTENT mass matrix (cubic Hermite shape functions)."""
    return (m_bar * le / 420.0) * np.array([
        [156,     22 * le,    54,    -13 * le],
        [22 * le,  4 * le**2, 13 * le, -3 * le**2],
        [54,      13 * le,   156,    -22 * le],
        [-13 * le, -3 * le**2, -22 * le, 4 * le**2],
    ])


def assemble(beam: Beam) -> tuple[np.ndarray, np.ndarray]:
    """Assemble global mass M and stiffness K (size n_dof x n_dof).

    Same scatter-add idea as the truss: each element's 4x4 matrices go into the
    global arrays at its two nodes' DOFs.
    """
    n = beam.n_dof
    M = np.zeros((n, n))
    K = np.zeros((n, n))
    EI = beam.E * beam.I
    ke = _element_stiffness(EI, beam.le)
    me = _element_mass(beam.mass_per_length, beam.le)
    for e in range(beam.n_elements):
        i, j = e, e + 1  # node indices of this element
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        # scale this element's stiffness if it is damaged (mass stays intact)
        factor = 1.0 if beam.stiffness_factors is None else beam.stiffness_factors[e]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += factor * ke[a, b]
                M[dofs[a], dofs[b]] += me[a, b]
    return M, K


def make_damaged_beam(base: Beam, x_start: float, x_end: float,
                      severity: float) -> Beam:
    """Copy `base` with a localized stiffness loss over [x_start, x_end].

    severity in [0, 1) : fractional EI loss in the damaged zone (0.3 = -30% EI).
    Elements whose centre falls in the zone get factor (1 - severity).
    """
    factors = np.ones(base.n_elements)
    for e in range(base.n_elements):
        xc = (e + 0.5) * base.le
        if x_start <= xc <= x_end:
            factors[e] = 1.0 - severity
    return Beam(L=base.L, E=base.E, I=base.I,
                mass_per_length=base.mass_per_length,
                n_elements=base.n_elements, stiffness_factors=factors)


def simply_supported_constrained_dofs(beam: Beam) -> list[int]:
    """Constrained transverse-deflection DOFs (w = 0 at the supports).

    Default: pin both ends (single simply-supported span). If `support_nodes` is
    set, pin w = 0 at every listed node instead — a multi-span continuous viaduct
    over its abutments and piers. Rotations always stay free.
    """
    if beam.support_nodes is not None:
        return [2 * k for k in beam.support_nodes]
    return [0, 2 * (beam.n_nodes - 1)]


def make_viaduct(span_len: float, n_spans: int, E: float, I: float,
                 mass_per_length: float, elems_per_span: int = 6) -> Beam:
    """A continuous multi-span viaduct: `n_spans` equal spans over piers.

    Total length = span_len * n_spans. Piers (w = 0, rotation free) sit at every
    span boundary, so this is a continuous girder — the right model for a long
    bridge, unlike a single 500 m simply-supported beam (which is not physical).
    """
    n_el = n_spans * elems_per_span
    supports = tuple(s * elems_per_span for s in range(n_spans + 1))
    return Beam(L=span_len * n_spans, E=E, I=I, mass_per_length=mass_per_length,
                n_elements=n_el, support_nodes=supports)


def free_dofs(beam: Beam) -> list[int]:
    constrained = set(simply_supported_constrained_dofs(beam))
    return [d for d in range(beam.n_dof) if d not in constrained]


def static_solve(beam: Beam, loads: dict[int, float]) -> np.ndarray:
    """Solve K u = F for a static set of nodal transverse point loads.

    loads : node index -> transverse force [N] (downward negative, your choice;
            here positive = +w direction). Returns the full DOF vector u.
    """
    M, K = assemble(beam)
    F = np.zeros(beam.n_dof)
    for node, value in loads.items():
        F[2 * node] += value  # transverse DOF of that node

    free = free_dofs(beam)
    u = np.zeros(beam.n_dof)
    K_ff = K[np.ix_(free, free)]
    u[free] = np.linalg.solve(K_ff, F[free])
    return u


def rayleigh_alpha_beta(beam: Beam, zeta: float,
                        f1: float | None = None,
                        f2: float | None = None) -> tuple[float, float]:
    """Rayleigh damping coefficients (alpha, beta) for C = alpha*M + beta*K.

    Real bridges dissipate energy; we model it with classical Rayleigh damping,
    which gives the SAME damping ratio `zeta` at two chosen frequencies (default:
    the bridge's first two natural frequencies). The modal damping ratio is
        zeta(omega) = alpha/(2 omega) + beta*omega/2,
    and forcing it to equal `zeta` at omega1, omega2 gives the closed form below.
    Typical highway bridges sit around zeta = 1-3%.
    """
    if f1 is None or f2 is None:
        f = natural_frequencies(beam, 2)
        f1, f2 = float(f[0]), float(f[1])
    w1, w2 = 2.0 * np.pi * f1, 2.0 * np.pi * f2
    alpha = 2.0 * zeta * w1 * w2 / (w1 + w2)
    beta = 2.0 * zeta / (w1 + w2)
    return alpha, beta


def natural_frequencies(beam: Beam, n_modes: int = 5) -> np.ndarray:
    """Lowest natural frequencies [Hz] from the generalized eigenproblem
    K phi = omega^2 M phi on the free DOFs.

    Solved numpy-only: M is symmetric positive-definite, so factor M = L L^T
    (Cholesky) and reduce to the standard symmetric problem
        A y = omega^2 y,   A = L^-1 K L^-T,
    whose eigenvalues are omega^2. This avoids any extra dependency.
    """
    M, K = assemble(beam)
    free = free_dofs(beam)
    M_ff = M[np.ix_(free, free)]
    K_ff = K[np.ix_(free, free)]

    L = np.linalg.cholesky(M_ff)
    B = np.linalg.solve(L, K_ff)          # L^-1 K
    A = np.linalg.solve(L, B.T).T          # L^-1 K L^-T  (K symmetric)
    eigvals = np.linalg.eigvalsh(A)        # = omega^2, ascending
    eigvals = np.clip(eigvals, 0.0, None)  # guard tiny negative round-off
    omega = np.sqrt(eigvals)
    freqs_hz = omega / (2.0 * np.pi)
    return freqs_hz[:n_modes]
