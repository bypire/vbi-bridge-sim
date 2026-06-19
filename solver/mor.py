"""Model order reduction (MOR) by modal truncation.

The full beam FEM has 2N coupled DOFs and its stiffest mode forces a tiny explicit
time step. But the bridge response is dominated by a handful of low modes, so we
project the dynamics onto the first r mass-normalised eigenmodes:

    u(t) = Phi q(t),   Phi^T M Phi = I,   Phi^T K Phi = diag(omega^2)

The 2N coupled equations become r DECOUPLED scalar oscillators
    q_i'' + omega_i^2 q_i = phi_i^T F(t),
which are tiny to integrate AND allow a much larger stable step (set by the
retained omega_r, not the full omega_max). Standard reduced-order modelling — the
bread and butter of computational dynamics. numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beam_fem import Beam, assemble, free_dofs
from moving_load import moving_force_vector


def modal_basis(beam: Beam, r: int):
    """First r mass-normalised modes on the free DOFs.

    Returns (omega[r], Phi[nf x r], free, mid_free) with Phi^T M Phi = I.
    Solved numpy-only via the Cholesky reduction M = Lc Lc^T (same trick as
    natural_frequencies), so it needs no extra dependency.
    """
    M, K = assemble(beam)
    free = free_dofs(beam)
    M_ff = M[np.ix_(free, free)]
    K_ff = K[np.ix_(free, free)]
    Lc = np.linalg.cholesky(M_ff)
    B = np.linalg.solve(Lc, K_ff)
    A = np.linalg.solve(Lc, B.T).T          # L^-1 K L^-T (symmetric)
    eigvals, Y = np.linalg.eigh(A)          # ascending; Y orthonormal
    eigvals = np.clip(eigvals[:r], 0.0, None)
    Phi = np.linalg.solve(Lc.T, Y[:, :r])   # modes; Phi^T M Phi = I
    omega = np.sqrt(eigvals)
    mid_free = free.index(2 * beam.mid_node)
    return omega, Phi, free, mid_free


@dataclass
class ModalResult:
    t: np.ndarray
    w_mid: np.ndarray
    dt: float
    n_modes: int
    daf: float
    static_midspan: float


def integrate_modal_moving_force(beam: Beam, P: float, speed: float, r: int,
                                 dt: float | None = None) -> ModalResult:
    """RK4 on the r decoupled modal oscillators for a moving constant force."""
    omega, Phi, free, mid_free = modal_basis(beam, r)
    omega_r = float(omega[-1])
    if dt is None:
        dt = 0.3 / omega_r                  # stable for the retained modes
    crossing = beam.L / speed
    n_steps = int(np.ceil(crossing / dt))
    w2 = omega**2
    phi_mid = Phi[mid_free, :]

    def g(t):                               # modal forcing phi^T F(t)
        F = moving_force_vector(beam, P, speed * t)[free]
        return Phi.T @ F

    def deriv(t, y):
        q, qd = y[:r], y[r:]
        return np.concatenate((qd, g(t) - w2 * q))

    y = np.zeros(2 * r)
    t_hist = np.empty(n_steps + 1)
    w_mid = np.empty(n_steps + 1)
    t_hist[0] = 0.0; w_mid[0] = 0.0
    t = 0.0
    for s in range(1, n_steps + 1):
        k1 = deriv(t, y); k2 = deriv(t + .5*dt, y + .5*dt*k1)
        k3 = deriv(t + .5*dt, y + .5*dt*k2); k4 = deriv(t + dt, y + dt*k3)
        y = y + (dt/6.)*(k1 + 2*k2 + 2*k3 + k4)
        t += dt
        t_hist[s] = t
        w_mid[s] = phi_mid @ y[:r]
    static_mid = P * beam.L**3 / (48.*beam.E*beam.I)
    daf = float(np.max(np.abs(w_mid)) / abs(static_mid))
    return ModalResult(t=t_hist, w_mid=w_mid, dt=dt, n_modes=r, daf=daf,
                       static_midspan=static_mid)
