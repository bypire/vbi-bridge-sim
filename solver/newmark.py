"""Newmark-beta time integration — the standard structural integrator.

Our demo marched the equation of motion with explicit RK4. RK4 is conditionally
stable: with stiffness-proportional damping the highest FEM modes force a tiny
step (we measured dt ~ 4e-6 s). Newmark-beta (average-acceleration, gamma=1/2,
beta=1/4) is IMPLICIT and unconditionally stable, so it takes large steps on the
same stiff, damped system — the integrator real structural-dynamics codes use.

System:  M u'' + C u' + K u = F(t).
Average-acceleration update, solved on the free DOFs each step via a constant
effective stiffness (factorised once since dt is fixed). numpy only.

Reference: Newmark (1959); any structural dynamics text (Bathe, Chopra).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beam_fem import Beam, assemble, free_dofs, rayleigh_alpha_beta
from moving_load import moving_force_vector


@dataclass
class NewmarkResult:
    t: np.ndarray
    w_mid: np.ndarray
    dt: float
    static_midspan: float
    daf: float
    crossing_time: float


def integrate_newmark_moving_force(
    beam: Beam,
    P: float,
    speed: float,
    dt: float,
    damping_ratio: float | None = None,
    gamma: float = 0.5,
    beta: float = 0.25,
) -> NewmarkResult:
    """Newmark-beta march of a constant force P crossing at `speed`.

    dt is chosen freely (Newmark is unconditionally stable for gamma>=1/2,
    beta>=1/4) — unlike RK4 there is no stiffness/damping step limit.
    """
    M, K = assemble(beam)
    free = free_dofs(beam)
    M_ff = M[np.ix_(free, free)]
    K_ff = K[np.ix_(free, free)]
    nf = len(free)

    if damping_ratio:
        a_ray, b_ray = rayleigh_alpha_beta(beam, damping_ratio)
        C_ff = a_ray * M_ff + b_ray * K_ff
    else:
        C_ff = np.zeros((nf, nf))

    crossing_time = beam.L / speed
    n_steps = int(np.ceil(crossing_time / dt))

    mid_free = free.index(2 * beam.mid_node)

    # Newmark integration constants
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    a6 = dt * (1.0 - gamma)
    a7 = dt * gamma

    # constant effective stiffness, inverted once (nf is small)
    K_eff = K_ff + a0 * M_ff + a1 * C_ff
    K_eff_inv = np.linalg.inv(K_eff)

    def F_free(t):
        return moving_force_vector(beam, P, speed * t)[free]

    u = np.zeros(nf)
    v = np.zeros(nf)
    acc = np.zeros(nf)             # rest -> zero initial acceleration

    t_hist = np.empty(n_steps + 1)
    w_mid = np.empty(n_steps + 1)
    t_hist[0], w_mid[0] = 0.0, 0.0

    t = 0.0
    for step in range(1, n_steps + 1):
        t += dt
        rhs = (F_free(t)
               + M_ff @ (a0 * u + a2 * v + a3 * acc)
               + C_ff @ (a1 * u + a4 * v + a5 * acc))
        u_new = K_eff_inv @ rhs
        acc_new = a0 * (u_new - u) - a2 * v - a3 * acc
        v_new = v + a6 * acc + a7 * acc_new
        u, v, acc = u_new, v_new, acc_new
        t_hist[step] = t
        w_mid[step] = u[mid_free]

    static_midspan = P * beam.L**3 / (48.0 * beam.E * beam.I)
    daf = float(np.max(np.abs(w_mid)) / abs(static_midspan))
    return NewmarkResult(t=t_hist, w_mid=w_mid, dt=dt,
                         static_midspan=static_midspan, daf=daf,
                         crossing_time=crossing_time)
