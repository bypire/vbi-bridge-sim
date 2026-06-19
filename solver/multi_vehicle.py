"""Multiple vehicles on the span at once — real coupled forward physics.

Until now a "stream" of trucks meant one truck at a time. Real traffic puts
SEVERAL vehicles on the bridge simultaneously, and their bending effects
superpose. This module marches that: N independent moving forces, each with its
own entry time and speed, all loading the SAME beam, integrated together as one
M u'' + K u = F(t).

Because an Euler-Bernoulli beam is LINEAR, the convoy response is the exact
superposition of the single-vehicle responses — which is both the physics and a
free, exact correctness check (verify_multi.py). The reason this matters: B-WIM
assumes ONE vehicle on the span; simultaneous presence corrupts the inverse, and
that corruption is the scientific question (quantified in verify_multi.py).

numpy only; explicit RK4 on the free DOFs (undamped, for speed/throughput).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beam_fem import Beam, assemble, free_dofs
from moving_load import bending_moment_at, moving_force_vector


@dataclass
class ConvoyResult:
    t: np.ndarray              # time stamps [s]
    moment: np.ndarray         # bending moment at the gauge section, M(t) [N m]
    w_mid: np.ndarray          # mid-span deflection history [m]
    frame_t: np.ndarray        # time of each animation frame [s]
    frame_w: np.ndarray        # (n_frames, n_nodes) deck deflection per frame [m]
    frame_x: np.ndarray        # (n_frames, n_vehicles) each vehicle's position [m]
                               #   (NaN when the vehicle is not on the span)
    node_x: np.ndarray         # node x-coordinates [m]
    dt: float
    n_on_max: int              # max number of vehicles simultaneously on the span


def integrate_convoy(beam: Beam, vehicles, moment_section: float,
                     n_steps_per_crossing: int = 1500, n_frames: int = 140):
    """March a convoy of independent moving forces across the beam.

    vehicles : list of dicts {"P": force [N], "speed": [m/s], "enter": entry time [s]}.
               Vehicle k sits at x_k(t) = speed_k * (t - enter_k); it loads the beam
               only while 0 <= x_k <= L.
    moment_section : gauge location for the recorded bending moment [m].
    """
    M, K = assemble(beam)
    free = free_dofs(beam)
    M_ff = M[np.ix_(free, free)]
    K_ff = K[np.ix_(free, free)]
    M_inv = np.linalg.inv(M_ff)
    A = M_inv @ K_ff
    nf = len(free)
    omega_max = float(np.sqrt(np.max(np.linalg.eigvals(A).real)))

    P = np.array([v["P"] for v in vehicles], float)
    spd = np.array([v["speed"] for v in vehicles], float)
    ent = np.array([v["enter"] for v in vehicles], float)
    # each vehicle exits when speed*(t-enter) > L
    exit_t = ent + beam.L / spd
    total_time = float(exit_t.max()) * 1.02
    dt = min(1.8 / omega_max, total_time / max(1, n_steps_per_crossing * 1))
    n_steps = int(np.ceil(total_time / dt))
    frame_stride = max(1, n_steps // n_frames)

    mid_free = free.index(2 * beam.mid_node)
    node_x = np.array([beam.node_x(i) for i in range(beam.n_nodes)])

    def positions(t):
        return spd * (t - ent)                      # x of each vehicle at time t

    def F_free(t):
        F = np.zeros(beam.n_dof)
        x = positions(t)
        for k in range(len(P)):
            if 0.0 <= x[k] <= beam.L:
                F += moving_force_vector(beam, P[k], x[k])
        return F[free]

    def deriv(t, y):
        u = y[:nf]; v = y[nf:]
        return np.concatenate((v, M_inv @ F_free(t) - A @ u))

    y = np.zeros(2 * nf)
    t = 0.0
    t_hist = np.empty(n_steps + 1); mom = np.empty(n_steps + 1)
    w_mid = np.empty(n_steps + 1)
    u_full = np.zeros(beam.n_dof)
    t_hist[0] = mom[0] = w_mid[0] = 0.0
    fr_t, fr_w, fr_x = [0.0], [np.zeros(beam.n_nodes)], [
        np.where((positions(0.0) >= 0) & (positions(0.0) <= beam.L), positions(0.0), np.nan)]
    n_on_max = 0

    for step in range(1, n_steps + 1):
        k1 = deriv(t, y); k2 = deriv(t + .5*dt, y + .5*dt*k1)
        k3 = deriv(t + .5*dt, y + .5*dt*k2); k4 = deriv(t + dt, y + dt*k3)
        y = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t += dt
        u_full[free] = y[:nf]
        t_hist[step] = t
        w_mid[step] = y[mid_free]
        mom[step] = bending_moment_at(beam, u_full, moment_section)
        x = positions(t)
        on = (x >= 0) & (x <= beam.L)
        n_on_max = max(n_on_max, int(on.sum()))
        if step % frame_stride == 0:
            w_nodes = u_full[0::2].copy()           # transverse DOF at each node
            fr_t.append(t); fr_w.append(w_nodes)
            fr_x.append(np.where(on, x, np.nan))

    return ConvoyResult(
        t=t_hist, moment=mom, w_mid=w_mid,
        frame_t=np.array(fr_t), frame_w=np.array(fr_w), frame_x=np.array(fr_x),
        node_x=node_x, dt=dt, n_on_max=n_on_max)
