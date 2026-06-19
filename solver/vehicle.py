"""V3 — quarter-car vehicle coupled to the moving point on the beam.

Step up from V2's *constant* moving force: the force the wheel applies to the
bridge is no longer fixed. The vehicle is a 2-DOF quarter-car —

    sprung mass  m_s  (vehicle body) on suspension  k_s, c_s
    unsprung mass m_u (axle + wheel) on tyre        k_t   (tyre damping c_t = 0)

riding on the bridge surface. As the bridge deflects under the wheel, the tyre
deformation changes, so the contact force changes, which changes the bridge
deflection — a TWO-WAY coupling. We integrate beam + vehicle as one state vector
with the same RK4 used in V2.

Sign convention: everything DOWNWARD-POSITIVE (bridge w, vehicle z_s, z_u, and
gravity g). With gravity carried explicitly, the static contact force comes out
to the full weight (m_s+m_u)g — see `static_equilibrium` — so initialising the
vehicle there means it drives onto the span with no spurious entry kick.

Contact force on the bridge (downward, +):
    F_c(t) = k_t * ( z_u(t) - w_c(t) ),     w_c = N(x_p)^T u   (beam deflection
                                            interpolated at the contact point)
This F_c is what gets distributed to the beam DOFs via the Hermite functions N.

numpy only; explicit and commented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beam_fem import Beam, assemble, free_dofs
from moving_load import bending_moment_at, moving_force_vector


@dataclass
class QuarterCar:
    """A 2-DOF quarter-car. Consistent SI units.

    m_s, m_u : sprung / unsprung mass [kg]
    k_s, k_t : suspension / tyre stiffness [N/m]
    c_s, c_t : suspension / tyre damping [N s/m]  (c_t defaults to 0)
    g        : gravitational acceleration [m/s^2]
    """

    m_s: float
    m_u: float
    k_s: float
    k_t: float
    c_s: float = 0.0
    c_t: float = 0.0
    g: float = 9.81

    @property
    def weight(self) -> float:
        """Total static axle load transmitted to the bridge [N]."""
        return (self.m_s + self.m_u) * self.g

    def static_equilibrium(self) -> tuple[float, float]:
        """Rest positions (z_s0, z_u0) on a RIGID flat road (w_c = 0).

        Tyre carries the whole weight: k_t z_u0 = (m_s+m_u) g.
        Suspension carries the body:   k_s (z_s0 - z_u0) = m_s g.
        """
        z_u0 = self.weight / self.k_t
        z_s0 = z_u0 + self.m_s * self.g / self.k_s
        return z_s0, z_u0

    def natural_frequencies_rigid(self) -> np.ndarray:
        """The two vehicle natural frequencies [Hz] on a rigid road (w_c = 0).

        Undamped generalized eigenproblem of the 2-DOF system
            M_v = diag(m_s, m_u),  K_v = [[k_s, -k_s], [-k_s, k_s+k_t]].
        Lower root ~ body bounce, upper ~ wheel hop. Solved as a 2x2 — this is
        the analytical reference the coupled integrator is checked against.
        """
        M_v = np.diag([self.m_s, self.m_u])
        K_v = np.array([[self.k_s, -self.k_s],
                        [-self.k_s, self.k_s + self.k_t]])
        # generalized eig via M^{-1/2} symmetric reduction (2x2, exact)
        w2 = np.linalg.eigvals(np.linalg.solve(M_v, K_v))
        w2 = np.sort(np.real(w2))
        return np.sqrt(np.clip(w2, 0.0, None)) / (2.0 * np.pi)


@dataclass
class TwoAxleResult:
    """Two-axle crossing: per-axle accelerations for the residual CP method."""
    t: np.ndarray
    a_s1: np.ndarray   # front sprung / unsprung acceleration
    a_u1: np.ndarray
    a_s2: np.ndarray   # rear sprung / unsprung acceleration
    a_u2: np.ndarray
    x1: np.ndarray     # front contact position [m]
    x2: np.ndarray     # rear contact position [m] (= x1 - spacing)
    dt: float
    spacing: float
    crossing_time: float


def integrate_two_axle(beam: Beam, car: QuarterCar, spacing: float, speed: float,
                       road=None, n_steps_per_crossing: int = 2000) -> TwoAxleResult:
    """RK4 march of a TWO-axle vehicle (two quarter-cars, spacing `d`) on the beam.

    Both axles load the same beam (so they interact through it) and ride the same
    road profile, the rear trailing the front by `spacing`. Recording both axles'
    accelerations lets us form the time-shifted residual that cancels the common
    road roughness and exposes the bridge response (Yang/OBrien residual method).

    State y = [ u_f , z_s1,z_u1,z_s2,z_u2 , u_f' , z_s1',z_u1',z_s2',z_u2' ].
    """
    M, K = assemble(beam)
    free = free_dofs(beam)
    M_ff = M[np.ix_(free, free)]
    K_ff = K[np.ix_(free, free)]
    M_inv = np.linalg.inv(M_ff)
    A = M_inv @ K_ff
    nf = len(free)

    omega_max = float(np.sqrt(np.max(np.linalg.eigvals(A).real)))
    crossing_time = (beam.L + spacing) / speed     # until the REAR axle exits
    dt = min(1.8 / omega_max, crossing_time / n_steps_per_crossing)
    n_steps = int(np.ceil(crossing_time / dt))

    m_s, m_u, k_s, k_t, c_s, g = car.m_s, car.m_u, car.k_s, car.k_t, car.c_s, car.g
    z_s0, z_u0 = car.static_equilibrium()

    def Nf(x):
        return moving_force_vector(beam, 1.0, x)[free]

    def road_h(x):
        return float(road(x)) if road is not None else 0.0

    # indices
    iZ = nf                       # z_s1 at nf, z_u1 nf+1, z_s2 nf+2, z_u2 nf+3
    iV = nf + 4                   # velocities block start
    iVz = 2 * nf + 4              # vehicle velocities

    def deriv(t, y):
        u = y[:nf]
        z = y[iZ:iZ + 4]                         # [zs1,zu1,zs2,zu2]
        v = y[iV:iV + nf]
        zd = y[iVz:iVz + 4]                       # vehicle velocities
        x1 = speed * t
        x2 = speed * t - spacing
        N1, N2 = Nf(x1), Nf(x2)
        wc1, wc2 = N1 @ u, N2 @ u
        Fc1 = k_t * (z[1] - wc1 - road_h(x1))
        Fc2 = k_t * (z[3] - wc2 - road_h(x2))
        beam_acc = M_inv @ (N1 * Fc1 + N2 * Fc2) - A @ u
        susp1 = k_s * (z[0] - z[1]) + c_s * (zd[0] - zd[1])
        susp2 = k_s * (z[2] - z[3]) + c_s * (zd[2] - zd[3])
        acc = np.array([
            g - susp1 / m_s, g + (susp1 - Fc1) / m_u,
            g - susp2 / m_s, g + (susp2 - Fc2) / m_u,
        ])
        dy = np.empty_like(y)
        dy[:nf] = v
        dy[iZ:iZ + 4] = zd
        dy[iV:iV + nf] = beam_acc
        dy[iVz:iVz + 4] = acc
        return dy

    y = np.zeros(2 * nf + 8)
    y[iZ:iZ + 4] = [z_s0, z_u0, z_s0, z_u0]

    t = 0.0
    th = np.empty(n_steps + 1)
    a = {k: np.zeros(n_steps + 1) for k in ("s1", "u1", "s2", "u2")}
    for step in range(1, n_steps + 1):
        k1 = deriv(t, y)
        k2 = deriv(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = deriv(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = deriv(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        d = deriv(t, y)                           # accelerations at the new state
        th[step] = t
        a["s1"][step] = d[iVz]
        a["u1"][step] = d[iVz + 1]
        a["s2"][step] = d[iVz + 2]
        a["u2"][step] = d[iVz + 3]

    x1 = speed * th
    return TwoAxleResult(t=th, a_s1=a["s1"], a_u1=a["u1"], a_s2=a["s2"],
                         a_u2=a["u2"], x1=x1, x2=x1 - spacing, dt=dt,
                         spacing=spacing, crossing_time=crossing_time)


@dataclass
class CoupledResult:
    t: np.ndarray            # time [s]
    w_mid: np.ndarray        # bridge mid-span deflection [m]
    z_s: np.ndarray          # sprung-mass displacement [m]
    z_u: np.ndarray          # unsprung-mass displacement [m]
    contact_force: np.ndarray  # F_c(t) on the bridge [N]
    a_s: np.ndarray            # sprung-mass acceleration [m/s^2]
    a_u: np.ndarray            # unsprung-mass acceleration [m/s^2]
    dt: float
    f_max_hz: float
    static_midspan: float    # W L^3 / 48EI  (W = vehicle weight) [m]
    daf: float
    crossing_time: float
    moment_section: float | None = None  # x of the strain-gauge section [m]
    moment: np.ndarray | None = None     # bending moment there, M(t) [N m]
    # Down-sampled animation frames (only if frame_stride was requested):
    frames_t: np.ndarray | None = None   # time of each frame [s]
    frames_u: np.ndarray | None = None   # FREE-DOF vector per frame (n_frames, nf)
    frames_zs: np.ndarray | None = None  # sprung-mass displacement per frame [m]
    frames_zu: np.ndarray | None = None  # unsprung-mass displacement per frame [m]
    frames_fc: np.ndarray | None = None  # contact force per frame [N]


def integrate_coupled(
    beam: Beam,
    car: QuarterCar,
    speed: float,
    n_steps_per_crossing: int = 2000,
    extra_crossings: float = 0.0,
    moment_section: float | None = None,
    frame_stride: int | None = None,
    n_frames: int | None = None,
    road=None,
    damping_ratio: float | None = None,
) -> CoupledResult:
    """RK4 time march of the coupled beam + quarter-car system.

    Combined state  y = [ u_f , z_s , z_u , u_f' , z_s' , z_u' ]  where u_f are
    the FREE beam DOFs. The derivative couples the two sub-systems through the
    contact force F_c (depends on z_u and the bridge deflection w_c at x_p) and
    through w_c feeding back into the unsprung-mass equation.

    moment_section : if given, also record the bending moment at this section
        (the "strain gauge" the B-WIM capstone reads).
    frame_stride : if given, snapshot the full state every `frame_stride` steps
        for animation export (the time march itself is unaffected).
    road : optional callable h(x) giving the road-surface elevation [m] at
        position x (e.g. an ISO 8608 RoadProfile). The contact point follows the
        bridge deflection PLUS the road profile, so the tyre force becomes
        F_c = k_t (z_u - w_c - h(x_p)). This is the drive-by excitation source.
    """
    M, K = assemble(beam)
    free = free_dofs(beam)
    M_ff = M[np.ix_(free, free)]
    K_ff = K[np.ix_(free, free)]
    M_inv = np.linalg.inv(M_ff)
    A = M_inv @ K_ff
    nf = len(free)

    # dt from the beam's stiffest mode (same RK4 stability argument as V2); the
    # vehicle modes are far slower, so the beam sets the limit.
    omega_max = float(np.sqrt(np.max(np.linalg.eigvals(A).real)))
    f_max_hz = omega_max / (2.0 * np.pi)

    # Rayleigh damping on the bridge: M^-1 C v = a v + b A v.
    if damping_ratio:
        from beam_fem import rayleigh_alpha_beta
        a_ray, b_ray = rayleigh_alpha_beta(beam, damping_ratio)
    else:
        a_ray = b_ray = 0.0

    crossing_time = beam.L / speed
    total_time = crossing_time * (1.0 + extra_crossings)
    dt = min(1.8 / omega_max, crossing_time / n_steps_per_crossing)
    if b_ray > 0:                       # stiffness-prop. damping stiffens RK4
        dt = min(dt, 2.2 / (b_ray * omega_max**2))
    n_steps = int(np.ceil(total_time / dt))

    # derive the frame stride from n_frames (avoids a wasteful probe run)
    if frame_stride is None and n_frames is not None:
        frame_stride = max(1, n_steps // n_frames)

    mid_free = free.index(2 * beam.mid_node)

    m_s, m_u = car.m_s, car.m_u
    k_s, k_t, c_s = car.k_s, car.k_t, car.c_s
    g = car.g

    def contact_shape(t: float) -> np.ndarray:
        """Free-DOF Hermite vector N_f at the current contact position."""
        return moving_force_vector(beam, 1.0, speed * t)[free]

    def road_h(t: float) -> float:
        """Road-surface elevation under the wheel at time t [m]."""
        return float(road(speed * t)) if road is not None else 0.0

    def deriv(t: float, y: np.ndarray) -> np.ndarray:
        u = y[:nf]
        z_s = y[nf]
        z_u = y[nf + 1]
        v = y[nf + 2:2 * nf + 2]
        z_s_dot = y[2 * nf + 2]
        z_u_dot = y[2 * nf + 3]

        Nf = contact_shape(t)
        w_c = Nf @ u                      # bridge deflection under the wheel
        # contact point follows bridge + road profile (c_t = 0)
        F_c = k_t * (z_u - w_c - road_h(t))

        beam_acc = M_inv @ (Nf * F_c) - A @ u - (a_ray * v + b_ray * (A @ v))

        susp = k_s * (z_s - z_u) + c_s * (z_s_dot - z_u_dot)
        z_s_dd = g - susp / m_s
        z_u_dd = g + (susp - F_c) / m_u

        dy = np.empty_like(y)
        dy[:nf] = v
        dy[nf] = z_s_dot
        dy[nf + 1] = z_u_dot
        dy[nf + 2:2 * nf + 2] = beam_acc
        dy[2 * nf + 2] = z_s_dd
        dy[2 * nf + 3] = z_u_dd
        return dy

    # initial state: bridge at rest, vehicle in static equilibrium on rigid road
    z_s0, z_u0 = car.static_equilibrium()
    y = np.zeros(2 * nf + 4)
    y[nf] = z_s0
    y[nf + 1] = z_u0

    t = 0.0
    t_hist = np.empty(n_steps + 1)
    w_mid = np.empty(n_steps + 1)
    zs_hist = np.empty(n_steps + 1)
    zu_hist = np.empty(n_steps + 1)
    fc_hist = np.empty(n_steps + 1)
    as_hist = np.empty(n_steps + 1)       # sprung-mass acceleration
    au_hist = np.empty(n_steps + 1)       # unsprung-mass acceleration
    t_hist[0], w_mid[0] = 0.0, 0.0
    zs_hist[0], zu_hist[0] = z_s0, z_u0
    fc_hist[0] = k_t * (z_u0 - 0.0)       # = weight at entry
    as_hist[0], au_hist[0] = 0.0, 0.0     # at rest in static equilibrium

    record_moment = moment_section is not None
    mom_hist = np.empty(n_steps + 1) if record_moment else None
    u_full = np.zeros(beam.n_dof)         # scratch for moment reconstruction
    if record_moment:
        mom_hist[0] = 0.0                 # bridge flat at entry

    record_frames = frame_stride is not None
    fr_t, fr_u, fr_zs, fr_zu, fr_fc = [], [], [], [], []
    if record_frames:
        fr_t.append(0.0); fr_u.append(np.zeros(nf))
        fr_zs.append(z_s0); fr_zu.append(z_u0); fr_fc.append(fc_hist[0])

    for step in range(1, n_steps + 1):
        k1 = deriv(t, y)
        k2 = deriv(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = deriv(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = deriv(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt

        Nf = contact_shape(t)
        w_c = Nf @ y[:nf]
        F_c = k_t * (y[nf + 1] - w_c - road_h(t))
        susp = k_s * (y[nf] - y[nf + 1]) + c_s * (y[2 * nf + 2] - y[2 * nf + 3])
        t_hist[step] = t
        w_mid[step] = y[mid_free]
        zs_hist[step] = y[nf]
        zu_hist[step] = y[nf + 1]
        fc_hist[step] = F_c
        as_hist[step] = g - susp / m_s            # sprung-mass acceleration
        au_hist[step] = g + (susp - F_c) / m_u    # unsprung-mass acceleration
        if record_moment:
            u_full[free] = y[:nf]
            mom_hist[step] = bending_moment_at(beam, u_full, moment_section)
        if record_frames and step % frame_stride == 0:
            fr_t.append(t); fr_u.append(y[:nf].copy())
            fr_zs.append(y[nf]); fr_zu.append(y[nf + 1])
            fr_fc.append(fc_hist[step])

    static_midspan = car.weight * beam.L**3 / (48.0 * beam.E * beam.I)
    daf = float(np.max(np.abs(w_mid)) / abs(static_midspan))

    return CoupledResult(
        t=t_hist, w_mid=w_mid, z_s=zs_hist, z_u=zu_hist,
        contact_force=fc_hist, a_s=as_hist, a_u=au_hist, dt=dt, f_max_hz=f_max_hz,
        static_midspan=static_midspan, daf=daf, crossing_time=crossing_time,
        moment_section=moment_section, moment=mom_hist,
        frames_t=np.array(fr_t) if record_frames else None,
        frames_u=np.array(fr_u) if record_frames else None,
        frames_zs=np.array(fr_zs) if record_frames else None,
        frames_zu=np.array(fr_zu) if record_frames else None,
        frames_fc=np.array(fr_fc) if record_frames else None,
    )
