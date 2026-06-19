"""V2 — moving CONSTANT force across the simply-supported beam.

Builds on the V1 beam core (assemble M, K). A point force of constant
magnitude P travels at constant speed v from the left support to the right
support. At every instant the force sits somewhere *inside* an element, so we
distribute it to that element's 4 DOFs with the SAME cubic Hermite shape
functions used to build the consistent mass matrix — this is the consistent
nodal load vector. Rebuilding it every step gives the time-varying F(t).

We then march  M u'' + K u = F(t)  in time with classic RK4 on the first-order
state-space form, undamped (C = 0) to match the analytical reference. Only the
free DOFs are integrated; the constrained support DOFs stay zero.

The ground truth is the Frýba closed-form moving-force solution,
implemented here as its modal series `fryba_deflection`. If the FEM disagrees
with Frýba, the FEM (or the integration) is wrong — not the reference.

numpy only; kept explicit and commented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beam_fem import (
    Beam,
    assemble,
    free_dofs,
    static_solve,
)


# ---------------------------------------------------------------------------
# Moving-load assembly (FEM side)
# ---------------------------------------------------------------------------
def hermite_shape_functions(le: float, s: float) -> np.ndarray:
    """Cubic Hermite shape functions of a 2-node beam element.

    le : element length.
    s  : position of the load measured from the element's LEFT node, 0 <= s <= le.

    Returns N = [N1, N2, N3, N4] matching element DOF order
    [w_i, theta_i, w_j, theta_j]. A point force P placed at s produces the
    consistent nodal load vector P * N. These are exactly the functions that
    interpolate w(s) inside the element, so this is *consistent* with the FEM
    displacement field (and with the consistent mass matrix).
    """
    xi = s / le  # non-dimensional position in [0, 1]
    return np.array([
        1.0 - 3.0 * xi**2 + 2.0 * xi**3,        # N1 -> w_i
        le * (xi - 2.0 * xi**2 + xi**3),        # N2 -> theta_i
        3.0 * xi**2 - 2.0 * xi**3,              # N3 -> w_j
        le * (-xi**2 + xi**3),                  # N4 -> theta_j
    ])


def moving_force_vector(beam: Beam, P: float, xp: float) -> np.ndarray:
    """Global load vector for a point force P located at axial position xp [m].

    Finds the element containing xp, evaluates the Hermite shape functions
    there, and scatters P * N into that element's four global DOFs. Outside the
    span (xp < 0 or xp > L) the load is off the bridge -> zero vector.
    """
    F = np.zeros(beam.n_dof)
    if xp < 0.0 or xp > beam.L:
        return F

    # Which element? Clamp the right edge so xp == L lands in the last element.
    e = min(int(xp / beam.le), beam.n_elements - 1)
    s = xp - e * beam.le                       # local coordinate from left node
    N = hermite_shape_functions(beam.le, s)

    i, j = e, e + 1
    dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
    for a in range(4):
        F[dofs[a]] += P * N[a]
    return F


# ---------------------------------------------------------------------------
# Bending moment from the FEM solution (needed by the B-WIM capstone, V4.5)
# ---------------------------------------------------------------------------
def hermite_curvature(le: float, s: float) -> np.ndarray:
    """Second derivative d^2N/ds^2 of the Hermite shape functions ("B" vector).

    w(s) = N(s) . d_e  =>  curvature w''(s) = B(s) . d_e. Used to get the
    bending moment M = -EI w'' inside an element. DOF order [w_i, th_i, w_j, th_j].
    """
    xi = s / le
    return np.array([
        (-6.0 + 12.0 * xi) / le**2,   # d2 N1
        (-4.0 + 6.0 * xi) / le,       # d2 N2
        (6.0 - 12.0 * xi) / le**2,    # d2 N3
        (-2.0 + 6.0 * xi) / le,       # d2 N4
    ])


def bending_moment_at(beam: Beam, u: np.ndarray, x_m: float) -> float:
    """Internal bending moment M = -EI w'' at section x_m, from the FEM field u.

    Sign convention: M > 0 is sagging (the static central-load case gives the
    familiar +P L / 4). At an interior node the curvature can jump between the
    two adjacent elements (C1 Hermite elements), so we average them — this is
    the value a strain gauge straddling the section would report.
    """
    EI = beam.E * beam.I
    e = min(int(x_m / beam.le), beam.n_elements - 1)
    s = x_m - e * beam.le

    def moment_in_element(elem: int, s_local: float) -> float:
        i, j = elem, elem + 1
        d = u[[2 * i, 2 * i + 1, 2 * j, 2 * j + 1]]
        B = hermite_curvature(beam.le, s_local)
        return -EI * (B @ d)

    m_right = moment_in_element(e, s)
    if abs(s) < 1e-9 and e > 0:
        m_left = moment_in_element(e - 1, beam.le)
        return 0.5 * (m_left + m_right)
    return m_right


# ---------------------------------------------------------------------------
# Time integration (RK4 on the state-space form)
# ---------------------------------------------------------------------------
@dataclass
class MovingForceResult:
    """Output of a moving-force time march."""
    t: np.ndarray            # time stamps [s]
    w_mid: np.ndarray        # mid-span transverse deflection history [m]
    dt: float                # time step actually used [s]
    f_max_hz: float          # highest FEM natural frequency (set the dt limit)
    static_midspan: float    # PL^3/48EI reference deflection [m]
    daf: float               # dynamic amplification factor (see below)
    crossing_time: float     # L / speed [s]


def integrate_moving_force(
    beam: Beam,
    P: float,
    speed: float,
    n_steps_per_crossing: int = 2000,
    extra_crossings: float = 0.0,
    damping_ratio: float | None = None,
) -> MovingForceResult:
    """March M u'' + K u = F(t) with RK4 for a force P crossing at `speed`.

    The state is y = [u_f, v_f] on the FREE DOFs only. The derivative is
        u_f'  = v_f
        v_f'  = M_ff^{-1} ( F_f(t) - K_ff u_f )
    Undamped: C = 0, matching Frýba.

    Time step: RK4 on an undamped structure is stable only if omega_max * dt is
    below ~2.83 (its imaginary-axis limit). The consistent-mass beam has very
    stiff high modes, so we read omega_max from the eigenproblem and cap dt at
    1.8/omega_max, then also demand enough steps to resolve the crossing.

    extra_crossings : integrate this many extra crossing-times AFTER the load
        leaves (force is then zero -> free vibration). 0.0 = stop at exit.
    """
    M, K = assemble(beam)
    free = free_dofs(beam)
    M_ff = M[np.ix_(free, free)]
    K_ff = K[np.ix_(free, free)]
    M_inv = np.linalg.inv(M_ff)
    A = M_inv @ K_ff                       # u_f'' = -A u_f + M_inv F_f

    # Highest FEM circular frequency -> RK4 stability ceiling on dt.
    eig = np.linalg.eigvals(A)
    omega_max = float(np.sqrt(np.max(eig.real)))
    f_max_hz = omega_max / (2.0 * np.pi)

    crossing_time = beam.L / speed
    total_time = crossing_time * (1.0 + extra_crossings)

    # Rayleigh damping: C = a*M + b*K, so M^-1 C v = a v + b A v (reuse A).
    if damping_ratio:
        from beam_fem import rayleigh_alpha_beta
        a_ray, b_ray = rayleigh_alpha_beta(beam, damping_ratio)
    else:
        a_ray = b_ray = 0.0

    dt_stability = 1.8 / omega_max
    dt_resolution = crossing_time / n_steps_per_crossing
    dt = min(dt_stability, dt_resolution)
    # stiffness-proportional damping over-damps high modes -> explicit RK4 needs
    # a smaller step (real-axis stability ~ 2.78/(b*omega_max^2)).
    if b_ray > 0:
        dt = min(dt, 2.2 / (b_ray * omega_max**2))
    n_steps = int(np.ceil(total_time / dt))

    # mid-span free-DOF index (w of the mid node, in the reduced ordering)
    mid_global = 2 * beam.mid_node
    mid_free = free.index(mid_global)

    nf = len(free)

    def F_free(t: float) -> np.ndarray:
        """Restricted load vector at time t (force at xp = speed * t)."""
        xp = speed * t
        return moving_force_vector(beam, P, xp)[free]

    def deriv(t: float, y: np.ndarray) -> np.ndarray:
        u = y[:nf]
        v = y[nf:]
        acc = M_inv @ F_free(t) - A @ u - (a_ray * v + b_ray * (A @ v))
        return np.concatenate((v, acc))

    # March from rest.
    y = np.zeros(2 * nf)
    t_hist = np.empty(n_steps + 1)
    w_mid = np.empty(n_steps + 1)
    t_hist[0] = 0.0
    w_mid[0] = 0.0

    t = 0.0
    for step in range(1, n_steps + 1):
        k1 = deriv(t, y)
        k2 = deriv(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = deriv(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = deriv(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        t_hist[step] = t
        w_mid[step] = y[mid_free]

    # Static reference: deflection with the load standing at mid-span.
    static_midspan = P * beam.L**3 / (48.0 * beam.E * beam.I)
    daf = float(np.max(np.abs(w_mid)) / abs(static_midspan))

    return MovingForceResult(
        t=t_hist,
        w_mid=w_mid,
        dt=dt,
        f_max_hz=f_max_hz,
        static_midspan=static_midspan,
        daf=daf,
        crossing_time=crossing_time,
    )


# ---------------------------------------------------------------------------
# Multi-axle moving force (for the multi-axle B-WIM demo, V4.5+)
# ---------------------------------------------------------------------------
@dataclass
class MovingAxlesResult:
    t: np.ndarray            # time [s]
    w_mid: np.ndarray        # mid-span deflection [m]
    moment: np.ndarray       # bending moment at the gauge section [N m]
    dt: float
    crossing_time: float     # time for the LAST axle to clear the span [s]


def integrate_moving_axles(
    beam: Beam,
    axle_loads,
    axle_offsets,
    speed: float,
    moment_section: float,
    n_steps_per_crossing: int = 2000,
) -> MovingAxlesResult:
    """RK4 march for several CONSTANT axle forces crossing together.

    A linear beam, so the response is just the superposition of the single
    forces — but we assemble them into one F(t) and march once. The lead axle is
    at x = speed*t; axle k trails it by axle_offsets[k]. We run until the LAST
    axle has cleared the span. Records the bending moment at `moment_section`
    (the gauge) — the input to the multi-axle Moses recovery.
    """
    axle_loads = np.asarray(axle_loads, dtype=float)
    axle_offsets = np.asarray(axle_offsets, dtype=float)

    M, K = assemble(beam)
    free = free_dofs(beam)
    M_ff = M[np.ix_(free, free)]
    K_ff = K[np.ix_(free, free)]
    M_inv = np.linalg.inv(M_ff)
    A = M_inv @ K_ff
    nf = len(free)

    omega_max = float(np.sqrt(np.max(np.linalg.eigvals(A).real)))
    # lead axle must travel L + (largest trailing offset) for all axles to exit
    crossing_time = (beam.L + float(axle_offsets.max())) / speed
    dt = min(1.8 / omega_max, crossing_time / n_steps_per_crossing)
    n_steps = int(np.ceil(crossing_time / dt))

    mid_free = free.index(2 * beam.mid_node)

    def F_free(t: float) -> np.ndarray:
        xlead = speed * t
        F = np.zeros(beam.n_dof)
        for P_k, off_k in zip(axle_loads, axle_offsets):
            F += moving_force_vector(beam, P_k, xlead - off_k)
        return F[free]

    def deriv(t, y):
        u = y[:nf]
        v = y[nf:]
        return np.concatenate((v, M_inv @ F_free(t) - A @ u))

    y = np.zeros(2 * nf)
    t = 0.0
    t_hist = np.empty(n_steps + 1)
    w_mid = np.empty(n_steps + 1)
    mom = np.empty(n_steps + 1)
    u_full = np.zeros(beam.n_dof)
    t_hist[0], w_mid[0], mom[0] = 0.0, 0.0, 0.0

    for step in range(1, n_steps + 1):
        k1 = deriv(t, y)
        k2 = deriv(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = deriv(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = deriv(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        u_full[free] = y[:nf]
        t_hist[step] = t
        w_mid[step] = y[mid_free]
        mom[step] = bending_moment_at(beam, u_full, moment_section)

    return MovingAxlesResult(t=t_hist, w_mid=w_mid, moment=mom, dt=dt,
                             crossing_time=crossing_time)


# ---------------------------------------------------------------------------
# Frýba closed-form reference (modal series)
# ---------------------------------------------------------------------------
def fryba_deflection(
    beam: Beam,
    P: float,
    speed: float,
    t: np.ndarray,
    x: float | None = None,
    n_modes: int = 50,
) -> np.ndarray:
    """Frýba's moving constant-force deflection, undamped, valid DURING crossing.

    Modal superposition for a simply-supported beam (modes sin(n pi x / L)):

        w(x,t) = w0 * sum_n  1/(n^2 (n^2 - alpha^2))
                            * [ sin(Omega_n t) - (alpha/n) sin(omega_n t) ]
                            * sin(n pi x / L)

    with
        w0       = 2 P L^3 / (pi^4 E I)          reference amplitude
        omega_1  = (pi/L)^2 sqrt(E I / m_bar)    first natural circ. frequency
        omega_n  = n^2 omega_1
        Omega_n  = n pi speed / L                n-th forcing circ. frequency
        alpha    = Omega_1 / omega_1 = (pi speed / L) / omega_1   speed parameter

    This is exactly the closed form in Frýba, "Vibration of Solids and Structures
    under Moving Loads". It holds while the load is on the span (0 <= speed*t <= L);
    we evaluate it on the given t (the caller restricts to the crossing window).

    x : position to evaluate at; default = mid-span.
    """
    if x is None:
        x = 0.5 * beam.L

    EI = beam.E * beam.I
    omega1 = (np.pi / beam.L) ** 2 * np.sqrt(EI / beam.mass_per_length)
    alpha = (np.pi * speed / beam.L) / omega1
    w0 = 2.0 * P * beam.L**3 / (np.pi**4 * EI)

    t = np.asarray(t, dtype=float)
    w = np.zeros_like(t)
    for n in range(1, n_modes + 1):
        omega_n = n**2 * omega1
        Omega_n = n * np.pi * speed / beam.L
        denom = n**2 * (n**2 - alpha**2)
        if abs(denom) < 1e-12:
            # alpha == n : true resonance, this term diverges (handle in V3).
            continue
        modal = (np.sin(Omega_n * t) - (alpha / n) * np.sin(omega_n * t)) / denom
        w += modal * np.sin(n * np.pi * x / beam.L)
    return w0 * w


def fryba_speed_parameter(beam: Beam, speed: float) -> float:
    """alpha = (pi*speed/L) / omega_1 ; alpha = 1 is the critical speed."""
    EI = beam.E * beam.I
    omega1 = (np.pi / beam.L) ** 2 * np.sqrt(EI / beam.mass_per_length)
    return (np.pi * speed / beam.L) / omega1
