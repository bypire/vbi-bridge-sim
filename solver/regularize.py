"""P2 -- Tikhonov regularization for the ill-posed multi-axle B-WIM inverse.

Moses' algorithm recovers the GROSS weight superbly but splits CLOSELY-SPACED
axles badly. The reason is purely linear-algebraic: the influence matrix C has one
column per axle (the section influence line shifted by the axle offset), and for a
tandem (axles ~1.3 m apart) two columns are nearly parallel. C^T C is then
ill-conditioned, so the least-squares split amplifies measurement noise -- the
classic signature of an ill-posed inverse problem.

This module adds the standard remedy: **Tikhonov (ridge) regularization**, with an
**L-curve** to choose the regularization weight lambda. It is the bias-variance
trade-off made explicit -- exactly the language of computational inverse problems.

    min_P  || C P - m ||^2  +  lambda^2 || P - P_prior ||^2
    =>  (C^T C + lambda^2 I) P = C^T m + lambda^2 P_prior

numpy only.
"""

from __future__ import annotations

import numpy as np


def condition_number(C: np.ndarray) -> float:
    """2-norm condition number of C^T C (how ill-posed the axle split is).

    cond(C^T C) = (sigma_max / sigma_min)^2 where sigma are the singular values of
    C. A huge value means the split direction is barely excited by the data, so
    noise in that direction is amplified by ~sqrt(cond) in the recovered loads.
    """
    s = np.linalg.svd(C, compute_uv=False)
    s = s[s > 0]
    return float((s[0] / s[-1]) ** 2)


def tikhonov_recover(C: np.ndarray, m: np.ndarray, lam: float,
                     P_prior: np.ndarray | None = None) -> np.ndarray:
    """Ridge-regularized axle loads: (C^T C + lam^2 I) P = C^T m + lam^2 P_prior.

    lam = 0 reproduces ordinary least squares (Moses). Increasing lam pulls the
    solution toward P_prior (default 0), trading a little bias for a large drop in
    noise amplification on the ill-conditioned split direction.
    """
    n = C.shape[1]
    if P_prior is None:
        P_prior = np.zeros(n)
    A = C.T @ C + lam**2 * np.eye(n)
    b = C.T @ m + lam**2 * P_prior
    return np.linalg.solve(A, b)


def l_curve(C: np.ndarray, m: np.ndarray, lams: np.ndarray,
            P_prior: np.ndarray | None = None):
    """Trace the L-curve and return (residual_norms, solution_norms, lams, i_corner).

    For each lambda we record the data-misfit ||C P - m|| and the solution norm
    ||P - P_prior||. Plotted log-log these trace an "L"; its corner is the best
    trade-off (smallest lambda that has stopped reducing the misfit while still
    holding the solution norm down). The corner is picked by maximum curvature of
    the log-log curve (Hansen's criterion), via a discrete curvature estimate.
    """
    res, sol, Ps = [], [], []
    for lam in lams:
        P = tikhonov_recover(C, m, lam, P_prior)
        Ps.append(P)
        res.append(np.linalg.norm(C @ P - m))
        sol.append(np.linalg.norm(P if P_prior is None else P - P_prior))
    res = np.array(res); sol = np.array(sol)

    # discrete curvature of the log-log L-curve (Menger curvature of triples)
    x, y = np.log(res + 1e-300), np.log(sol + 1e-300)
    curv = np.zeros_like(x)
    for k in range(1, len(x) - 1):
        x1, y1, x2, y2, x3, y3 = x[k-1], y[k-1], x[k], y[k], x[k+1], y[k+1]
        a = np.hypot(x2 - x1, y2 - y1)
        b = np.hypot(x3 - x2, y3 - y2)
        c = np.hypot(x3 - x1, y3 - y1)
        area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        curv[k] = 2 * area2 / (a * b * c + 1e-300)
    i_corner = int(np.argmax(curv))
    return res, sol, np.asarray(lams), i_corner, np.array(Ps)
