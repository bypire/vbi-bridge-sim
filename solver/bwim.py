"""V4.5 — B-WIM capstone: weigh the truck from the bridge's response.

This is the "solves-a-problem" milestone. The forward simulator (V1–V3) told us
the bridge response given the vehicle. Here we run that backwards: given the
measured bridge response, recover the axle WEIGHT — the principle behind Bridge
Weigh-in-Motion, which turns an instrumented bridge into a scale to catch
overloaded trucks (road damage grows with the ~4th power of axle load).

Method — Moses' algorithm (1979)
--------------------------------
A strain gauge at a fixed section measures the bending moment time-history M(t)
as the truck crosses. For static loads the moment at the section is

    M(t) = sum_k  P_k * IL( x_k(t) )

where P_k are the (unknown) axle loads, x_k(t) their known positions (from speed
and axle spacing), and IL(x) is the moment *influence line* at the gauge section
(the moment there from a unit load at x). With the response sampled at many
instants this is an over-determined linear system  M = C P, solved for the loads
by least squares. The dynamic vibration rides on top of the static signal; the
least-squares fit over the whole crossing averages it out and returns the static
weight. The residual is B-WIM's "dynamic error".

numpy only.
"""

from __future__ import annotations

import numpy as np

from beam_fem import Beam


def moment_influence_line(beam: Beam, x_gauge: float, x_load: np.ndarray) -> np.ndarray:
    """Moment influence line at section x_gauge for a unit load at x_load.

    Exact for a simply-supported beam (a triangle peaking at the section):
        IL(x) = x (L - x_gauge) / L     for x <= x_gauge
        IL(x) = x_gauge (L - x) / L     for x >= x_gauge
    Loads off the span contribute nothing. Sagging-positive, matching
    `bending_moment_at`. (Real B-WIM calibrates this line from a known truck; we
    verify our analytical line against the FEM in verify_bwim.py.)
    """
    x = np.asarray(x_load, dtype=float)
    L = beam.L
    il = np.where(
        x <= x_gauge,
        x * (L - x_gauge) / L,
        x_gauge * (L - x) / L,
    )
    il[(x < 0.0) | (x > L)] = 0.0
    return il


def moses_recover(
    beam: Beam,
    x_gauge: float,
    front_position: np.ndarray,
    measured_moment: np.ndarray,
    axle_offsets=(0.0,),
) -> tuple[np.ndarray, np.ndarray]:
    """Recover axle loads from a measured moment history (Moses least squares).

    beam, x_gauge   : bridge and gauge section.
    front_position  : position of the FIRST axle at each sample [m] (= speed*t).
    measured_moment : bending moment at the gauge at each sample [N m].
    axle_offsets    : spacing of each axle BEHIND the first one [m]; (0.0,) for a
                      single axle. Axle k sits at front_position - offset_k.

    Builds the influence matrix C (n_samples x n_axles) and solves
    measured = C @ P for the loads P [N] via numpy.linalg.lstsq.

    Returns (P, fitted_moment) — recovered loads and the moment the fit
    reproduces, C @ P (handy for plotting the quality of the fit).
    """
    front_position = np.asarray(front_position, dtype=float)
    cols = []
    for offset in axle_offsets:
        x_k = front_position - offset
        cols.append(moment_influence_line(beam, x_gauge, x_k))
    C = np.column_stack(cols)                       # (n_samples, n_axles)

    P, *_ = np.linalg.lstsq(C, measured_moment, rcond=None)
    fitted = C @ P
    return P, fitted


def bayesian_axle_loads(beam, x_gauge, front_position, measured_moment,
                        axle_offsets=(0.0,)):
    """Bayesian (linear-Gaussian) B-WIM: posterior over the axle loads.

    The model is linear, measured = C P + noise, noise ~ N(0, sigma^2 I). With a
    flat prior the posterior over the loads P is Gaussian:

        mean  = (C^T C)^{-1} C^T measured        (= Moses' least-squares point)
        cov   = sigma^2 (C^T C)^{-1}             (the NEW part: the uncertainty)

    sigma^2 is estimated from the fit residual (so it absorbs not just electrical
    noise but the dynamic/model mismatch the static influence line cannot explain
    — which is the dominant real B-WIM error, honestly folded into the error bars).

    Returns (mean, cov, sigma): posterior mean loads [N], covariance [N^2], and
    the estimated noise std [N m]. The point estimate matches `moses_recover`;
    this adds calibrated uncertainty on top.
    """
    front_position = np.asarray(front_position, dtype=float)
    C = np.column_stack([moment_influence_line(beam, x_gauge, front_position - o)
                         for o in axle_offsets])
    CtC = C.T @ C
    CtC_inv = np.linalg.inv(CtC)
    mean = CtC_inv @ (C.T @ measured_moment)
    resid = measured_moment - C @ mean
    dof = max(1, len(measured_moment) - len(axle_offsets))
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * CtC_inv
    return mean, cov, float(np.sqrt(sigma2))


def gross_weight_posterior(mean, cov):
    """Posterior mean and std of the GROSS weight (sum of axle loads).

    Gross = 1^T P, so mean = sum(mean) and var = 1^T cov 1 (= cov.sum()).
    """
    mu = float(np.sum(mean))
    var = float(np.ones(len(mean)) @ cov @ np.ones(len(mean)))
    return mu, float(np.sqrt(max(var, 0.0)))


def prob_exceed(mu, sigma, limit):
    """P(weight > limit) for a Gaussian posterior (the overload probability)."""
    import math
    if sigma <= 0:
        return 1.0 if mu > limit else 0.0
    return 0.5 * math.erfc((limit - mu) / (sigma * math.sqrt(2.0)))


def add_measurement_noise(
    moment: np.ndarray, noise_frac: float, rng: np.random.Generator
) -> np.ndarray:
    """Add zero-mean Gaussian noise with std = noise_frac * peak |moment|.

    Models strain-gauge / electrical noise as a fraction of the peak signal.
    """
    sigma = noise_frac * np.max(np.abs(moment))
    return moment + rng.normal(0.0, sigma, size=moment.shape)
