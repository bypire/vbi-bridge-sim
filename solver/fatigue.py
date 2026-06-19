"""Fatigue & economics layer: turn a recovered axle weight into bridge life spent.

This is the "why it matters" layer on top of B-WIM. Weighing a truck is only useful
if the number means something; here the meaning is FATIGUE DAMAGE and its cost.

The physics (standard):
  * A passing truck induces a bending-stress range Dsigma at a fatigue-critical
    detail, proportional to its axle/gross load.
  * Steel-detail fatigue follows an S-N (Wohler) curve N(Dsigma) = N_C (Dsigma_C/Dsigma)^m
    with slope m (Eurocode EN 1993-1-9 uses m = 3 below the knee). So the cycles to
    failure fall with the m-th power of load, and the damage per pass rises with the
    m-th power:  d(W) ~ (W/W_ref)^m   (a "load-equivalency factor", LEF).
  * Pavement design (AASHTO 4th-power law) is the same idea with m = 4.
  * Miner's rule sums it linearly: total damage D = sum_i d_i; failure at D = 1.

The consequence is the whole point: because damage grows with the CUBE-to-4TH power
of load, a small fraction of the heaviest trucks causes most of the fatigue
consumption -- and B-WIM identifies exactly those trucks. The ratios (2x load -> 8x
or 16x damage) and the Pareto concentration are assumption-free; the euro figures are
an explicit asset-depreciation illustration with stated inputs.

numpy only.
"""

from __future__ import annotations

import numpy as np

# Eurocode EN 1993-1-9 lower-slope exponent for welded steel details.
M_STEEL = 3.0
# AASHTO pavement "fourth power law".
M_PAVEMENT = 4.0


def load_equivalency(W, W_ref, m=M_STEEL):
    """Damage per pass relative to one reference (legal) pass: (W/W_ref)^m.

    W may be a scalar or an array of gross/axle loads [N]; W_ref is the reference
    legal load [N]. A legal-weight truck returns 1.0; double the load returns 2^m.
    """
    return (np.asarray(W, dtype=float) / W_ref) ** m


def damage_per_pass(W, W_ref, n_design, m=M_STEEL):
    """Fraction of the bridge's fatigue life consumed by one pass of load W.

    n_design = number of equivalent legal passes the detail is designed to survive
    (Miner failure at cumulative D = 1). So one legal pass spends 1/n_design, and a
    heavier truck spends LEF/n_design.
    """
    return load_equivalency(W, W_ref, m) / float(n_design)


def cost_per_pass(W, W_ref, n_design, replacement_cost, m=M_STEEL):
    """Euro value of the fatigue life one pass consumes (asset depreciation).

    Treats the structure as a depreciating asset: spending a fraction f of its
    fatigue life costs f * replacement_cost. Illustrative; inputs are explicit.
    """
    return damage_per_pass(W, W_ref, n_design, m) * float(replacement_cost)


def stream_damage(weights, W_ref, n_design, m=M_STEEL):
    """Per-truck damage fractions for a traffic stream (array of gross loads [N])."""
    return damage_per_pass(np.asarray(weights, float), W_ref, n_design, m)


def pareto_concentration(damages, top_frac=0.10):
    """Share of total damage caused by the heaviest `top_frac` of passes.

    Returns (share_of_damage, share_of_count) -- e.g. (0.55, 0.10) means the top
    10% of trucks by damage cause 55% of the fatigue consumption.
    """
    d = np.sort(np.asarray(damages, float))[::-1]      # heaviest first
    k = max(1, int(round(top_frac * len(d))))
    return float(d[:k].sum() / d.sum()), float(k / len(d))


def lorenz(damages):
    """Cumulative (truck-fraction, damage-fraction) curve, trucks sorted lightest->heaviest.

    A diagonal = every truck equally damaging; a curve bowed to the bottom-right =
    damage concentrated in the heaviest trucks (the fatigue reality).
    """
    d = np.sort(np.asarray(damages, float))            # lightest first
    cd = np.cumsum(d) / d.sum()
    x = np.arange(1, len(d) + 1) / len(d)
    return np.concatenate([[0], x]), np.concatenate([[0], cd])
