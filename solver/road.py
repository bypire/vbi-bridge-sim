"""D1 — ISO 8608 random road-roughness profiles.

Drive-by monitoring lives or dies on the road surface: the roughness under the
tyre excites the vehicle far more than the bridge does, and separating the two
is THE open problem in the field. To study that honestly we need realistic,
standardised roughness — so we generate profiles to ISO 8608.

ISO 8608 fits the road's vertical-displacement power spectral density (PSD) as a
straight line on log–log axes against spatial frequency n [cycles/m]:

    Gd(n) = Gd(n0) * (n / n0)^(-w),     n0 = 0.1 cycles/m,  w = 2

The "degree of roughness" Gd(n0) sets the class (each class is ×4 the previous):

    A (very good) 16e-6,  B 64e-6,  C 256e-6,  D 1024e-6   [m^3]   (class centres)

We synthesise a profile by superposing harmonics whose amplitudes come from the
PSD and whose phases are random — the standard sum-of-sinusoids method:

    h(x) = sum_i sqrt(2 Gd(n_i) dn) * cos(2*pi*n_i*x + phi_i)

numpy only. A deterministic seed keeps results reproducible.
"""

from __future__ import annotations

import numpy as np

# class-centre degrees of roughness Gd(n0) [m^3] at n0 = 0.1 cycles/m
ISO8608_GD = {
    "A": 16e-6, "B": 64e-6, "C": 256e-6, "D": 1024e-6,
    "E": 4096e-6, "F": 16384e-6,
}
N0 = 0.1  # reference spatial frequency [cycles/m]


def iso8608_gd_n0(road_class: str) -> float:
    return ISO8608_GD[road_class.upper()]


class RoadProfile:
    """A synthesised ISO 8608 profile, sampleable at any position x [m]."""

    def __init__(self, x: np.ndarray, h: np.ndarray, road_class: str):
        self.x = x
        self.h = h
        self.road_class = road_class

    def __call__(self, xq):
        """Elevation h at position xq [m] (linear interp; 0 outside the span)."""
        return np.interp(xq, self.x, self.h, left=0.0, right=0.0)


def generate_iso8608(
    length: float,
    road_class: str = "A",
    dx: float = 0.05,
    n_min: float = 0.011,
    n_max: float = 2.83,
    n_harmonics: int = 1000,
    seed: int = 1,
) -> RoadProfile:
    """Synthesise an ISO 8608 profile of the given class over [0, length].

    n_min..n_max span the wavelengths the standard covers (~0.35 m to ~90 m).
    A fixed seed makes the (random-phase) profile reproducible.
    """
    gd_n0 = iso8608_gd_n0(road_class)
    rng = np.random.default_rng(seed)

    x = np.arange(0.0, length + dx, dx)
    # harmonic spatial frequencies and their PSD-derived amplitudes
    n = np.linspace(n_min, n_max, n_harmonics)
    dn = (n_max - n_min) / (n_harmonics - 1)
    gd = gd_n0 * (n / N0) ** (-2.0)                  # ISO 8608 PSD, w = 2
    amp = np.sqrt(2.0 * gd * dn)                      # harmonic amplitudes
    phase = rng.uniform(0.0, 2.0 * np.pi, n_harmonics)

    # h(x) = sum_i amp_i cos(2 pi n_i x + phi_i)   (vectorised over x)
    h = (amp[None, :] * np.cos(2.0 * np.pi * np.outer(x, n) + phase[None, :])).sum(axis=1)
    return RoadProfile(x, h, road_class.upper())


def estimate_psd(h: np.ndarray, dx: float):
    """One-sided displacement PSD of a profile vs spatial frequency [cycles/m].

    Returns (n, Gd) suitable for checking the ISO 8608 slope/level. Uses a plain
    periodogram (Hann-windowed) — enough to confirm the ~n^-2 trend.
    """
    N = len(h)
    win = np.hanning(N)
    hw = (h - h.mean()) * win
    # window power-loss correction
    u = (win ** 2).mean()
    fft = np.fft.rfft(hw)
    n = np.fft.rfftfreq(N, dx)                        # cycles/m
    psd = (np.abs(fft) ** 2) / (N * u) * dx * 2.0     # one-sided, m^2/(cycle/m)=m^3
    return n[1:], psd[1:]                             # drop the DC bin
