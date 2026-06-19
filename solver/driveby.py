"""D-series — drive-by signal processing: get the bridge frequency from the
vehicle's acceleration.

The physics core (beam + quarter-car + road) already produces the vehicle
acceleration. This module is the SIGNAL-PROCESSING side: turn an acceleration
time-history into a spectrum and pick out the bridge's natural-frequency peak.

Kept deliberately simple (FFT + windowing + parabolic peak interpolation) so the
method is transparent; the cleverness in drive-by monitoring is in *what signal*
you feed it (raw vehicle vs contact-point response, D3), not in exotic transforms.

numpy only.
"""

from __future__ import annotations

import numpy as np


def acceleration_spectrum(t: np.ndarray, a: np.ndarray, zero_pad: int = 8):
    """One-sided amplitude spectrum of an acceleration record.

    t : uniformly-spaced time stamps [s]  (our RK4 march uses a fixed dt)
    a : acceleration samples
    zero_pad : pad to this multiple of the record length before the FFT, which
        interpolates the spectrum (finer peak location) without inventing
        resolution. True resolution is still 1 / record-length.

    Returns (freq [Hz], amplitude). The signal is de-trended (mean + linear) and
    Hann-windowed to suppress the big quasi-static "deflection bowl" ramp.
    """
    dt = t[1] - t[0]
    N = len(a)
    # remove mean + linear trend (the slow traversal ramp is not vibration)
    coef = np.polyfit(t, a, 1)
    a_dt = a - np.polyval(coef, t)
    a_dt = a_dt * np.hanning(N)

    Npad = int(N * zero_pad)
    spec = np.abs(np.fft.rfft(a_dt, n=Npad)) * (2.0 / N)
    freq = np.fft.rfftfreq(Npad, dt)
    return freq, spec


def peak_in_band(freq, spec, f_lo, f_hi):
    """Largest spectral peak within [f_lo, f_hi], refined by parabolic interp.

    Returns (f_peak, amplitude). Refinement fits a parabola to the 3 points
    around the discrete maximum — standard for sub-bin frequency estimation.
    """
    band = (freq >= f_lo) & (freq <= f_hi)
    idx = np.where(band)[0]
    k = idx[np.argmax(spec[idx])]
    if 0 < k < len(spec) - 1:
        y0, y1, y2 = spec[k - 1], spec[k], spec[k + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        df = freq[1] - freq[0]
        return float(freq[k] + delta * df), float(y1)
    return float(freq[k]), float(spec[k])


def _detrend(t, y):
    """Remove a linear trend (cheap high-pass; kills integration drift / ramp)."""
    return y - np.polyval(np.polyfit(t, y, 1), t)


def _cumtrapz(y, dt):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
    return out


def double_integrate(a, t):
    """Acceleration -> displacement by trapezoid integration, de-trended twice.

    Integrating an accelerometer signal drifts; we de-trend after each pass, a
    simple stand-in for the high-pass filtering used on real field data. The
    vibration content (e.g. the bridge frequency) survives; only the unobservable
    DC / ramp is removed.
    """
    dt = t[1] - t[0]
    v = _detrend(t, _cumtrapz(a, dt))
    x = _detrend(t, _cumtrapz(v, dt))
    return x


def contact_point_response(car, t, a_s, a_u):
    """Reconstruct the contact-point displacement r_c(t) from vehicle accels.

    From the two quarter-car equations the contact (tyre) force is the sum of
    'weight minus inertia', so

        F_c = (m_s+m_u) g - m_s a_s - m_u a_u
        r_c = z_u - F_c / k_t

    The only piece needing integration is the axle displacement z_u (double
    integral of the axle acceleration a_u). Inputs are exactly what two
    accelerometers (body + axle) give, plus the known vehicle parameters — the
    contact-point method of Yang et al. (2020). The result is de-trended (we want
    its vibration spectrum, not absolute position).
    """
    z_u = double_integrate(a_u, t)
    F_c = (car.m_s + car.m_u) * car.g - car.m_s * a_s - car.m_u * a_u
    r_c = z_u - F_c / car.k_t
    return _detrend(t, r_c)


def true_contact_response(car, t, z_u, contact_force):
    """The exact contact-point motion from the simulation, for verification.

    By definition F_c = k_t (z_u - r_c), so r_c = z_u - F_c/k_t using the recorded
    axle displacement and contact force. De-trended to match the reconstruction.
    """
    return _detrend(t, z_u - contact_force / car.k_t)


def residual_contact_response(car, t, a_s1, a_u1, a_s2, a_u2, spacing, speed):
    """Road-cancelling residual of two axles' contact-point responses.

    The rear axle rides the same road as the front, delayed by tau = spacing/speed.
    So r_front(t) and r_rear(t + tau) share the SAME road profile -> subtracting
    them cancels the (dominant) road roughness, leaving the difference in BRIDGE
    response the two axles see. The bridge frequency survives; the road does not.

    Returns (t_valid, residual) over the window where both shifted signals exist.
    """
    rc1 = contact_point_response(car, t, a_s1, a_u1)
    rc2 = contact_point_response(car, t, a_s2, a_u2)
    tau = spacing / speed
    # r_rear evaluated at t + tau (so it is at the same road point as r_front(t))
    rc2_shift = np.interp(t + tau, t, rc2, left=np.nan, right=np.nan)
    resid = rc1 - rc2_shift
    good = ~np.isnan(resid)
    return t[good], _detrend(t[good], resid[good])


def peak_prominence(freq, spec, f_target, half_width=0.6):
    """How much the peak near f_target stands out above the local background.

    Ratio of the peak amplitude in [f_target ± half_width] to the median
    amplitude just outside it. A value near 1 means the bridge peak is buried;
    large means it is clearly visible. A simple, honest 'detectability' number.
    """
    near = (freq >= f_target - half_width) & (freq <= f_target + half_width)
    outer = ((freq >= f_target - 4 * half_width) & (freq < f_target - half_width)) | \
            ((freq > f_target + half_width) & (freq <= f_target + 4 * half_width))
    if not near.any() or not outer.any():
        return 0.0
    return float(spec[near].max() / (np.median(spec[outer]) + 1e-30))
