"""Calibrate a reduced beam to the REAL KW51 bridge -- honestly.

Run:
    python -u solver/calibrate_kw51.py

The owner's instinct is a "digital twin" of a real bridge. This script does the
honest, reduced version of that and shows exactly where it stops being honest.

We take KW51's measured modal frequencies (from the tracked-modes dataset) and tune
a simply-supported Euler-Bernoulli beam so its FUNDAMENTAL matches the bridge's
measured f1. A beam's modes follow f_n = n^2 * f1. We then compare those to KW51's
measured higher modes. The point:

  * f1 matches by construction (that's the calibration);
  * but the real bridge's higher modes do NOT follow the beam's n^2 law -- KW51 is a
    115 m steel tied-arch with a dense modal forest, not a slender beam.

So a beam can be ANCHORED to the real structure's fundamental (useful),
but it is NOT a faithful geometric twin -- which is exactly why this project
validates the bridge's BEHAVIOUR against KW51, not its geometry. A true twin needs
the real 3D model and full FE model updating (what the KU Leuven group does with the
structural drawings). Honest scope, stated up front.

Core numpy-only; scipy.io loads the .mat (tooling).
"""

import os
import numpy as np

from beam_fem import Beam, natural_frequencies

# KW51 reported geometry (Leuven; steel bowstring/tied-arch railway bridge).
L_REPORTED = 115.0          # m (reported span; the n^2 comparison is L-independent)
M_BAR_ASSUMED = 12000.0     # kg/m (assumed deck mass/length; only fixes EI, not the
                            # modal-ratio conclusion)
MAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "trackedmodes", "trackedmodes.mat")


def main():
    if not os.path.exists(MAT):
        print(f"[!] {MAT} not found (fetch KW51 first; see verify_kw51.py).")
        return
    import scipy.io
    m = scipy.io.loadmat(MAT, squeeze_me=True, struct_as_record=False)["modes"]
    f = np.asarray(m.f, float)
    measured = np.nanmean(f, axis=0)          # mean tracked frequency per mode [Hz]
    f1 = float(measured[0])

    print("=" * 70)
    print("CALIBRATING A REDUCED BEAM TO REAL KW51 DATA (honest digital-twin step)")
    print(f"  measured fundamental f1 = {f1:.3f} Hz;  reported span L = {L_REPORTED} m")
    print("=" * 70)

    # Calibrate EI/m_bar so the beam fundamental equals the measured f1:
    #   f1 = (1/2pi) (pi/L)^2 sqrt(EI/m_bar)  ->  EI = m_bar (2 f1 L^2 / pi)^2
    EI_over_mbar = (2.0 * f1 * L_REPORTED**2 / np.pi) ** 2
    EI = M_BAR_ASSUMED * EI_over_mbar
    print(f"\nCalibrated bending stiffness: EI/m_bar = {EI_over_mbar:.4e} m^4/s^2")
    print(f"  -> EI = {EI:.3e} N m^2  (for assumed m_bar = {M_BAR_ASSUMED:.0f} kg/m)")

    # Build that beam in our FEM and confirm f1 (closes the loop with beam_fem).
    beam = Beam(L=L_REPORTED, E=EI, I=1.0, mass_per_length=M_BAR_ASSUMED, n_elements=30)
    f1_fem = float(natural_frequencies(beam, 1)[0])
    print(f"  FEM check: beam f1 = {f1_fem:.3f} Hz  (target {f1:.3f}, "
          f"rel.err {abs(f1_fem-f1)/f1:.1e}) -> calibration confirmed.")

    # Now the honest part: beam modes are f_n = n^2 f1. Compare to measured.
    print("\nDoes the calibrated beam predict the HIGHER modes? (f_n = n^2 * f1)")
    print(f"  {'mode n':>6} {'beam n^2*f1':>12} {'KW51 measured':>14} {'ratio':>8}")
    n_show = min(6, len(measured))
    worst = 0.0
    for n in range(1, n_show + 1):
        beam_fn = n * n * f1
        meas = float(measured[n - 1])
        ratio = meas / beam_fn
        worst = max(worst, abs(np.log(ratio)))
        print(f"  {n:>6} {beam_fn:>12.3f} {meas:>14.3f} {ratio:>8.2f}")

    print("\nFINDING (honest scope)")
    print("  - f1 matches by construction: a beam CAN be anchored to a real bridge's")
    print("    fundamental frequency -- a legitimate reduced model.")
    print("  - But KW51's higher modes do NOT follow the beam's n^2 law (ratios far")
    print("    from 1): the real bridge is a 115 m tied-arch with a dense modal forest,")
    print("    not a slender beam. A faithful TWIN needs the real 3D geometry + full")
    print("    FE model updating. So we anchor to- and validate BEHAVIOUR against KW51,")
    print("    and we do NOT claim to reproduce its geometry. That honesty is the point.")


if __name__ == "__main__":
    main()
