# Bridge Weigh-in-Motion as a regularized Bayesian inverse problem: verification, validation against measured data, and model-error-aware uncertainty

**Extended abstract**
*B.Sc. Computational Engineering Science (CES), RWTH Aachen*
*Live demo: https://bypire.github.io/bridgebeat · Source: github.com/bypire/vbi-bridge-sim (private, access on request)*

---

## 1. Motivation

Road and bridge damage grows with roughly the **fourth power of axle load** (AASHO
Road Test): a truck at twice the legal axle weight does about **16×** the structural
damage. **Bridge Weigh-in-Motion (B-WIM)** turns an already-instrumented bridge into
a scale that weighs trucks *at traffic speed* by inverting the measured structural
response — cheaper and harder to evade than static weigh stations. The same dynamic
response also carries information about the structure's *health* (drive-by /
vibration-based monitoring). This project builds the **forward** physics (a vehicle
crossing a bridge), runs it **backwards** to recover the axle weights as an inverse
problem, and validates the result against an analytical reference, a regularization
analysis, and **real measured bridge data**. The governing rule: every claim is
checked against a reference. The core is numpy-only.

## 2. Method

The bridge is a **simply-supported Euler–Bernoulli beam** discretised with 2-node
beam finite elements (2 DOF/node), giving a consistent mass matrix **M** and bending
stiffness **K**; **M ü + C u̇ + K u = F(t)** with optional Rayleigh damping. The
**moving load** is mapped to its position each step via cubic Hermite shape functions.
The **vehicle** is a 2-DOF **quarter-car** coupled to the beam through the contact
force (two-way interaction). Time integration uses explicit **RK4** and implicit
**Newmark-β** (the latter unconditionally stable). The reference structure is a 40 m
highway overpass (f₁ ≈ 2.6 Hz, ζ = 2 %). The **inverse** problem recovers static axle
loads by fitting the measured bending moment to the section **influence line** (Moses'
least squares), extended to a **Tikhonov-regularized** and a **Bayesian** estimator.

## 3. Verification (the math is correct)

Each building block is checked against an independent **analytical** reference; if the
code disagrees, the code is wrong.

| Check | Reference | Result |
|---|---|---|
| Static mid-span deflection | `P L³ / 48 E I` | rel. err ≈ **1 × 10⁻¹³** |
| Natural frequencies | `(nπ/L)² √(EI/m̄)` | mode 1 rel. err ≈ **4 × 10⁻⁷** |
| Moving constant force | **Frýba** closed form | peak rel. err ≈ **6 × 10⁻⁶** |
| Quarter-car frequencies | analytic 2×2 eigenproblem | within FFT resolution |
| Damping ζ | free-vibration log-decrement | within **0.1 %** |
| Newmark-β vs Frýba | closed-form moving force | match to **≈ 2 × 10⁻⁴** |
| Model order reduction (3 modes) | full-order response | reproduces to **0.16 %** |

## 4. Validation and uncertainty (the model means something)

Verification is not validation. Three results separate "the math is right" from "the
answer is trustworthy."

**4.1 Model-error-aware uncertainty.** Casting B-WIM as a linear-Gaussian inverse
gives a recovered weight with a 95 % credible interval. A *self-consistent* coverage
test (forward and inverse both use the static influence line) returns 95.3 % coverage
— but this only verifies the linear-Gaussian machinery. The honest test replaces the
forward model with the **full coupled dynamic FEM**: the recovered weight is then
biased by dynamic amplification (**+0.3 % quasi-static rising to +5 % at 40 m/s**),
and because the residual σ measures goodness-of-*fit* (the signal still looks
influence-line-shaped) rather than the dynamic *scale* error, the nominal-95 %
interval's true coverage **collapses to ≈ 0 %**. The RMS dynamic bias over the highway
band (**≈ 3.1 %**) is the model-error term that restores calibration — and it matches
the value independently used in the operational dashboard, so that uncertainty is now
*calibrated from physics*, not assumed.

**4.2 Regularizing the ill-posed axle split.** Moses' method recovers **gross** weight
to < 1 % but splits **closely spaced (tandem) axles** poorly. This is purely
linear-algebraic: a tandem makes two columns of the influence matrix nearly parallel,
so `cond(CᵀC)` rises sharply (≈ 9 at 8 m spacing → ≈ 2000 at 0.5 m) and the split
amplifies noise. **Tikhonov regularization** with an **L-curve**-chosen weight cuts the
per-axle scatter **≈ 6×** (16 % → 2.6 %) for a small bias, while leaving the well-posed
gross weight unchanged — the bias–variance trade-off made explicit.

**4.3 Validation against real data (KW51).** The 16-month tracked-modal-frequency
record of the instrumented KW51 railway bridge (Leuven; public dataset) is ground
truth no model can argue with. It confirms the drive-by-SHM premise — a real,
dated **retrofit** (stiffening) shifts the modal frequencies by up to **+2.6 %** — and
exposes its dominant confounder: **temperature** swings the same frequencies by a
comparable amount (**up to ≈ 3.8 %**, with some modes' temperature swing *exceeding*
the retrofit jump). Removing the temperature trend (fit on the healthy period) sharpens
the change from a raw to a temperature-corrected residual (e.g. **9.8 σ → 17.3 σ** for
one mode). This is exactly the limit the synthetic damage study predicted — frequency
is a blunt, environment-contaminated indicator — now confirmed in the field.

**4.4 From weight to decision (fatigue economics).** A recovered weight is only useful
if it drives a decision. Steel-detail fatigue follows an S-N law with slope *m* (Eurocode
EN 1993-1-9, *m* = 3; AASHTO pavement *m* = 4), so damage per pass scales as
`(W/W_ref)ᵐ`: a **2× overload does 8×–16×** the damage (exact). Over a simulated traffic
stream this concentrates sharply — the heaviest **10 %** of trucks cause **≈ 45 %** of the
fatigue damage and the overloaded minority **≈ half** — and B-WIM identifies exactly those
trucks. An asset-depreciation estimate puts one heavy crossing at **≈ €10** of bridge life
(vs €0.80 for a legal pass): the quantified case for self-funding overload enforcement.

## 5. Limitations (honest scope)

One simply-supported single span; linear, small-displacement Euler–Bernoulli theory.
A single quarter-car (one wheel path); the multi-axle B-WIM uses constant axle forces.
The drive-by study **detects** a global stiffness change but does **not localize** it —
a single fundamental frequency cannot, by construction. KW51 is a 115 m steel bowstring
**arch**, not a simply-supported beam, so it validates the SHM *behaviour and method*,
not the FEM geometry. B-WIM uses the analytical influence line; field systems calibrate
it from a known truck and lose accuracy to roughness, multiple vehicles, and temperature.

## 6. References

1. Fourth power law / AASHO Road Test; Equivalent Single Axle Load (80 kN ESAL).
2. L. Frýba, *Vibration of Solids and Structures under Moving Loads* — moving-load closed form.
3. F. Moses, "Weigh-in-motion system using instrumented bridges," *J. Transp. Eng.*, 1979.
4. OBrien et al., "A regularised solution to the bridge weigh-in-motion equations."
5. P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse problems."
6. Maes & Lombaert, "Monitoring data for railway bridge KW51," Zenodo 3745914 (2020).
7. COST323, *European Specification on Weigh-in-Motion* — accuracy classes.
