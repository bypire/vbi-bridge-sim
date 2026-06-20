# Vehicle-Bridge Interaction and Bridge Weigh-in-Motion

[![verify](https://github.com/bypire/vbi-bridge-sim/actions/workflows/verify.yml/badge.svg)](https://github.com/bypire/vbi-bridge-sim/actions/workflows/verify.yml)

A two-dimensional simulator that computes the dynamic response of a bridge as a vehicle crosses
it, and then inverts that response to recover the vehicle's axle weights. The inverse direction is
the principle behind Bridge Weigh-in-Motion (B-WIM). Every result is checked against a reference
that ships with the code: an analytical solution, a published benchmark, or a simulate-then-recover
round trip. The numerical core depends only on NumPy.

**Live demonstration: https://bypire.github.io/bridgebeat/**

![BridgeBeat: the measured natural frequency of the KW51 bridge over sixteen months](output/bridgebeat_shot.png)

## Background and motivation

Pavement and bridge damage grows with approximately the fourth power of axle load (the load
equivalency observed in the AASHO Road Test), so an axle at twice the legal weight causes on the
order of sixteen times the damage. The reference legal axle is the 80 kN Equivalent Single Axle
Load. B-WIM uses an already-instrumented bridge as a scale that weighs trucks at traffic speed,
which is cheaper and harder to evade than a static weigh station. The same dynamic response also
carries information about the structure itself: vibration-based monitoring infers structural
condition from the response and reduces the need for manual inspection of an ageing bridge stock.

## What it does

1. **Forward model.** A vehicle crosses a simply-supported bridge; the simulator returns the
   deflection time-history, the bending moment at a sensor section, the wheel-bridge contact force,
   and the dynamic amplification factor (DAF).
2. **Inverse model (B-WIM).** From that bridge response it recovers the axle weights using Moses'
   least-squares algorithm, including the separation of the axles of a multi-axle truck.
3. **Web viewer.** Animates the crossing, with the deck coloured by bending moment and the
   quarter-car visible, alongside live readouts and response plots. It opens by double-click and
   requires no server.

## Method

The bridge is an Euler-Bernoulli beam discretised with two-node beam finite elements (two degrees
of freedom per node), which gives a consistent mass matrix M and a bending stiffness K. The
equation of motion `M u'' + C u' + K u = F(t)` is integrated in time with explicit RK4 or with the
implicit, unconditionally stable Newmark-beta method. The moving load is placed at its current
position each step through the element's cubic Hermite shape functions, so `F(t)` is rebuilt
continuously. The vehicle is a two-degree-of-freedom quarter-car (sprung and unsprung mass,
suspension and tyre) coupled to the beam through the contact force, which makes the interaction
two-way. B-WIM inverts the measured bending moment against the section influence line by least
squares to recover the static axle loads.

## Verification

Each block is checked against an independent analytical reference.

| Check | Reference | Result |
|---|---|---|
| Static mid-span deflection | `P L^3 / 48 E I` (closed form) | rel. err about 1e-13 |
| Natural frequencies | `(n pi / L)^2 sqrt(EI / m_bar)` | mode 1 rel. err about 4e-7 |
| Moving constant force | Frýba closed-form solution | peak rel. err about 6e-6 |
| Quarter-car frequencies | analytic 2x2 eigenproblem | within FFT resolution |
| Newmark-beta vs Frýba | closed-form moving force | match about 2e-4 |
| Model order reduction (3 modes) | full-order response | reproduces to 0.16 percent |
| B-WIM weight (round trip) | known input weight | gross error below 1 percent |

## Validation and uncertainty

Verification establishes that the code reproduces a known solution. Validation asks a different
question: whether the model holds up against dynamics it was not fitted to, and against real
measured data. The two are kept separate.

- **Model-error-aware uncertainty** (`verify_coverage.py`). A self-consistent Bayesian coverage
  test, in which the synthetic data and the inverse both use the static influence line, returns
  95.3 percent coverage, but this only verifies the linear-Gaussian machinery. When the inverse is
  driven instead with the full coupled dynamic FEM, the recovered weight is biased by dynamic
  amplification (about +0.3 percent quasi-static, rising to +5 percent at 40 m/s). Because the fit
  residual measures goodness of fit rather than the dynamic scale error, the nominal 95 percent
  interval's true coverage collapses toward zero. The RMS dynamic bias over the highway speed band,
  about 3.1 percent, is the model-error term that restores calibration, and it is derived from the
  physics rather than assumed.
- **Regularisation of the ill-posed axle split** (`verify_regularize.py`). A closely spaced
  (tandem) axle pair makes two columns of the influence matrix nearly parallel, so the condition
  number of `C^T C` rises from about 9 at 8 m spacing to about 2000 at 0.5 m, and Moses' split
  amplifies measurement noise. Tikhonov regularisation with an L-curve-selected parameter cuts the
  per-axle scatter by about a factor of six (from 16 percent to 2.6 percent) for a small bias,
  while leaving the well-posed gross weight unchanged.
- **Validation against real data, KW51** (`verify_kw51.py`). Using the sixteen-month record of
  tracked modal frequencies from the instrumented KW51 bridge (public dataset), a real and dated
  retrofit shifts the frequencies by up to +2.6 percent, which confirms the monitoring premise on a
  real structure. Temperature swings the same frequencies by a comparable amount, up to about 3.8
  percent, and for some modes by more than the retrofit. Removing the temperature trend sharpens
  the detectability of the change (for example from 9.8 to 17.3 standard deviations for one mode).
  The synthetic study's limit, that a single global frequency is a blunt and environment-sensitive
  indicator, holds on the real bridge. The data (about 13 MB) is fetched from Zenodo and not
  committed; see the script docstring.
- **Multiple-presence B-WIM** (`multi_vehicle.py`, `verify_multi.py`, `verify_gate.py`). Real
  traffic places several trucks on the span at once. The convoy response is the exact superposition
  of the single-vehicle responses (verified to 1e-12). Single-vehicle B-WIM is then corrupted: a
  following truck at short headway inflates the recovered weight. A free-flow gate, which weighs
  only single-presence events, restores clean accuracy at the cost of throughput. Hardened with a
  full dynamic round trip and measurement noise (`verify_gate.py`), the clean single-presence floor
  is about +1.2 percent, against an ungated stream error that climbs from 37 percent at 12 trucks
  per minute to 72 percent at 30 per minute.

## From measurement to decision

A recovered weight is only useful if it drives a decision. `solver/fatigue.py` closes that loop:
steel-detail fatigue follows an S-N law (Eurocode EN 1993-1-9, slope 3; AASHTO pavement, slope 4),
so damage per pass scales as `(W / W_ref)^m`. A 2x overload therefore does 8 to 16 times the
damage. Over a simulated traffic stream the damage concentrates sharply: the heaviest 10 percent of
trucks cause about 45 percent of the fatigue damage, and the overloaded minority about half, and
B-WIM identifies exactly those trucks. As an asset-depreciation estimate, one heavy crossing
consumes about 10 EUR of bridge life against about 0.80 EUR for a legal pass, which is the
quantified case for self-funding overload enforcement.

## Interactive demonstration

`web/bridgebeat.html` (deployed at the live link above) presents the project as a five-part
interactive sequence: weighing a moving truck, diagnosing damage from the response spectrum,
following the measured frequency of the real KW51 bridge across its retrofit, the fatigue and cost
consequences of overloading, and a 500 m multi-span viaduct under a full traffic stream with a
running ledger of load, overloads and accumulated fatigue cost. A Simple/Math toggle presents
either plain-language captions or the underlying equations, error bars, DAF and references from the
same page. The earlier tester-style pages are available behind the Lab link.

## How to run

The Python core requires only NumPy. matplotlib is used solely by the `verify_*` scripts for
optional plots; the core never imports it.

```bash
# Forward simulator and B-WIM
python solver/verify.py               # static deflection and frequencies
python solver/verify_v2.py            # moving force vs Frýba, and the DAF
python solver/verify_v3.py            # quarter-car coupling, DAF vs speed
python solver/verify_bwim.py          # B-WIM single-axle weight recovery
python solver/verify_bwim_multi.py    # B-WIM multi-axle separation
python solver/export.py               # writes output/sim_data.js  -> web/index.html

# Drive-by structural health monitoring
python solver/verify_d1.py            # ISO 8608 road roughness and vehicle acceleration
python solver/verify_d2.py            # bridge frequency from vehicle acceleration
python solver/verify_d3.py            # contact-point response vs raw acceleration
python solver/verify_d4.py            # detection of a localised stiffness loss
python solver/verify_residual.py      # two-axle residual against road roughness
python solver/export_driveby.py       # writes output/driveby_data.js  -> web/driveby.html

# Realistic 40 m overpass, Rayleigh damping, interactive explorer
python solver/verify_p1.py            # 40 m overpass and damping check
python solver/export_explore.py       # writes output/explore_data.js  -> web/explore.html

# Operations dashboard
python -u solver/export_traffic.py    # weighs a traffic stream by Bayesian B-WIM  -> web/dashboard.html

# Computational methods
python solver/verify_newmark.py       # implicit Newmark-beta vs Frýba and stability
python solver/verify_bayesian.py      # Bayesian B-WIM: credible intervals and coverage
python solver/verify_mor.py           # model order reduction (modal truncation)

# Validation and uncertainty
python -u solver/verify_coverage.py   # model-error coverage; calibrates the uncertainty
python -u solver/verify_regularize.py # Tikhonov and L-curve for the ill-posed axle split
python -u solver/verify_kw51.py       # validation against real KW51 data (fetch first; see docstring)
python -u solver/verify_fatigue.py    # fatigue and economics of an overload
python -u solver/verify_multi.py      # multi-vehicle physics and the multiple-presence problem
python -u solver/verify_gate.py       # free-flow gate: dynamic round trip with noise and error bars
```

## Project layout

```
solver/                Python core (NumPy only) and verification scripts
  beam_fem.py          Euler-Bernoulli beam FEM (assembles M, K)
  moving_load.py       moving load, bending moment, Frýba reference, multi-axle march
  vehicle.py           quarter-car and the coupled beam-vehicle RK4 integrator
  bwim.py              influence line and Moses least-squares recovery
  export.py            writes output/sim_data.js for the web viewer
  verify_*.py          reference checks (run these to print the numbers)
web/                   vanilla JavaScript and inline SVG viewer (no framework, no server)
output/                generated data and plots
```

## Limitations

- One simply-supported single span; linear, small-displacement Euler-Bernoulli theory, with no
  buckling and no material or geometric nonlinearity.
- Road roughness is modelled (ISO 8608), but the drive-by frequency extraction is demonstrated in
  the low-roughness regime; heavy roughness requires the two-axle residual technique.
- Rayleigh damping is supported and verified; some batch exports run undamped for speed, since
  stiffness-proportional damping shrinks the explicit time step.
- A single quarter-car (one wheel path); the multi-axle B-WIM uses constant axle forces.
- B-WIM uses the analytical influence line; field systems calibrate it from a known truck, and real
  accuracy is degraded by roughness, multiple vehicles, and temperature.
- The drive-by method detects a global stiffness change but does not localise it.
- KW51 is a 115 m steel arch, not a simply-supported beam, so the real data validates the
  monitoring behaviour and method rather than the FEM geometry.

## References

1. Fourth power law and the AASHO Road Test (load equivalency; the 80 kN Equivalent Single Axle Load).
2. L. Frýba, *Vibration of Solids and Structures under Moving Loads*, Noordhoff, 1972.
3. F. Moses, "Weigh-in-motion system using instrumented bridges," *J. Transp. Eng.*, 105(3), 1979.
4. E. OBrien et al., "A regularised solution to the bridge weigh-in-motion equations,"
   *Int. J. Heavy Vehicle Systems*, 2009.
5. COST 323, *European Specification on Weigh-in-Motion of Road Vehicles* (accuracy classes).
6. K. Maes and G. Lombaert, "Monitoring data for railway bridge KW51," Zenodo 3745914, 2020.
7. Eurocode EN 1993-1-9, *Fatigue strength of steel structures*.
