"""Reproduce every number and figure in the technical note from the solver core.

Run from the repository root:
    python note/make_note_figures.py

It imports the same numpy-only solver used throughout the project (no fitting,
no hidden constants) and writes:
    note/figures/validation.pdf      (+ .png)   FEM/Newmark/modal vs Fryba
    note/figures/restriction.pdf     (+ .png)   explicit RK4 stable dt vs damping
    note/figures/remedies.pdf        (+ .png)   modal convergence + step-count cost
    note/data.json                              all headline numbers

The single slow step is one explicit-RK4 march of the damped 40 m bridge
(~10^5 steps): that slowness is precisely the point of the note.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "solver"))

from beam_fem import (  # noqa: E402
    Beam,
    assemble,
    free_dofs,
    natural_frequencies,
    rayleigh_alpha_beta,
)
from moving_load import (  # noqa: E402
    fryba_deflection,
    integrate_moving_force,
)
from newmark import integrate_newmark_moving_force  # noqa: E402
from mor import integrate_modal_moving_force, modal_basis  # noqa: E402

FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# --- two beams, exactly as in the project's verification scripts -----------
# 20 m beam: the clean Fryba ground truth (undamped).
L1, E1, I1, MB1, P1 = 20.0, 2.1e11, 0.02, 2000.0, 1.0e5
V1 = 30.0
# 40 m realistic damped highway overpass: the stability/cost story.
L2, E2, I2, MB2 = 40.0, 2.1e11, 0.40, 12000.0
P2, V2, ZETA = 3.0e5, 25.0, 0.02

data: dict = {}


def omega_max_of(beam: Beam) -> float:
    """Highest FEM circular frequency (sets the explicit step), as in the RK4
    integrator: eigenvalues of A = M_ff^{-1} K_ff."""
    M, K = assemble(beam)
    free = free_dofs(beam)
    A = np.linalg.solve(M[np.ix_(free, free)], K[np.ix_(free, free)])
    return float(np.sqrt(np.max(np.linalg.eigvals(A).real)))


def main() -> None:
    beam1 = Beam(L=L1, E=E1, I=I1, mass_per_length=MB1, n_elements=20)
    beam2 = Beam(L=L2, E=E2, I=I2, mass_per_length=MB2, n_elements=20)
    nf1 = len(free_dofs(beam1))

    # =====================================================================
    # (A) Accuracy of all three integrators vs the Fryba closed form (20 m)
    # =====================================================================
    rn = integrate_newmark_moving_force(beam1, P1, V1, dt=1.0e-3)
    w_fry = fryba_deflection(beam1, P1, V1, rn.t)
    peak_nm = rn.w_mid[np.argmax(np.abs(rn.w_mid))]
    peak_fr = w_fry[np.argmax(np.abs(w_fry))]
    rel_newmark = abs(peak_nm - peak_fr) / abs(peak_fr)

    full = integrate_moving_force(beam1, P1, V1)               # explicit RK4
    w_fry_full = fryba_deflection(beam1, P1, V1, full.t)
    peak_fry_full = w_fry_full[np.argmax(np.abs(w_fry_full))]
    rel_rk4 = abs(full.w_mid[np.argmax(np.abs(full.w_mid))] - peak_fry_full) \
        / abs(peak_fry_full)

    rs = [1, 2, 3, 5, 10]
    modal_err = {}
    modal_hist = {}
    for r in rs:
        rm = integrate_modal_moving_force(beam1, P1, V1, r)
        pk = rm.w_mid[np.argmax(np.abs(rm.w_mid))]
        modal_err[r] = abs(pk - peak_fry_full) / abs(peak_fry_full)
        modal_hist[r] = rm
    rm3 = modal_hist[3]

    data["accuracy_20m"] = {
        "static_midspan_m": P1 * L1**3 / (48.0 * E1 * I1),
        "daf_fryba": float(full.daf),
        "rel_err_rk4_vs_fryba": float(rel_rk4),
        "rel_err_newmark_vs_fryba": float(rel_newmark),
        "rel_err_modal_vs_fryba": {str(r): float(modal_err[r]) for r in rs},
    }

    # Newmark dt^2 convergence
    conv = {}
    for dt in [4e-3, 2e-3, 1e-3, 5e-4]:
        r = integrate_newmark_moving_force(beam1, P1, V1, dt=dt)
        wf = fryba_deflection(beam1, P1, V1, r.t)
        pk = r.w_mid[np.argmax(np.abs(r.w_mid))]
        pf = wf[np.argmax(np.abs(wf))]
        conv[dt] = abs(pk - pf) / abs(pf)
    data["newmark_convergence"] = {f"{k:.0e}": float(v) for k, v in conv.items()}

    # =====================================================================
    # (B) Modal truncation: DOF count and stable-step gain (20 m)
    # =====================================================================
    om_full1 = omega_max_of(beam1)
    om_3_1 = float(modal_basis(beam1, 3)[0][-1])
    data["modal_reduction_20m"] = {
        "n_free_dofs": nf1,
        "omega_max_full_radps": om_full1,
        "omega_3_radps": om_3_1,
        "dof_reduction": nf1 / 3.0,
        "undamped_step_gain": om_full1 / om_3_1,            # dt ~ 1/omega
        "damped_step_gain": (om_full1 / om_3_1) ** 2,        # dt ~ 1/omega^2
    }

    # =====================================================================
    # (C) The damping-dominated explicit step restriction (40 m bridge)
    # =====================================================================
    om_max2 = omega_max_of(beam2)
    f12 = natural_frequencies(beam2, 2)
    f1_40 = float(f12[0])
    _, beta2 = rayleigh_alpha_beta(beam2, ZETA)
    dt_undamped_cap = 1.8 / om_max2                  # imaginary-axis RK4 limit
    dt_damped_cap = 2.2 / (beta2 * om_max2**2)        # real-axis limit (damping)
    slowdown = dt_undamped_cap / dt_damped_cap

    # dt cap as a function of damping ratio (analytic)
    zetas = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05])
    dt_cap = []
    for z in zetas:
        if z == 0.0:
            dt_cap.append(dt_undamped_cap)
        else:
            _, b = rayleigh_alpha_beta(beam2, float(z))
            dt_cap.append(2.2 / (b * om_max2**2))
    dt_cap = np.array(dt_cap)

    # actual marches on the damped 40 m bridge
    print("  [slow] explicit RK4 march of the damped 40 m bridge ...", flush=True)
    t0 = time.time()
    rk4 = integrate_moving_force(beam2, P2, V2, damping_ratio=ZETA)
    t_rk4 = time.time() - t0
    nm = integrate_newmark_moving_force(beam2, P2, V2, dt=4.0e-3,
                                        damping_ratio=ZETA)
    rk4_steps = len(rk4.t) - 1
    nm_steps = len(nm.t) - 1

    data["stability_40m"] = {
        "f1_hz": f1_40,
        "omega_max_radps": om_max2,
        "rayleigh_beta": beta2,
        "zeta": ZETA,
        "dt_undamped_cap_s": dt_undamped_cap,
        "dt_damped_cap_s": dt_damped_cap,
        "explicit_slowdown_factor": slowdown,
        "rk4_dt_s": float(rk4.dt),
        "newmark_dt_s": float(nm.dt),
        "newmark_over_rk4_step": float(nm.dt / rk4.dt),
        "rk4_steps": rk4_steps,
        "newmark_steps": nm_steps,
        "step_count_ratio": rk4_steps / nm_steps,
        "daf_rk4": float(rk4.daf),
        "daf_newmark": float(nm.daf),
        "daf_rel_diff": abs(rk4.daf - nm.daf) / rk4.daf,
        "rk4_wall_s": t_rk4,
        "crossing_time_s": float(rk4.crossing_time),
    }

    # =====================================================================
    # Figure 1 — validation
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(rn.t, w_fry * 1e3, "k-", lw=2.8, label="Fryba (exact)")
    ax1.plot(full.t, full.w_mid * 1e3, "-", color="tab:green", lw=1.0,
             alpha=0.9, label="explicit RK4")
    ax1.plot(rn.t, rn.w_mid * 1e3, "--", color="tab:blue", lw=1.4,
             label="Newmark-$\\beta$")
    ax1.plot(rm3.t, rm3.w_mid * 1e3, ":", color="tab:red", lw=1.8,
             label="modal, $r{=}3$")
    ax1.invert_yaxis()
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("mid-span deflection [mm]")
    ax1.set_title("(a) 20 m beam: all methods vs Fryba")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    dts = sorted(conv.keys(), reverse=True)
    errs = [conv[d] for d in dts]
    ax2.loglog(dts, errs, "o-", color="tab:blue", label="Newmark peak error")
    guide = errs[0] * (np.array(dts) / dts[0]) ** 2
    ax2.loglog(dts, guide, "k--", lw=1.0, label="$\\mathcal{O}(\\Delta t^2)$")
    ax2.set_xlabel("time step $\\Delta t$ [s]")
    ax2.set_ylabel("peak rel. error vs Fryba")
    ax2.set_title("(b) Newmark convergence")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "validation.pdf"))
    fig.savefig(os.path.join(FIGDIR, "validation.png"), dpi=130)
    plt.close(fig)

    # =====================================================================
    # Figure 2 — the damping-dominated step restriction
    # =====================================================================
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    zz = zetas * 100.0
    ax.plot(zz[1:], dt_cap[1:], "o-", color="tab:red",
            label="damped cap $2.2/(\\beta\\,\\omega_{\\max}^2)$")
    ax.axhline(dt_undamped_cap, ls="--", color="tab:green",
               label="undamped cap $1.8/\\omega_{\\max}$")
    ax.scatter([2.0], [dt_damped_cap], s=80, facecolors="none",
               edgecolors="k", zorder=5)
    ax.annotate(
        f"$\\zeta=2\\%$:\n$\\Delta t$ drops {slowdown:.0f}$\\times$",
        xy=(2.0, dt_damped_cap), xytext=(2.9, dt_undamped_cap * 0.5),
        arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("Rayleigh damping ratio $\\zeta$ [%]")
    ax.set_ylabel("max stable explicit step $\\Delta t$ [s]")
    ax.set_title("Explicit RK4 step is throttled by damping (40 m bridge)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "restriction.pdf"))
    fig.savefig(os.path.join(FIGDIR, "restriction.png"), dpi=130)
    plt.close(fig)

    # =====================================================================
    # Figure 3 — the two remedies
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.loglog(rs, [modal_err[r] for r in rs], "o-", color="tab:purple")
    ax1.set_xlabel("retained modes $r$")
    ax1.set_ylabel("peak rel. error vs Fryba")
    ax1.set_title("(a) modal truncation convergence (20 m)")
    ax1.grid(alpha=0.3, which="both")
    ax1.annotate(f"$r{{=}}3$: {modal_err[3]*100:.2f}%",
                 xy=(3, modal_err[3]), xytext=(3.4, modal_err[3] * 4),
                 arrowprops=dict(arrowstyle="->"), fontsize=10)

    labels = ["explicit\nRK4", "implicit\nNewmark"]
    steps = [rk4_steps, nm_steps]
    bars = ax2.bar(labels, steps, color=["tab:red", "tab:blue"], width=0.6)
    ax2.set_yscale("log")
    ax2.set_ylabel("time steps over one crossing")
    ax2.set_title("(b) cost on the damped 40 m bridge")
    for b, s, daf in zip(bars, steps, [rk4.daf, nm.daf]):
        ax2.text(b.get_x() + b.get_width() / 2, s * 1.3,
                 f"{s:,}\nDAF={daf:.3f}", ha="center", fontsize=9)
    ax2.text(0.5, 0.05, f"same DAF, {round(rk4_steps / nm_steps)}$\\times$ fewer steps",
             transform=ax2.transAxes, ha="center", fontsize=10,
             bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
    ax2.set_ylim(top=steps[0] * 6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "remedies.pdf"))
    fig.savefig(os.path.join(FIGDIR, "remedies.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(HERE, "data.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    print("\n=== headline numbers ===")
    print(json.dumps(data, indent=2))
    print("\nfigures + data.json written under note/")


if __name__ == "__main__":
    main()
