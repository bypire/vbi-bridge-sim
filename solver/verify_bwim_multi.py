"""V4.5+ — MULTI-AXLE B-WIM: separate and weigh each axle of a truck.

Run directly:
    python solver/verify_bwim_multi.py

The single-axle capstone (verify_bwim.py) recovered one weight. Real B-WIM must
also SEPARATE the axles of a multi-axle truck from one moment signal. Moses'
algorithm extends naturally: each axle contributes its own (shifted) influence
line, so the influence matrix C gets one column per axle and least squares
returns all the axle loads at once.

The hard part is conditioning: axles close together produce nearly parallel
columns in C, so noise gets amplified when splitting them — a real, well-known
B-WIM limitation. We show it with the condition number of C.

Checks:
  [1] Noise-free recovery of a 3-axle truck (steer + tandem drive).
  [2] Noise robustness (Monte-Carlo) — per-axle bias and scatter.
  [3] Separability vs tandem spacing — cond(C) and error grow as axles close in.

Core stays numpy-only; matplotlib optional.
"""

import os

import numpy as np

from beam_fem import Beam
from moving_load import integrate_moving_axles
from bwim import add_measurement_noise, moment_influence_line, moses_recover

# --- same bridge as before --------------------------------------------------
L, E, I, M_BAR = 20.0, 2.1e11, 0.02, 2000.0
X_GAUGE = L / 2.0
SPEED = 20.0
SEED = 2024

# --- a 3-axle truck: steer axle + tandem drive axles ------------------------
AXLE_LOADS = np.array([60.0e3, 95.0e3, 90.0e3])   # N  (≈ 24.9 t total)
AXLE_OFFSETS = np.array([0.0, 4.20, 5.50])        # m behind the steer axle
AXLE_NAMES = ["steer", "drive-1", "drive-2"]


def recover(beam, loads, offsets, noise_frac=0.0, rng=None):
    res = integrate_moving_axles(beam, loads, offsets, SPEED, moment_section=X_GAUGE)
    moment = res.moment
    if noise_frac > 0.0:
        moment = add_measurement_noise(moment, noise_frac, rng)
    front = SPEED * res.t
    P, fitted = moses_recover(beam, X_GAUGE, front, moment, axle_offsets=offsets)
    # condition number of the influence matrix (separability indicator)
    C = np.column_stack([moment_influence_line(beam, X_GAUGE, front - o)
                         for o in offsets])
    cond = float(np.linalg.cond(C))
    return res, front, moment, fitted, P, cond


def main():
    beam = Beam(L=L, E=E, I=I, mass_per_length=M_BAR, n_elements=20)
    total = AXLE_LOADS.sum()

    print("=" * 70)
    print("V4.5+ VERIFICATION - MULTI-AXLE B-WIM (Moses, one column per axle)")
    print(f"  3-axle truck, total gross = {total/1e3:.1f} kN "
          f"({total/9.81/1e3:.2f} t)")
    for n, P, o in zip(AXLE_NAMES, AXLE_LOADS, AXLE_OFFSETS):
        print(f"    {n:>8}: {P/1e3:6.1f} kN  at offset {o:.2f} m")
    print("=" * 70)

    # --- [1] noise-free ----------------------------------------------------
    res, front, moment, fitted, P, cond = recover(beam, AXLE_LOADS, AXLE_OFFSETS)
    print(f"\n[1] NOISE-FREE recovery at v={SPEED} m/s; cond(C) = {cond:.1f}")
    print(f"    {'axle':>8} {'true[kN]':>9} {'recovered[kN]':>14} {'error[%]':>9}")
    for n, Pt, Pr in zip(AXLE_NAMES, AXLE_LOADS, P):
        print(f"    {n:>8} {Pt/1e3:>9.1f} {Pr/1e3:>14.1f} "
              f"{100*(Pr-Pt)/Pt:>9.2f}")
    print(f"    {'GROSS':>8} {total/1e3:>9.1f} {P.sum()/1e3:>14.1f} "
          f"{100*(P.sum()-total)/total:>9.2f}")
    print("    (Gross weight is recovered far better than the individual axle\n"
          "     split — exactly why B-WIM gross-weight enforcement is robust.)")

    # --- [2] noise robustness ---------------------------------------------
    rng = np.random.default_rng(SEED)
    res2 = integrate_moving_axles(beam, AXLE_LOADS, AXLE_OFFSETS, SPEED,
                                  moment_section=X_GAUGE)
    front2 = SPEED * res2.t
    N_MC = 300
    rec = np.empty((N_MC, len(AXLE_LOADS)))
    for k in range(N_MC):
        noisy = add_measurement_noise(res2.moment, 0.03, rng)
        Pk, _ = moses_recover(beam, X_GAUGE, front2, noisy, axle_offsets=AXLE_OFFSETS)
        rec[k] = Pk
    print(f"\n[2] NOISE ROBUSTNESS: 3% peak noise, {N_MC} passes")
    print(f"    {'axle':>8} {'mean[kN]':>9} {'bias[%]':>8} {'scatter[%]':>11}")
    for j, n in enumerate(AXLE_NAMES):
        mean = rec[:, j].mean()
        bias = 100 * (mean - AXLE_LOADS[j]) / AXLE_LOADS[j]
        scat = 100 * rec[:, j].std() / AXLE_LOADS[j]
        print(f"    {n:>8} {mean/1e3:>9.1f} {bias:>8.2f} {scat:>11.2f}")
    gross_scat = 100 * rec.sum(axis=1).std() / total
    print(f"    gross-weight scatter = {gross_scat:.2f} %  (axle splits scatter\n"
          "     much more than the gross — the conditioning penalty).")

    # --- [3] separability vs tandem spacing -------------------------------
    print("\n[3] SEPARABILITY vs TANDEM SPACING (drive-2 moved toward drive-1)")
    print(f"    {'spacing[m]':>10} {'cond(C)':>9} {'drive-1 err[%]':>15} "
          f"{'drive-2 err[%]':>15}")
    spac = []
    for d in [3.0, 2.0, 1.3, 0.8, 0.5]:
        offs = np.array([0.0, 4.20, 4.20 + d])
        _, _, _, _, Pd, cd = recover(beam, AXLE_LOADS, offs)
        e1 = 100 * (Pd[1] - AXLE_LOADS[1]) / AXLE_LOADS[1]
        e2 = 100 * (Pd[2] - AXLE_LOADS[2]) / AXLE_LOADS[2]
        spac.append((d, cd, e1, e2))
        print(f"    {d:>10.2f} {cd:>9.1f} {e1:>15.2f} {e2:>15.2f}")
    print("    (Closer axles -> larger cond(C) -> the split is more fragile.)")

    # --- visualization -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng_p = np.random.default_rng(SEED)
        noisy = add_measurement_noise(res.moment, 0.03, rng_p)
        Pn, fit_n = moses_recover(beam, X_GAUGE, front, noisy,
                                  axle_offsets=AXLE_OFFSETS)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

        ax1.plot(res.t, noisy / 1e3, color="0.6", lw=0.7, label="measured (+3%)")
        ax1.plot(res.t, fit_n / 1e3, "r-", lw=1.8, label="Moses total fit")
        for j, n in enumerate(AXLE_NAMES):
            contrib = Pn[j] * moment_influence_line(beam, X_GAUGE,
                                                    front - AXLE_OFFSETS[j])
            ax1.plot(res.t, contrib / 1e3, "--", lw=1.0,
                     label=f"{n} → {Pn[j]/1e3:.0f} kN")
        ax1.set_xlabel("time t [s]"); ax1.set_ylabel("moment at gauge [kN·m]")
        ax1.set_title("Measured moment = sum of per-axle influence lines")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        x = np.arange(len(AXLE_NAMES))
        ax2.bar(x - 0.2, AXLE_LOADS / 1e3, 0.4, label="true", color="steelblue")
        ax2.bar(x + 0.2, P / 1e3, 0.4, label="recovered", color="indianred")
        ax2.set_xticks(x); ax2.set_xticklabels(AXLE_NAMES)
        ax2.set_ylabel("axle load [kN]")
        ax2.set_title("True vs recovered (noise-free)")
        ax2.legend(); ax2.grid(alpha=0.3, axis="y")

        ds = [s[0] for s in spac]
        ax3.plot(ds, [s[1] for s in spac], "m-o")
        ax3.set_xlabel("tandem spacing [m]"); ax3.set_ylabel("cond(C)")
        ax3.set_title("Axle separability\n(higher cond = harder to split)")
        ax3.grid(alpha=0.3); ax3.invert_xaxis()

        fig.tight_layout()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(root, "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "v4_5_bwim_multiaxle.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
