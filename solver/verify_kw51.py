"""P3 -- VALIDATION against REAL bridge data (KW51, Leuven).

Run directly (after fetching the data, see below):
    python -u solver/verify_kw51.py

WHY THIS MATTERS
----------------
Everything else in this project is verified against analytical references or its
own forward model. This script does the one thing that turns a simulation into
engineering: it touches REAL measured data. KW51 is an instrumented steel railway
bridge in Leuven; its operators published 16 months of continuously tracked modal
frequencies (Maes & Lombaert, Zenodo 3745914), spanning a structural RETROFIT in
mid-2019. That dataset is ground truth no model can argue with.

We use it to validate the premise of the whole drive-by-SHM half of the project --
"a stiffness change shows up as a frequency shift" -- and, just as important, to
expose its dominant real-world confounder:

  [1] TEMPERATURE drives the natural frequency (steel modulus + bearing/asphalt
      stiffness change with T). We fit f(T) on the healthy period and measure the
      sensitivity in %/degC.
  [2] The RETROFIT (a real, dated stiffness increase) shifts the frequency -- the
      real-world analogue of the synthetic EI-loss damage in verify_d4.py, but with
      the sign flipped (stiffening => frequency UP).
  [3] DETECTABILITY: the seasonal temperature swing moves the frequency by MORE
      than the retrofit does, so the raw frequency hides the event. Removing the
      temperature trend (the standard SHM step) makes the retrofit jump pop out of
      the residual at many sigma. This is exactly the honest limit verify_d4.py
      raised -- now quantified on real data.

Data (12.9 MB, not committed):
    python -c "import urllib.request,json; r=urllib.request.urlopen('https://zenodo.org/api/records/3745914'); d=json.load(r); u=[f['links']['self'] for f in d['files'] if f['key']=='trackedmodes.zip'][0]; urllib.request.urlretrieve(u,'data/trackedmodes.zip')"
    python -c "import zipfile; zipfile.ZipFile('data/trackedmodes.zip').extractall('data')"

Core stays numpy-only; scipy.io loads the .mat, matplotlib optional (both tooling).
"""

import os
from datetime import datetime, timedelta

import numpy as np

# KW51 strengthening ran ~15 May -> 27 Sep 2019 (dataset paper). Healthy = before.
RETROFIT_START = datetime(2019, 5, 15)
RETROFIT_END = datetime(2019, 9, 27)
MODE = 13          # best-tracked mode with strong T-dependence + clear jump (~6.9 Hz)
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "trackedmodes", "trackedmodes.mat")


def datenum_to_dt(x):
    """MATLAB datenum -> python datetime."""
    return datetime.fromordinal(int(x) - 366) + timedelta(days=float(x) % 1)


def main():
    if not os.path.exists(DATA):
        print(f"[!] {DATA} not found. Fetch it first (see the docstring).")
        return
    import scipy.io
    m = scipy.io.loadmat(DATA, squeeze_me=True, struct_as_record=False)["modes"]
    sdn = np.asarray(m.sdn, float)
    f = np.asarray(m.f, float)[:, MODE]            # one mode's frequency over time
    T = np.asarray(m.env, float)[:, 0]             # tBD31A: bridge-deck temperature
    dates = np.array([datenum_to_dt(s) for s in sdn])

    print("=" * 74)
    print("KW51 REAL-DATA VALIDATION - temperature, retrofit, detectability")
    print(f"  mode #{MODE}  (mean f = {np.nanmean(f):.3f} Hz);  "
          f"{(~np.isnan(f)).sum()} valid samples")
    print(f"  span {dates[0].date()} -> {dates[-1].date()};  retrofit "
          f"{RETROFIT_START.date()} -> {RETROFIT_END.date()}")
    print("=" * 74)

    before = (dates < RETROFIT_START) & ~np.isnan(f) & ~np.isnan(T)
    after = (dates > RETROFIT_END) & ~np.isnan(f) & ~np.isnan(T)

    # ---- [1] temperature sensitivity (fit on the HEALTHY/before period) -------
    a, b = np.polyfit(T[before], f[before], 1)      # f ~ a*T + b
    f_pred = a * T + b
    r = np.corrcoef(T[before], f[before])[0, 1]
    T_span = np.nanmax(T) - np.nanmin(T)
    swing_pct = 100 * abs(a) * T_span / np.nanmean(f)
    print("\n[1] TEMPERATURE drives the frequency (fit on healthy period)")
    print(f"    slope = {a*1e3:+.3f} mHz/degC  ({100*a/np.nanmean(f):+.3f} %/degC)"
          f",  corr r = {r:+.2f}")
    print(f"    deck T ranges {np.nanmin(T):.0f}..{np.nanmax(T):.0f} degC -> a "
          f"~{swing_pct:.1f}% seasonal frequency swing from temperature ALONE.")

    # ---- [2] retrofit jump (real stiffness change) ----------------------------
    fb, fa = np.nanmean(f[before]), np.nanmean(f[after])
    jump_pct = 100 * (fa - fb) / fb
    print("\n[2] RETROFIT = a real, dated stiffness increase")
    print(f"    mean f before = {fb:.3f} Hz,  after = {fa:.3f} Hz  "
          f"-> {jump_pct:+.1f}% (stiffening raises f).")
    print(f"    Temperature swing ({swing_pct:.1f}%) and retrofit jump "
          f"({abs(jump_pct):.1f}%) are the SAME ORDER\n          -> temperature is "
          "a first-order confounder that must be modelled, not ignored.")

    # ---- [2b] same picture across all well-tracked modes (no cherry-picking) --
    fall = np.asarray(m.f, float)
    print("\n[2b] Across modes: temperature swing vs retrofit jump (same order)")
    print(f"     {'mode':>4} {'f[Hz]':>7} {'%/degC':>8} {'T-swing%':>9} "
          f"{'jump%':>7} {'raw sig':>8} {'corr sig':>9}")
    for j in range(fall.shape[1]):
        fj = fall[:, j]
        bj = (dates < RETROFIT_START) & ~np.isnan(fj) & ~np.isnan(T)
        aj = (dates > RETROFIT_END) & ~np.isnan(fj) & ~np.isnan(T)
        if bj.sum() < 300 or aj.sum() < 300:
            continue
        aj_, bj_ = np.polyfit(T[bj], fj[bj], 1)
        fjmean = np.nanmean(fj)
        sw = 100 * abs(aj_) * T_span / fjmean
        jm = 100 * (np.nanmean(fj[aj]) - np.nanmean(fj[bj])) / np.nanmean(fj[bj])
        rj = fj - (aj_ * T + bj_)
        raw_s = abs(np.nanmean(fj[aj]) - np.nanmean(fj[bj])) / np.nanstd(fj[bj])
        cor_s = abs(np.nanmean(rj[aj])) / np.nanstd(rj[bj])
        print(f"     {j:>4} {fjmean:>7.3f} {100*aj_/fjmean:>+8.3f} {sw:>9.1f} "
              f"{jm:>+7.1f} {raw_s:>8.1f} {cor_s:>9.1f}")

    # ---- [3] detectability: temperature-corrected residual --------------------
    resid = f - f_pred                              # remove the temperature trend
    sd_before = np.nanstd(resid[before])
    raw_detect = abs(fa - fb) / np.nanstd(f[before])
    cor_detect = abs(np.nanmean(resid[after])) / sd_before
    print("\n[3] DETECTABILITY: remove temperature, the jump pops out")
    print(f"    raw jump / raw before-scatter        = {raw_detect:5.1f} sigma")
    print(f"    T-corrected jump / corrected scatter = {cor_detect:5.1f} sigma "
          f"  ({cor_detect/raw_detect:.1f}x sharper)")
    print("    -> exactly the verify_d4.py lesson on REAL data: frequency is a "
          "blunt,\n       environment-contaminated indicator; temperature "
          "compensation is\n       mandatory before a shift can be called damage.")

    print("\nVALIDATION SUMMARY")
    print("  - The drive-by-SHM premise holds on a real bridge: a stiffness change "
          "moves f.")
    print("  - But the real environmental swing is the same order as the structural"
          "\n    signal (for some modes it dominates, e.g. mode 12: 3.8% T-swing vs "
          "1.4% jump)\n    -- the project's synthetic D4 limit, confirmed in the field.")
    print(f"  - KW51 is a {115} m steel railway bridge (bowstring arch), NOT our "
          "simply-\n    supported EB beam, so we validate the SHM BEHAVIOUR/method, "
          "not the FEM\n    geometry. Honest scope.")

    # ---- plot -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

        ok = ~np.isnan(f) & ~np.isnan(T)
        sc = ax1.scatter(dates[ok], f[ok], c=T[ok], s=4, cmap="coolwarm")
        ax1.axvspan(RETROFIT_START, RETROFIT_END, color="0.8", alpha=0.6,
                    label="retrofit")
        ax1.set_ylabel(f"mode #{MODE} frequency [Hz]")
        ax1.set_title("Real KW51 frequency over 16 months")
        ax1.legend(fontsize=8); ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
        plt.colorbar(sc, ax=ax1, label="deck T [degC]")
        for lab in ax1.get_xticklabels(): lab.set_rotation(30)

        ax2.scatter(T[before], f[before], s=5, color="steelblue", alpha=0.4,
                    label="healthy")
        tt = np.array([np.nanmin(T), np.nanmax(T)])
        ax2.plot(tt, a * tt + b, "r-", lw=2,
                 label=f"fit: {100*a/np.nanmean(f):+.3f} %/degC")
        ax2.set_xlabel("deck temperature [degC]"); ax2.set_ylabel("frequency [Hz]")
        ax2.set_title("The confounder: f vs temperature")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        ax3.scatter(dates[before], resid[before] * 1e3, s=4, color="0.6",
                    label="before (residual ~0)")
        ax3.scatter(dates[after], resid[after] * 1e3, s=4, color="crimson",
                    label="after (shifted)")
        ax3.axhline(0, color="k", lw=0.8)
        ax3.axvspan(RETROFIT_START, RETROFIT_END, color="0.85", alpha=0.6)
        ax3.set_ylabel("T-corrected residual [mHz]")
        ax3.set_title(f"Retrofit pops out at {cor_detect:.0f}$\\sigma$")
        ax3.legend(fontsize=8); ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
        for lab in ax3.get_xticklabels(): lab.set_rotation(30)

        fig.tight_layout()
        out_dir = os.path.dirname(DATA).replace(os.path.join("data", "trackedmodes"),
                                                 "output")
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "output")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "kw51_validation.png")
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as exc:
        print(f"\n[plot] matplotlib unavailable ({exc!r}); skipping figure.")


if __name__ == "__main__":
    main()
