"""
19_ei_balance_and_criticality.py
=================================
TIER 1/2 — EXCITATION/INHIBITION PROXY + CRITICALITY

Two analyses that move "we can't claim excitability" toward "we report a
validated E/I proxy and a criticality trajectory", partially answering the
reviewers' core mechanistic objection without intracellular data.

PART A — E/I BALANCE via APERIODIC SLOPE (FOOOF-style)
  The aperiodic (1/f) exponent of the power spectrum is an established proxy
  for the E/I ratio: a flatter spectrum (smaller exponent) indexes a more
  excitation-dominated regime, a steeper spectrum a more inhibition-dominated
  one (Gao, Peterson & Voytek 2017). This part:
   (1) Separates periodic from aperiodic spectral components (specparam/FOOOF
       if installed; robust log-log linear fit otherwise).
   (2) Reports the aperiodic exponent as an explicit E/I proxy, state-specific,
       across the disease course — an E/I TRAJECTORY.
   (3) Decontaminates the band-power findings by also reporting the periodic
       (oscillatory) beta/theta power above the aperiodic component, so the
       headline effects are shown not to be 1/f-shift artifacts.

PART B — CRITICALITY / NEURONAL AVALANCHES
  Healthy cortex operates near a critical point; disease often pushes networks
  away from criticality. This part:
   (1) Detects avalanches from the band-limited LFP (threshold crossings of the
       analytic amplitude), computes avalanche size & duration distributions.
   (2) Estimates the branching parameter sigma (sub-critical <1, critical ~1,
       super-critical >1) and the power-law slope of avalanche sizes.
   (3) Tests whether C9orf72 networks drift from criticality over the course,
       and whether the 4m (hyper) and 12m (hypo) phases sit on OPPOSITE sides
       of the critical point — reframing the biphasic finding as bidirectional
       departure from criticality.

Outputs:
  results/ei_aperiodic_trajectory.csv
  results/periodic_decontaminated_power.csv
  results/criticality_metrics.csv
  figures/ei_trajectory.png
  figures/periodic_vs_aperiodic.png
  figures/criticality_trajectory.png
  figures/avalanche_distributions.png

INPUT: raw ABF (preferred) OR precomputed per-epoch PSDs. Run:
    python src/19_ei_balance_and_criticality.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt, hilbert, decimate
from scipy.stats import mannwhitneyu, linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys, os as _os2
_sys.path.insert(0, _os2.path.dirname(_os2.path.abspath(__file__)))
try:
    from abf_paths import iter_recordings, find_video_near
    HAVE_PATHS = True
except ImportError:
    HAVE_PATHS = False

warnings.filterwarnings("ignore")

try:
    import pyabf
    HAVE_ABF = True
except ImportError:
    HAVE_ABF = False

try:
    from fooof import FOOOF
    HAVE_FOOOF = True
except ImportError:
    HAVE_FOOOF = False

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")
ABF_DIR     = os.path.join(PORT_DIR, "data")

CA3_CH, CTX_CH = 0, 1
FS_TARGET = 500.0
EPOCH_S   = 4.0
TP_ORDER  = ["3m", "4m", "6m", "7m", "9m", "12m"]
COLORS    = {"WT": "#378ADD", "KO": "#D85A30"}
MAX_EPOCHS = 60

FIT_RANGE = (2, 45)   # Hz, aperiodic fit range (avoid line noise)
AVAL_BAND = (1, 45)   # broadband for avalanche detection
AVAL_THRESH_SD = 2.5  # threshold in SD for avalanche events


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2: return np.nan
    p = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return 0.0 if p == 0 else (np.mean(a)-np.mean(b))/p


def fdr_bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0: return p
    o = np.argsort(p); r = np.empty(n,int); r[o] = np.arange(1,n+1)
    f = p*n/r; f = np.minimum.accumulate(f[o][::-1])[::-1]
    out = np.empty(n); out[o] = f
    return np.minimum(out, 1.0)


def bandpass(x, lo, hi, fs=FS_TARGET, order=3):
    ny = 0.5*fs
    b, a = butter(order, [max(lo/ny,1e-4), min(hi/ny,0.999)], btype="band")
    return filtfilt(b, a, x)


def load_signal(abf_path, channel=0, target_fs=FS_TARGET):
    """Portfolio-consistent loader (matches script 14 load_signal)."""
    try:
        import gc
        abf = pyabf.ABF(abf_path)
        fs  = float(abf.dataRate)
        if channel >= abf.channelCount:
            del abf; gc.collect()
            return None, None
        abf.setSweep(0, channel=channel)
        sig = abf.sweepY.copy().astype(np.float64)
        del abf; gc.collect()
        if len(sig) < int(fs*30):
            return None, None
        factor = max(1, int(round(fs/target_fs)))
        if factor > 1:
            sig = decimate(sig - sig.mean(), factor, zero_phase=True)
        return sig, float(target_fs)
    except Exception:
        return None, None


def load_abf(abf_path):
    ca3, fs = load_signal(abf_path, channel=CA3_CH)
    if ca3 is None:
        return None, None
    ctx, _ = load_signal(abf_path, channel=CTX_CH)
    if ctx is None:
        return None, None
    return ca3, ctx
    try:
        abf = pyabf.ABF(abf_path)
        fs = abf.dataRate
        abf.setSweep(0, channel=CA3_CH); ca3 = abf.sweepY.copy()
        abf.setSweep(0, channel=CTX_CH); ctx = abf.sweepY.copy()
        f = int(round(fs/FS_TARGET))
        if f > 1:
            ca3 = decimate(ca3, f, ftype="iir", zero_phase=True)
            ctx = decimate(ctx, f, ftype="iir", zero_phase=True)
        return ca3, ctx
    except Exception:
        return None, None


def states_by_basename(tp):
    """{abf_basename: [state per epoch]} from the epoch CSV (states only; genotype from folder)."""
    p = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
    if not os.path.exists(p):
        return {}
    ep = pd.read_csv(p)
    out = {}
    for abf_file, g in ep.groupby("abf_file"):
        out[os.path.basename(str(abf_file))] = g.sort_values("epoch_idx")["state"].tolist()
    return out


# ── PART A — E/I via aperiodic slope ───────────────────────────────────────
def aperiodic_exponent(freqs, psd):
    """Return (exponent, periodic_peaks) via FOOOF if available, else log-log fit."""
    mask = (freqs >= FIT_RANGE[0]) & (freqs <= FIT_RANGE[1])
    f, p = freqs[mask], psd[mask]
    if len(f) < 5:
        return np.nan, {}
    if HAVE_FOOOF:
        try:
            fm = FOOOF(max_n_peaks=6, verbose=False)
            fm.fit(f, p, FIT_RANGE)
            exp = fm.get_params("aperiodic_params", "exponent")
            # periodic power in theta/beta from gaussian peaks
            peaks = {}
            for pk in fm.get_params("peak_params"):
                cf = pk[0]
                if 4 <= cf < 8: peaks["theta_pk"] = pk[1]
                elif 13 <= cf < 30: peaks["beta_pk"] = pk[1]
            return exp, peaks
        except Exception:
            pass
    # robust log-log linear fit
    lr = linregress(np.log10(f), np.log10(p + 1e-30))
    return -lr.slope, {}


def part_a_ei_balance(states=("REM", "NREM", "Wake")):
    print("\n" + "="*64)
    print("PART A — E/I BALANCE via APERIODIC SLOPE")
    print("(flatter spectrum = more excitation-dominated)")
    print("="*64)
    if not HAVE_ABF:
        print("pyabf required.")
        return None
    if not HAVE_FOOOF:
        print("NOTE: fooof/specparam not installed — using robust log-log fit.")
        print("      pip install fooof --break-system-packages for periodic/aperiodic split.")

    rows = []
    for tp in TP_ORDER:
        state_lut = states_by_basename(tp)
        recs = list(iter_recordings(tp)) if HAVE_PATHS else []
        if not recs:
            print(f"[{tp}] no recordings found on disk"); continue
        print(f"[{tp}] {len(recs)} recordings on disk")
        for ridx, (path, mouse_id, group) in enumerate(recs, 1):
            if ridx % 20 == 0:
                print(f"    [{tp}] {ridx}/{len(recs)} recordings...")
            seq = state_lut.get(os.path.basename(path))
            if seq is None: continue
            ca3, ctx = load_abf(path)
            if ctx is None: continue
            aid = mouse_id
            ep_len = int(EPOCH_S*FS_TARGET)
            for state in states:
                idx = [i for i, s in enumerate(seq) if s == state][:MAX_EPOCHS]
                if len(idx) < 5: continue
                # Average the per-epoch PSDs FIRST (shared freq grid), then one FOOOF fit.
                psd_ctx, psd_ca3, freqs = [], [], None
                for ei in idx:
                    a, b = ei*ep_len, (ei+1)*ep_len
                    if b > len(ctx): continue
                    fC, pC = welch(ctx[a:b], FS_TARGET, nperseg=min(1024, b-a))
                    fH, pH = welch(ca3[a:b], FS_TARGET, nperseg=min(1024, b-a))
                    psd_ctx.append(pC); psd_ca3.append(pH); freqs = fC
                if len(psd_ctx) < 5 or freqs is None:
                    continue
                mean_ctx = np.mean(np.vstack(psd_ctx), axis=0)
                mean_ca3 = np.mean(np.vstack(psd_ca3), axis=0)
                eC, pkC = aperiodic_exponent(freqs, mean_ctx)   # ONE fit per recording-state
                eH, _   = aperiodic_exponent(freqs, mean_ca3)
                if np.isnan(eC):
                    continue
                rows.append({
                    "timepoint": tp, "animal_id": aid, "group": group, "state": state,
                    "ctx_aperiodic_exp": eC,
                    "ca3_aperiodic_exp": eH if not np.isnan(eH) else np.nan,
                    "ctx_beta_periodic": pkC.get("beta_pk", np.nan),
                    "ctx_theta_periodic": pkC.get("theta_pk", np.nan),
                    "n_epochs": len(psd_ctx),
                })
    if not rows:
        print("No spectra computed.")
        return None
    ei = pd.DataFrame(rows)
    ei.to_csv(os.path.join(RESULTS_DIR, "ei_aperiodic_trajectory.csv"), index=False)
    print(f"Saved: ei_aperiodic_trajectory.csv ({len(ei)} rows)")

    # Decontaminated periodic power table
    ei[["timepoint","animal_id","group","state","ctx_beta_periodic","ctx_theta_periodic"]]\
        .to_csv(os.path.join(RESULTS_DIR, "periodic_decontaminated_power.csv"), index=False)

    print("\nCortical E/I proxy (aperiodic exp): WT vs KO  [lower=more excitation]")
    for state in states:
        for tp in TP_ORDER:
            sub = ei[(ei.state == state) & (ei.timepoint == tp)]
            wt = sub[sub.group=="WT"]["ctx_aperiodic_exp"].dropna().values
            ko = sub[sub.group=="KO"]["ctx_aperiodic_exp"].dropna().values
            if len(wt) < 2 or len(ko) < 2: continue
            d = cohens_d(ko, wt); _, p = mannwhitneyu(ko, wt, alternative="two-sided")
            if p < 0.10:
                print(f"  [{state} {tp}] WT={np.mean(wt):.3f} KO={np.mean(ko):.3f} "
                      f"d={d:.2f} p={p:.3f}")

    plot_ei_trajectory(ei)
    return ei


def plot_ei_trajectory(ei):
    states = [s for s in ["REM","NREM","Wake"] if s in ei.state.unique()]
    fig, axes = plt.subplots(1, len(states), figsize=(5*len(states), 4.3), squeeze=False)
    for ci, state in enumerate(states):
        ax = axes[0][ci]
        sub = ei[ei.state == state]
        for group, color in COLORS.items():
            g = sub[sub.group == group]
            xs, ms, es = [], [], []
            for tp in TP_ORDER:
                v = g[g.timepoint == tp]["ctx_aperiodic_exp"].dropna()
                if len(v)==0: continue
                xs.append(TP_ORDER.index(tp)); ms.append(v.mean()); es.append(v.sem())
            if xs:
                ax.errorbar(xs, ms, yerr=es, fmt="-o", color=color, lw=2, capsize=3, label=group)
        ax.set_xticks(range(len(TP_ORDER))); ax.set_xticklabels(TP_ORDER)
        ax.set_ylabel("Aperiodic exponent (E/I proxy)")
        ax.set_xlabel("Timepoint")
        ax.set_title(f"Cortical E/I trajectory — {state}\n(↓ = more excitation-dominated)", fontsize=9)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ei_trajectory.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: ei_trajectory.png")


# ── PART B — Criticality / avalanches ──────────────────────────────────────
def detect_avalanches(sig, fs=FS_TARGET, thresh_sd=AVAL_THRESH_SD):
    """
    Avalanche = continuous supra-threshold excursion of the analytic amplitude.
    Returns (sizes, durations, branching_sigma).
    """
    amp = np.abs(hilbert(bandpass(sig, *AVAL_BAND, fs)))
    thr = np.mean(amp) + thresh_sd*np.std(amp)
    over = amp > thr
    sizes, durs = [], []
    i = 0
    n = len(over)
    while i < n:
        if over[i]:
            j = i
            s = 0.0
            while j < n and over[j]:
                s += amp[j] - thr
                j += 1
            sizes.append(s)
            durs.append((j - i)/fs*1000.0)   # ms
            i = j
        else:
            i += 1
    # Branching parameter: ratio of descendants to ancestors (bin-based)
    bin_w = int(0.004*fs)  # 4ms bins
    if bin_w < 1: bin_w = 1
    binned = [over[k:k+bin_w].sum() for k in range(0, n, bin_w)]
    binned = np.array(binned, float)
    sigma = np.nan
    if len(binned) > 2:
        anc = binned[:-1]
        des = binned[1:]
        nz = anc > 0
        if nz.sum() > 5:
            sigma = np.mean(des[nz]/anc[nz])
    return np.array(sizes), np.array(durs), sigma


def powerlaw_slope(sizes):
    """Slope of log-log avalanche-size distribution (more negative = steeper)."""
    sizes = sizes[sizes > 0]
    if len(sizes) < 20:
        return np.nan
    hist, edges = np.histogram(np.log10(sizes), bins=20)
    centers = 0.5*(edges[:-1]+edges[1:])
    nz = hist > 0
    if nz.sum() < 5:
        return np.nan
    lr = linregress(centers[nz], np.log10(hist[nz]))
    return lr.slope


def part_b_criticality(states=("REM", "NREM")):
    print("\n" + "="*64)
    print("PART B — CRITICALITY / NEURONAL AVALANCHES")
    print("Branching sigma: <1 sub-critical, ~1 critical, >1 super-critical")
    print("="*64)
    if not HAVE_ABF:
        print("pyabf required.")
        return None

    rows = []
    for tp in TP_ORDER:
        state_lut = states_by_basename(tp)
        recs = list(iter_recordings(tp)) if HAVE_PATHS else []
        if not recs:
            print(f"[{tp}] no recordings found on disk"); continue
        print(f"[{tp}] {len(recs)} recordings on disk")
        for ridx, (path, mouse_id, group) in enumerate(recs, 1):
            if ridx % 20 == 0:
                print(f"    [{tp}] {ridx}/{len(recs)} recordings...")
            seq = state_lut.get(os.path.basename(path))
            if seq is None: continue
            ca3, ctx = load_abf(path)
            if ctx is None: continue
            aid = mouse_id
            ep_len = int(EPOCH_S*FS_TARGET)
            for state in states:
                idx = [i for i, s in enumerate(seq) if s == state][:MAX_EPOCHS]
                if len(idx) < 8: continue
                seg = np.concatenate([ctx[i*ep_len:(i+1)*ep_len] for i in idx
                                      if (i+1)*ep_len <= len(ctx)])
                if len(seg) < FS_TARGET*8: continue
                sizes, durs, sigma = detect_avalanches(seg)
                slope = powerlaw_slope(sizes)
                rows.append({
                    "timepoint": tp, "animal_id": aid, "group": group, "state": state,
                    "branching_sigma": sigma,
                    "avalanche_slope": slope,
                    "n_avalanches": len(sizes),
                    "mean_size": np.mean(sizes) if len(sizes) else np.nan,
                    "distance_from_critical": abs(sigma - 1.0) if not np.isnan(sigma) else np.nan,
                })
    if not rows:
        print("No avalanche metrics computed.")
        return None
    cr = pd.DataFrame(rows)
    cr.to_csv(os.path.join(RESULTS_DIR, "criticality_metrics.csv"), index=False)
    print(f"Saved: criticality_metrics.csv ({len(cr)} rows)")

    print("\nBranching sigma (criticality): WT vs KO")
    for state in states:
        for tp in TP_ORDER:
            sub = cr[(cr.state == state) & (cr.timepoint == tp)]
            wt = sub[sub.group=="WT"]["branching_sigma"].dropna().values
            ko = sub[sub.group=="KO"]["branching_sigma"].dropna().values
            if len(wt) < 2 or len(ko) < 2: continue
            d = cohens_d(ko, wt); _, p = mannwhitneyu(ko, wt, alternative="two-sided")
            flag = "  *" if p < 0.05 else ""
            print(f"  [{state} {tp}] sigma WT={np.mean(wt):.3f} KO={np.mean(ko):.3f} "
                  f"d={d:.2f} p={p:.3f}{flag}")

    plot_criticality(cr)
    return cr


def plot_criticality(cr):
    states = [s for s in ["REM","NREM"] if s in cr.state.unique()]
    fig, axes = plt.subplots(1, len(states), figsize=(6*len(states), 4.3), squeeze=False)
    for ci, state in enumerate(states):
        ax = axes[0][ci]
        sub = cr[cr.state == state]
        for group, color in COLORS.items():
            g = sub[sub.group == group]
            xs, ms, es = [], [], []
            for tp in TP_ORDER:
                v = g[g.timepoint == tp]["branching_sigma"].dropna()
                if len(v)==0: continue
                xs.append(TP_ORDER.index(tp)); ms.append(v.mean()); es.append(v.sem())
            if xs:
                ax.errorbar(xs, ms, yerr=es, fmt="-o", color=color, lw=2, capsize=3, label=group)
        ax.axhline(1.0, color="k", ls="--", lw=1, label="critical (σ=1)")
        ax.set_xticks(range(len(TP_ORDER))); ax.set_xticklabels(TP_ORDER)
        ax.set_ylabel("Branching parameter σ")
        ax.set_xlabel("Timepoint")
        ax.set_title(f"Criticality trajectory — {state}\n(departure from σ=1 in either direction)", fontsize=9)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "criticality_trajectory.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: criticality_trajectory.png")


def main():
    print("="*64)
    print("E/I BALANCE + CRITICALITY")
    print("="*64)
    if not HAVE_ABF:
        print("\npyabf not installed: pip install pyabf --break-system-packages")
        return
    part_a_ei_balance()
    part_b_criticality()
    print("\n" + "="*64)
    print("COMPLETE")
    print("="*64)


if __name__ == "__main__":
    main()
