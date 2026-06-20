"""
14_scekic_zahirovic_pac_replication.py
========================================
State-specific Phase-Amplitude Coupling analysis replicating and extending
Scekic-Zahirovic et al., Science Translational Medicine 2024.

KEY METHODOLOGICAL DETAILS FROM PAPER:
  - Phase band:      4-8 Hz (theta)
  - Amplitude bands: ~60 Hz  (theta-LOW gamma,  T-LG)
                     ~120 Hz (theta-HIGH gamma, T-HG) ← primary finding
  - States:          REM sleep (primary in mice) + active Wake
  - Method:          Tort et al. modulation index (same as current implementation)
  - Their finding:   T-HG PAC DECREASED in ALS models during REM sleep,
                     from PRESYMPTOMATIC stage onward
  - Channel:         Motor CORTEX (ECoG/EEG over sensorimotor areas)
                     → your CTX channel (channel 1) is the replication channel
                     → your CA3 channel (channel 0) is hippocampal (novel extension)

YOUR ADVANTAGE OVER THEIR PAPER:
  - 6 timepoints (3m–12m) vs their cross-sectional/4-timepoint design
  - Sleep state labels already computed (script 05)
  - Both cortex AND hippocampus (circuit dissociation)
  - C9orf72-KO = third genetic model (they had SOD1, FUS)
  - KA challenge at 4m = unique acute phase not in their paper

BANDS TESTED:
  T-LG: theta (4-8 Hz) phase × low gamma (30-80 Hz) amplitude
  T-HG: theta (4-8 Hz) phase × high gamma (80-150 Hz) amplitude  ← PRIMARY
  A-LG: alpha (8-13 Hz) phase × low gamma (30-80 Hz) amplitude
  A-HG: alpha (8-13 Hz) phase × high gamma (80-150 Hz) amplitude

STATES:
  REM   ← primary (their main finding)
  Wake  ← secondary (they found it in SOD1 during wake too)
  NREM  ← exploratory (not in their paper)

Outputs:
  results/pac_state_specific_{tp}.csv
  results/pac_state_specific_stats.csv
  figures/pac_rem_trajectory.png
  figures/pac_wake_trajectory.png
  figures/pac_scekic_replication_summary.png

Run:
    python src/14_scekic_zahirovic_pac_replication.py
    python src/14_scekic_zahirovic_pac_replication.py --timepoint 3m
"""

import os
import gc
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import pyabf
from scipy.signal import decimate, butter, hilbert
from scipy.signal import sosfiltfilt
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")

COLORS      = {"WT": "#378ADD", "KO": "#D85A30"}
STATE_COLORS = {"REM": "#9B59B6", "Wake": "#E67E22", "NREM": "#3498DB"}
TP_ORDER    = ["3m", "4m", "6m", "7m", "9m", "12m"]
TP_X        = [3, 4, 6, 7, 9, 12]
FS_DS       = 500
EPOCH_S     = 4.0
MIN_EPOCHS  = 20   # minimum epochs per state to compute PAC

# ── PAC bands — matching Scekic-Zahirovic exactly ─────────────────────────
PAC_BANDS = {
    "T_LG": {"phase": (4,   8),  "amp": (30,  80),  "label": "Theta-Low Gamma"},
    "T_HG": {"phase": (4,   8),  "amp": (80, 150),  "label": "Theta-High Gamma"},  # PRIMARY
    "A_LG": {"phase": (8,  13),  "amp": (30,  80),  "label": "Alpha-Low Gamma"},
    "A_HG": {"phase": (8,  13),  "amp": (80, 150),  "label": "Alpha-High Gamma"},
}

STATES_TO_ANALYZE = ["REM", "Wake", "NREM"]


# ── Signal helpers ─────────────────────────────────────────────────────────

def fdr_bh(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=int)
    ranked[order] = np.arange(1, n+1)
    fdr = pvals * n / ranked
    fdr_adj = np.minimum.accumulate(fdr[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = fdr_adj
    return np.minimum(result, 1.0)


def cohens_d_ci(x, y, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    d = (np.mean(y)-np.mean(x))/np.sqrt((np.std(x)**2+np.std(y)**2)/2+1e-10)
    boot = []
    for _ in range(n_boot):
        bx = rng.choice(x, len(x), replace=True)
        by = rng.choice(y, len(y), replace=True)
        boot.append((np.mean(by)-np.mean(bx))/
                    np.sqrt((np.std(bx)**2+np.std(by)**2)/2+1e-10))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def load_signal(abf_path, channel=0, target_fs=FS_DS):
    try:
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
            sig = decimate(sig-sig.mean(), factor, zero_phase=True)
        return sig, float(target_fs)
    except Exception:
        return None, None


def bandpass(sig, fs, lo, hi, order=4):
    # For high gamma (80-150 Hz) at 500 Hz sampling, need to be careful
    # Nyquist = 250 Hz, so 150 Hz is safely below
    hi_safe = min(hi, fs*0.45)   # stay below Nyquist
    if lo >= hi_safe:
        return None
    sos = butter(order, [lo, hi_safe], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def compute_pac_mi(sig, fs, phase_band, amp_band, n_bins=18):
    """
    Compute Modulation Index (MI) following Tort et al. 2010.
    Exactly matches Scekic-Zahirovic methodology.
    Returns MI value.
    """
    ph_sig = bandpass(sig, fs, phase_band[0], phase_band[1])
    am_sig = bandpass(sig, fs, amp_band[0],   amp_band[1])

    if ph_sig is None or am_sig is None:
        return np.nan

    phase = np.angle(hilbert(ph_sig))
    amp   = np.abs(hilbert(am_sig))

    bin_edges = np.linspace(-np.pi, np.pi, n_bins+1)
    amp_dist  = np.zeros(n_bins)
    for k in range(n_bins):
        mask = (phase >= bin_edges[k]) & (phase < bin_edges[k+1])
        if mask.sum() > 0:
            amp_dist[k] = amp[mask].mean()

    amp_dist = np.clip(amp_dist, 1e-12, None)
    amp_dist /= amp_dist.sum()
    MI = np.sum(amp_dist * np.log(amp_dist / (1/n_bins) + 1e-12)) / np.log(n_bins)
    return float(MI)


def compute_pac_per_epoch(sig, fs, phase_band, amp_band):
    """
    Compute PAC MI per epoch, return median.
    More robust than single-segment PAC.
    """
    epoch_n  = int(EPOCH_S * fs)
    n_epochs = len(sig) // epoch_n
    mi_vals  = []

    for i in range(n_epochs):
        ep = sig[i*epoch_n:(i+1)*epoch_n]
        if np.abs(ep).max() > 500 or np.std(ep) < 0.001:
            continue
        mi = compute_pac_mi(ep, fs, phase_band, amp_band)
        if not np.isnan(mi):
            mi_vals.append(mi)

    return float(np.median(mi_vals)) if len(mi_vals) >= 5 else np.nan


# ── Load state-specific signal segments ───────────────────────────────────

def get_state_epochs(tp, animal_id, abf_file):
    """Load epoch indices per state from script 05 output."""
    path = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
    if not os.path.exists(path):
        return {}
    ep = pd.read_csv(path)
    ep["animal_id"] = ep["animal_id"].astype(str)
    mask = (ep["animal_id"] == str(animal_id)) & (ep["abf_file"] == abf_file)
    sub  = ep[mask][["epoch_idx","state"]].copy()
    if len(sub) == 0:
        return {}
    state_dict = {}
    for state in STATES_TO_ANALYZE:
        idxs = sub[sub.state == state]["epoch_idx"].tolist()
        state_dict[state] = idxs
    return state_dict


def extract_state_signal(sig, fs, epoch_indices):
    """Extract and concatenate signal segments for given epoch indices."""
    epoch_n = int(EPOCH_S * fs)
    segments = []
    for idx in epoch_indices:
        start = idx * epoch_n
        end   = start + epoch_n
        if end > len(sig):
            continue
        ep = sig[start:end]
        if np.abs(ep).max() < 500 and np.std(ep) > 0.001:
            segments.append(ep)
    if not segments:
        return None
    return np.concatenate(segments)


# ── Core computation ───────────────────────────────────────────────────────

def compute_file_pac(abf_path, animal_id, group, tp, abf_file):
    """
    Compute state-specific PAC for all bands, both channels.
    Returns dict of results.
    """
    result = {
        "animal_id": animal_id,
        "group":     group,
        "timepoint": tp,
        "abf_file":  abf_file,
    }

    # Load state epoch indices
    state_epochs = get_state_epochs(tp, animal_id, abf_file)
    if not state_epochs:
        return None

    # Load both channels
    sig_ca3, fs = load_signal(abf_path, channel=0)
    sig_ctx, _  = load_signal(abf_path, channel=1)

    if sig_ca3 is None:
        return None

    for state in STATES_TO_ANALYZE:
        idxs = state_epochs.get(state, [])
        if len(idxs) < MIN_EPOCHS:
            continue

        result[f"{state.lower()}_n_epochs"] = len(idxs)

        # CA3 PAC
        seg_ca3 = extract_state_signal(sig_ca3, fs, idxs)
        if seg_ca3 is not None and len(seg_ca3) > int(fs * 10):
            for band_key, band_info in PAC_BANDS.items():
                mi = compute_pac_per_epoch(seg_ca3, fs,
                                            band_info["phase"], band_info["amp"])
                result[f"ca3_{state.lower()}_{band_key}"] = mi

        # CTX PAC (Scekic-Zahirovic replication channel)
        if sig_ctx is not None:
            seg_ctx = extract_state_signal(sig_ctx, fs, idxs)
            if seg_ctx is not None and len(seg_ctx) > int(fs * 10):
                for band_key, band_info in PAC_BANDS.items():
                    mi = compute_pac_mi(seg_ctx, fs,
                                         band_info["phase"], band_info["amp"])
                    result[f"ctx_{state.lower()}_{band_key}"] = mi

    del sig_ca3
    if sig_ctx is not None:
        del sig_ctx
    gc.collect()

    return result


# ── Process one timepoint ─────────────────────────────────────────────────

def process_timepoint(tp, inventory):
    print(f"\n{'='*60}")
    print(f"State-specific PAC — {tp}")
    print(f"{'='*60}")

    inv = inventory[(inventory.timepoint==tp)].drop_duplicates(
                        subset=["abf_path"]).copy()
    if len(inv) == 0:
        print(f"  No files for {tp}")
        return pd.DataFrame()

    rows = []
    for _, row in inv.iterrows():
        abf_path  = str(row["abf_path"])
        animal_id = str(row["animal_id"])
        group     = row["group"]
        abf_file  = row["abf_file"]

        if not os.path.exists(abf_path):
            continue

        print(f"  {animal_id} [{group}] | {abf_file[:20]}", end="", flush=True)
        res = compute_file_pac(abf_path, animal_id, group, tp, abf_file)

        if res:
            rows.append(res)
            # Print key values
            rem_thg_ca3 = res.get("ca3_rem_T_HG", np.nan)
            rem_thg_ctx = res.get("ctx_rem_T_HG", np.nan)
            print(f" CA3_REM_THG={rem_thg_ca3:.5f}"
                  f" CTX_REM_THG={rem_thg_ctx:.5f}"
                  if not np.isnan(rem_thg_ca3) else " (no REM epochs)")
        else:
            print(" SKIP")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, f"pac_state_specific_{tp}.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved: pac_state_specific_{tp}.csv ({len(df)} rows)")
    return df


# ── Statistics ────────────────────────────────────────────────────────────

def run_statistics(all_dfs):
    """
    Primary test: T-HG PAC during REM — Scekic-Zahirovic replication.
    Secondary: all other bands × states × channels.
    """
    print(f"\n{'='*60}")
    print("STATE-SPECIFIC PAC STATISTICS")
    print("Primary: T-HG PAC during REM (Scekic-Zahirovic replication)")
    print(f"{'='*60}")

    # Build feature list
    feats = []
    for ch in ["ca3", "ctx"]:
        for state in ["rem", "wake", "nrem"]:
            for band in PAC_BANDS:
                feats.append(f"{ch}_{state}_{band}")

    rows = []
    # Primary first
    primary_feat = "ctx_rem_T_HG"
    print(f"\n  PRIMARY: {primary_feat} (Scekic-Zahirovic CTX T-HG during REM)")
    for tp in TP_ORDER:
        if tp not in all_dfs or all_dfs[tp].empty:
            continue
        df = all_dfs[tp]
        if primary_feat not in df.columns:
            continue
        am = df.groupby(["animal_id","group"])[primary_feat].mean().reset_index()
        wt = am[am.group=="WT"][primary_feat].dropna().values
        ko = am[am.group=="KO"][primary_feat].dropna().values
        if len(wt)<2 or len(ko)<2:
            continue
        _, p = mannwhitneyu(wt, ko, alternative="two-sided")
        d, d_lo, d_hi = cohens_d_ci(wt, ko)
        ci = (d_lo>0) or (d_hi<0)
        print(f"  [{tp}] WT={np.mean(wt):.5f} KO={np.mean(ko):.5f} "
              f"d={d:.3f} [{d_lo:.3f},{d_hi:.3f}] p={p:.5f}"
              f"  n={len(wt)}/{len(ko)}"
              f"{'  CI✓' if ci else ''}")
        rows.append({
            "feature": primary_feat, "timepoint": tp,
            "channel": "CTX", "state": "REM", "band": "T_HG",
            "is_primary": True,
            "wt_mean": np.mean(wt), "ko_mean": np.mean(ko),
            "cohens_d": d, "d_ci_lo": d_lo, "d_ci_hi": d_hi,
            "ci_excludes_zero": ci, "pval": p,
            "n_wt": len(wt), "n_ko": len(ko),
        })

    # All other features
    print(f"\n  ALL FEATURES (p<0.10 shown):")
    for feat in feats:
        if feat == primary_feat:
            continue
        parts = feat.split("_")
        ch    = parts[0]
        state = parts[1].upper()
        band  = "_".join(parts[2:])

        for tp in TP_ORDER:
            if tp not in all_dfs or all_dfs[tp].empty:
                continue
            df = all_dfs[tp]
            if feat not in df.columns:
                continue
            am = df.groupby(["animal_id","group"])[feat].mean().reset_index()
            wt = am[am.group=="WT"][feat].dropna().values
            ko = am[am.group=="KO"][feat].dropna().values
            if len(wt)<2 or len(ko)<2:
                continue
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            d, d_lo, d_hi = cohens_d_ci(wt, ko)
            ci = (d_lo>0) or (d_hi<0)
            rows.append({
                "feature": feat, "timepoint": tp,
                "channel": ch.upper(), "state": state, "band": band,
                "is_primary": False,
                "wt_mean": np.mean(wt), "ko_mean": np.mean(ko),
                "cohens_d": d, "d_ci_lo": d_lo, "d_ci_hi": d_hi,
                "ci_excludes_zero": ci, "pval": p,
                "n_wt": len(wt), "n_ko": len(ko),
            })
            if p < 0.10 or ci:
                print(f"  [{tp}] {feat}: "
                      f"WT={np.mean(wt):.5f} KO={np.mean(ko):.5f} "
                      f"d={d:.3f} [{d_lo:.3f},{d_hi:.3f}] p={p:.5f}"
                      f"{'  CI✓' if ci else ''}")

    stat_df = pd.DataFrame(rows)
    if len(stat_df) > 1:
        stat_df["pval_fdr"] = fdr_bh(stat_df["pval"].values).round(6)
        stat_df["fdr_sig"]  = stat_df["pval_fdr"] < 0.05

    stat_df.to_csv(os.path.join(RESULTS_DIR, "pac_state_specific_stats.csv"),
                   index=False)
    print(f"\nSaved: pac_state_specific_stats.csv ({len(stat_df)} rows)")

    # Summary
    fdr_sig = stat_df[stat_df.get("fdr_sig", pd.Series(False))==True] \
        if "fdr_sig" in stat_df.columns else pd.DataFrame()
    ci_sig  = stat_df[stat_df.ci_excludes_zero]

    print(f"\nFDR significant: {len(fdr_sig)}")
    for _, r in fdr_sig.iterrows():
        print(f"  *** [{r.timepoint}] {r.feature}: "
              f"d={r.cohens_d:.3f} p={r.pval:.5f} FDR={r.pval_fdr:.5f}")

    print(f"\nCI excludes zero: {len(ci_sig)}")
    for _, r in ci_sig.sort_values("pval").iterrows():
        print(f"  ◄ [{r.timepoint}] {r.feature}: "
              f"WT={r.wt_mean:.5f} KO={r.ko_mean:.5f} "
              f"d={r.cohens_d:.3f} [{r.d_ci_lo:.3f},{r.d_ci_hi:.3f}] "
              f"p={r.pval:.5f}")

    return stat_df


# ── Figures ───────────────────────────────────────────────────────────────

def plot_pac_trajectory(all_dfs, stat_df, state, channel="ctx"):
    """
    Trajectory plot for all PAC bands in one state, one channel.
    Primary figure: CTX REM (Scekic-Zahirovic replication).
    """
    fig, axes = plt.subplots(1, len(PAC_BANDS), figsize=(len(PAC_BANDS)*4, 5),
                              sharey=False)
    if len(PAC_BANDS) == 1:
        axes = [axes]

    state_lower = state.lower()
    ch_label    = "CTX (cortex)" if channel == "ctx" else "CA3 (hippocampus)"

    for ax, (band_key, band_info) in zip(axes, PAC_BANDS.items()):
        feat = f"{channel}_{state_lower}_{band_key}"
        is_primary = (state == "REM" and channel == "ctx" and band_key == "T_HG")

        tp_top = {}   # highest (mean+sem) across groups, per timepoint x-position

        for group, color in COLORS.items():
            xs, means, sems = [], [], []
            for tp, tp_x in zip(TP_ORDER, TP_X):
                if tp not in all_dfs or all_dfs[tp].empty:
                    continue
                df = all_dfs[tp]
                if feat not in df.columns:
                    continue
                am   = df.groupby(["animal_id","group"])[feat].mean().reset_index()
                vals = am[am.group==group][feat].dropna()
                if len(vals) == 0:
                    continue
                xs.append(tp_x); means.append(vals.mean()); sems.append(vals.sem())
                tp_top[tp_x] = max(tp_top.get(tp_x, -np.inf), vals.max())
                ax.scatter([tp_x]*len(vals), vals,
                            color=color, s=25, alpha=0.4, zorder=3)

        # Significance markers from stats, positioned above each timepoint's top data point
        if len(stat_df) > 0 and "feature" in stat_df.columns:
            y0, y1 = ax.get_ylim()
            yspan = (y1 - y0) if y1 > y0 else 1e-3
            top_label_y = y1
            for tp, tp_x in zip(TP_ORDER, TP_X):
                sub = stat_df[(stat_df.feature==feat) & (stat_df.timepoint==tp)]
                if len(sub) == 0:
                    continue
                r = sub.iloc[0]
                pv = float(r.pval)
                if r.get("fdr_sig", False):
                    star, col = "**", "darkred"
                elif r.ci_excludes_zero:
                    star, col = "*", "#D85A30"
                elif pv < 0.05:
                    star, col = "*", "gray"
                else:
                    star, col = "ns", "gray"
                ptxt = "p<0.001" if pv < 0.001 else f"p={pv:.3f}"

                y_data = tp_top.get(tp_x, y0 + yspan * 0.45)
                y_ptxt = y_data + yspan * 0.05   # p-value text just above top point
                y_star = y_ptxt + yspan * 0.05   # star above the p-value
                top_label_y = max(top_label_y, y_star + yspan * 0.06)

                ax.text(tp_x, y_ptxt, ptxt, ha="center", va="bottom",
                        fontsize=6, color=col)
                ax.text(tp_x, y_star, star, ha="center", va="bottom",
                        fontsize=10, color=col, weight="bold")
            # expand y-axis so the tallest label sits inside the frame
            ax.set_ylim(y0, max(y1, top_label_y))

        ax.set_xticks(TP_X); ax.set_xticklabels(TP_ORDER, fontsize=9)
        xpad = (max(TP_X) - min(TP_X)) * 0.08
        ax.set_xlim(min(TP_X) - xpad, max(TP_X) + xpad)
        ax.set_xlabel("Age (months)", fontsize=10)
        ax.set_ylabel("Modulation Index", fontsize=10)
        title = f"{band_info['label']}"
        if is_primary:
            title += "\n★ PRIMARY (Scekic-Zahirovic replication)"
        ax.set_title(title, fontsize=9,
                     color="darkred" if is_primary else "black")
        ax.legend(fontsize=8)

        if is_primary:
            for spine in ax.spines.values():
                spine.set_edgecolor("darkred")
                spine.set_linewidth(2)

    state_label = {"REM": "REM Sleep", "Wake": "Active Wake", "NREM": "NREM Sleep"}
    fig.suptitle(f"Phase-Amplitude Coupling — {state_label.get(state, state)}\n"
                  f"{ch_label} | C9orf72-KO vs WT\n"
                  f"Replication of Scekic-Zahirovic et al. 2024 (Sci Transl Med)",
                  fontsize=10)
    plt.tight_layout()
    fname = f"pac_{channel}_{state_lower}_trajectory.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_replication_summary(all_dfs, stat_df):
    """
    Summary figure: T-HG PAC across states and channels.
    Shows both CTX and CA3 for REM, Wake, NREM — 6 panels.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    band_key  = "T_HG"   # high gamma — primary band
    band_info = PAC_BANDS[band_key]

    panel_configs = [
        ("ctx", "REM",  True,  "CTX — REM\n★ Scekic-Zahirovic replication"),
        ("ctx", "Wake", False, "CTX — Wake\n(Scekic-Zahirovic: sig in SOD1)"),
        ("ctx", "NREM", False, "CTX — NREM\n(exploratory)"),
        ("ca3", "REM",  False, "CA3 — REM\n(hippocampal, novel)"),
        ("ca3", "Wake", False, "CA3 — Wake\n(novel)"),
        ("ca3", "NREM", False, "CA3 — NREM\n(novel)"),
    ]

    for ax, (ch, state, is_primary, title) in zip(axes.flat, panel_configs):
        feat = f"{ch}_{state.lower()}_{band_key}"
        tp_top = {}   # highest (mean+sem) across groups, per timepoint x-position

        for group, color in COLORS.items():
            xs, means, sems = [], [], []
            for tp, tp_x in zip(TP_ORDER, TP_X):
                if tp not in all_dfs or all_dfs[tp].empty:
                    continue
                df = all_dfs[tp]
                if feat not in df.columns:
                    continue
                am   = df.groupby(["animal_id","group"])[feat].mean().reset_index()
                vals = am[am.group==group][feat].dropna()
                if len(vals) == 0:
                    continue
                xs.append(tp_x); means.append(vals.mean()); sems.append(vals.sem())
                tp_top[tp_x] = max(tp_top.get(tp_x, -np.inf), vals.max())
                ax.scatter([tp_x]*len(vals), vals,
                            color=color, s=20, alpha=0.35, zorder=3)
            if xs:
                ax.fill_between(xs,
                                 np.array(means)-np.array(sems),
                                 np.array(means)+np.array(sems),
                                 color=color, alpha=0.12)
                ax.plot(xs, means, "-o", color=color, lw=2,
                         markersize=7, label=group, zorder=4)

        # Significance markers from stats, positioned above each timepoint's data
        if len(stat_df) > 0 and "feature" in stat_df.columns:
            y0, y1 = ax.get_ylim()
            yspan = (y1 - y0) if y1 > y0 else 1e-3
            top_label_y = y1
            for tp, tp_x in zip(TP_ORDER, TP_X):
                sub = stat_df[(stat_df.feature==feat) & (stat_df.timepoint==tp)]
                if len(sub) == 0:
                    continue
                r = sub.iloc[0]
                pv = float(r.pval)
                if r.get("fdr_sig", False):
                    star, col = "**", "darkred"
                elif r.ci_excludes_zero:
                    star, col = "*", "#D85A30"
                elif pv < 0.05:
                    star, col = "*", "gray"
                else:
                    star, col = "ns", "gray"
                ptxt = "p<0.001" if pv < 0.001 else f"p={pv:.3f}"

                y_data = tp_top.get(tp_x, y0 + yspan * 0.5)
                y_lab  = y_data + yspan * 0.05
                y_ptxt = y_lab  + yspan * 0.045
                top_label_y = max(top_label_y, y_ptxt + yspan * 0.04)

                ax.text(tp_x, y_lab, star, ha="center", va="bottom",
                        fontsize=10, color=col, weight="bold")
                ax.text(tp_x, y_ptxt, ptxt, ha="center", va="bottom",
                        fontsize=6, color=col)
            if top_label_y > y1:
                ax.set_ylim(y0, top_label_y)

        ax.set_xticks(TP_X); ax.set_xticklabels(TP_ORDER, fontsize=8)
        xpad = (max(TP_X) - min(TP_X)) * 0.08
        ax.set_xlim(min(TP_X) - xpad, max(TP_X) + xpad)
        ax.set_xlabel("Age (months)", fontsize=9)
        ax.set_ylabel("MI (theta-high gamma PAC)", fontsize=9)
        ax.set_title(title, fontsize=9,
                     color="darkred" if is_primary else "black")
        if is_primary:
            for spine in ax.spines.values():
                spine.set_edgecolor("darkred"); spine.set_linewidth(2)
        ax.legend(fontsize=7)

    fig.suptitle("Theta-High Gamma PAC — C9orf72-KO vs WT\n"
                  "All states × channels | ★ = Scekic-Zahirovic 2024 replication test\n"
                  "† p<0.05  ◄ CI excl. 0  *** FDR q<0.05",
                  fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "pac_scekic_replication_summary.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: pac_scekic_replication_summary.png")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timepoint", default=None,
                        help="Single timepoint (e.g. 3m). Default: all.")
    args = parser.parse_args()

    print("=" * 60)
    print("STATE-SPECIFIC PAC — SCEKIC-ZAHIROVIC 2024 REPLICATION")
    print("Primary test: CTX theta-HIGH gamma during REM sleep")
    print("=" * 60)
    print("\nPAC bands:")
    for k, v in PAC_BANDS.items():
        primary = " ← PRIMARY (Scekic-Zahirovic)" if k == "T_HG" else ""
        print(f"  {k}: theta phase × {v['amp'][0]}-{v['amp'][1]} Hz amp"
              f" [{v['label']}]{primary}")
    print(f"\nStates: {STATES_TO_ANALYZE}")
    print(f"Channels: CA3 (hippocampal) + CTX (cortical — replication)")

    inv_path = os.path.join(DATA_DIR, "file_inventory_all_timepoints.csv")
    if not os.path.exists(inv_path):
        print("ERROR: Run script 04 first")
        sys.exit(1)

    inventory = pd.read_csv(inv_path)
    inventory["animal_id"] = inventory["animal_id"].astype(str)
    inventory = inventory.drop_duplicates(subset=["abf_path"])

    tp_to_run = [args.timepoint] if args.timepoint else TP_ORDER
    all_dfs   = {}

    for tp in tp_to_run:
        # Check if already processed
        path = os.path.join(RESULTS_DIR, f"pac_state_specific_{tp}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["animal_id"] = df["animal_id"].astype(str)
            all_dfs[tp] = df
            print(f"\n  Loaded {tp}: {len(df)} rows (cached)")
        else:
            if tp not in inventory["timepoint"].unique():
                continue
            df = process_timepoint(tp, inventory)
            if not df.empty:
                all_dfs[tp] = df

    if not all_dfs:
        print("No data available")
        return

    # Statistics
    stat_df = run_statistics(all_dfs)

    # Figures
    print("\n=== GENERATING FIGURES ===")
    # Primary: CTX REM all bands
    plot_pac_trajectory(all_dfs, stat_df, "REM", channel="ctx")
    # CA3 REM (novel hippocampal)
    plot_pac_trajectory(all_dfs, stat_df, "REM", channel="ca3")
    # Wake
    plot_pac_trajectory(all_dfs, stat_df, "Wake", channel="ctx")
    # Summary: T-HG across all states and channels
    plot_replication_summary(all_dfs, stat_df)

    print("\n" + "="*60)
    print("REPLICATION ANALYSIS COMPLETE")
    print(f"Key output: pac_scekic_replication_summary.png")
    print(f"Key stats:  pac_state_specific_stats.csv")
    print("="*60)


if __name__ == "__main__":
    main()
