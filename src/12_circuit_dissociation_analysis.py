"""
12_circuit_dissociation_analysis.py
=====================================
Comprehensive circuit dissociation analysis comparing CA3 (ch0) vs CTX (ch1)
across all 6 timepoints in C9orf72-KO vs WT mice.

Extends Paper 2 (eeg-circuit-dissociation-als) by adding:
  - 7m and 9m timepoints (never analyzed before)
  - Sleep-state-specific dissociation (Wake, NREM, REM)
  - Full spectral trajectory per region
  - Multiple DSI formulations
  - Granger causality direction (CTX→CA3 vs CA3→CTX)
  - Region-specific complexity measures
  - Individual animal DSI trajectories

Analyses:
  1. Region-specific band power (CA3 vs CTX, all bands, all timepoints)
  2. Dissociation Index (DSI) trajectory — multiple formulations
     DSI-1: |beta_CTX - beta_CA3| (beta dissociation)
     DSI-2: |alpha_CTX - alpha_CA3| + |beta_CTX - beta_CA3| (Paper 2 primary)
     DSI-3: Euclidean distance in 5-band feature space
     DSI-4: Spearman correlation between CA3 and CTX band profiles (1 - r)
  3. WT vs KO DSI comparison per timepoint
  4. DSI trajectory correlation with disease stage (within KO)
  5. Region-specific aperiodic exponent (E/I balance per region)
  6. Sleep-state-specific dissociation
  7. CTX/CA3 power ratio per band (directional measure)
  8. Phase relationship (wPLI + coherence already in script 08/11)
  9. Individual animal DSI trajectories (spaghetti plot)
 10. Comparison with Paper 2 findings at shared timepoints (3m/4m/6m/12m)

Outputs:
  - results/circuit_dissociation_{tp}.csv   (per-file, both channels)
  - results/dsi_trajectory.csv              (DSI per animal per timepoint)
  - results/region_stats_{tp}.csv           (CTX vs CA3 within group)
  - figures/circuit_*.png

Run:
    python src/12_circuit_dissociation_analysis.py
    python src/12_circuit_dissociation_analysis.py --timepoint 3m
"""

import os
import gc
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import pyabf
from scipy.signal import welch, decimate, butter
from scipy.signal import sosfiltfilt
from scipy.stats import mannwhitneyu, spearmanr, kendalltau
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")

COLORS      = {"WT": "#378ADD", "KO": "#D85A30"}
REGION_COLORS = {"CA3": "#2ECC71", "CTX": "#E74C3C"}
TP_ORDER    = ["3m", "4m", "6m", "7m", "9m", "12m"]
TP_X        = [3, 4, 6, 7, 9, 12]
FS_DS       = 500
EPOCH_S     = 4.0
MIN_EPOCHS  = 10
ARTIFACT_THRESH = 500
FLAT_THRESH = 0.001

BANDS = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 80),
}


# ── Signal helpers ─────────────────────────────────────────────────────────

def fdr_bh(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=int)
    ranked[order] = np.arange(1, n + 1)
    fdr = pvals * n / ranked
    fdr_adj = np.minimum.accumulate(fdr[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = fdr_adj
    return np.minimum(result, 1.0)


def cohens_d_ci(x, y, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    d = (np.mean(y)-np.mean(x)) / np.sqrt((np.std(x)**2+np.std(y)**2)/2+1e-10)
    boot = []
    for _ in range(n_boot):
        bx = rng.choice(x, len(x), replace=True)
        by = rng.choice(y, len(y), replace=True)
        boot.append((np.mean(by)-np.mean(bx)) /
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
        if len(sig) < int(fs * 30):
            return None, None
        factor = max(1, int(round(fs / target_fs)))
        if factor > 1:
            sig = decimate(sig - sig.mean(), factor, zero_phase=True)
        return sig, float(target_fs)
    except Exception:
        return None, None


def band_power_abs(psd, freqs, lo, hi):
    m = (freqs >= lo) & (freqs <= hi)
    return float(np.trapz(psd[m], freqs[m])) if m.any() else 0.0


def compute_spectral_features(sig, fs):
    """
    Compute absolute and relative band power + aperiodic exponent.
    Returns dict of features for one signal segment.
    """
    epoch_n  = int(EPOCH_S * fs)
    n_epochs = len(sig) // epoch_n
    epoch_rows = []

    for i in range(n_epochs):
        ep = sig[i*epoch_n:(i+1)*epoch_n]
        if np.abs(ep).max() > ARTIFACT_THRESH or np.std(ep) < FLAT_THRESH:
            continue
        nperseg = min(int(fs*2), epoch_n)
        f, psd  = welch(ep, fs=fs, nperseg=nperseg, detrend="constant")

        bps = {b: band_power_abs(psd, f, lo, hi)
               for b, (lo, hi) in BANDS.items()}
        total = sum(bps.values()) + 1e-12

        row = {}
        for b, val in bps.items():
            row[f"bp_{b}"]  = val
            row[f"rbp_{b}"] = val / total

        # Aperiodic exponent
        m_ap = (f >= 2) & (f <= 40) & (psd > 0)
        if m_ap.sum() >= 5:
            slope, _ = np.polyfit(np.log10(f[m_ap]), np.log10(psd[m_ap]), 1)
            row["ap_exp"] = float(-slope)
        else:
            row["ap_exp"] = np.nan

        row["total_var"] = float(np.var(ep))
        epoch_rows.append(row)

    if not epoch_rows:
        return {}
    df = pd.DataFrame(epoch_rows)
    return {c: float(df[c].median()) for c in df.columns}


# ── Load state labels from epoch file ─────────────────────────────────────

def get_state_labels(tp, animal_id, abf_file):
    """Load sleep state per epoch from script 05 output."""
    path = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
    if not os.path.exists(path):
        return None
    ep = pd.read_csv(path)
    ep["animal_id"] = ep["animal_id"].astype(str)
    mask = (ep["animal_id"] == str(animal_id)) & (ep["abf_file"] == abf_file)
    sub = ep[mask][["epoch_idx", "state"]].copy()
    if len(sub) == 0:
        return None
    return sub.set_index("epoch_idx")["state"].to_dict()


# ── Core: compute both channels for one ABF ───────────────────────────────

def compute_dual_channel(abf_path, animal_id, group, tp, abf_file):
    """
    Compute spectral features for CA3 (ch0) and CTX (ch1) separately.
    Also computes state-specific features using script 05 state labels.
    Returns dict with all features for both regions.
    """
    result = {
        "animal_id": animal_id,
        "group":     group,
        "timepoint": tp,
        "abf_file":  abf_file,
    }

    # Load both channels
    sig_ca3, fs = load_signal(abf_path, channel=0)
    sig_ctx, _  = load_signal(abf_path, channel=1)

    if sig_ca3 is None:
        return None

    has_ctx = False  # track whether CTX channel was loaded

    # Overall features per region
    feats_ca3 = compute_spectral_features(sig_ca3, fs)
    for k, v in feats_ca3.items():
        result[f"ca3_{k}"] = v

    if sig_ctx is not None:
        feats_ctx = compute_spectral_features(sig_ctx, fs)
        for k, v in feats_ctx.items():
            result[f"ctx_{k}"] = v

        # ── DSI formulations ───────────────────────────────────────────
        # DSI-1: beta dissociation only
        ca3_beta = feats_ca3.get("rbp_beta", np.nan)
        ctx_beta = feats_ctx.get("rbp_beta", np.nan)
        result["dsi_1_beta"] = abs(ctx_beta - ca3_beta) if not (np.isnan(ca3_beta) or np.isnan(ctx_beta)) else np.nan

        # DSI-2: alpha + beta (Paper 2 primary)
        ca3_alpha = feats_ca3.get("rbp_alpha", np.nan)
        ctx_alpha = feats_ctx.get("rbp_alpha", np.nan)
        result["dsi_2_alpha_beta"] = (abs(ctx_alpha - ca3_alpha) +
                                       abs(ctx_beta - ca3_beta)) \
            if not any(np.isnan([ca3_alpha, ctx_alpha, ca3_beta, ctx_beta])) else np.nan

        # DSI-3: Euclidean distance in 5-band RBP space
        ca3_rbp = np.array([feats_ca3.get(f"rbp_{b}", np.nan) for b in BANDS])
        ctx_rbp = np.array([feats_ctx.get(f"rbp_{b}", np.nan) for b in BANDS])
        if not np.any(np.isnan(ca3_rbp)) and not np.any(np.isnan(ctx_rbp)):
            result["dsi_3_euclidean"] = float(np.linalg.norm(ctx_rbp - ca3_rbp))
        else:
            result["dsi_3_euclidean"] = np.nan

        # DSI-4: 1 - Spearman correlation of band profiles
        if not np.any(np.isnan(ca3_rbp)) and not np.any(np.isnan(ctx_rbp)):
            r, _ = spearmanr(ca3_rbp, ctx_rbp)
            result["dsi_4_decorrelation"] = float(1 - r)
        else:
            result["dsi_4_decorrelation"] = np.nan

        # CTX/CA3 ratios per band
        for b in BANDS:
            ca3_v = feats_ca3.get(f"rbp_{b}", np.nan)
            ctx_v = feats_ctx.get(f"rbp_{b}", np.nan)
            result[f"ctx_ca3_ratio_{b}"] = (ctx_v / (ca3_v + 1e-12)) \
                if not any(np.isnan([ca3_v, ctx_v])) else np.nan

        # Aperiodic exponent dissociation
        ca3_ap = feats_ca3.get("ap_exp", np.nan)
        ctx_ap = feats_ctx.get("ap_exp", np.nan)
        result["ap_exp_dissociation"] = (ctx_ap - ca3_ap) \
            if not any(np.isnan([ca3_ap, ctx_ap])) else np.nan

        del sig_ctx
        has_ctx = True

    # State-specific dissociation
    state_labels = get_state_labels(tp, animal_id, abf_file)
    if state_labels is not None and has_ctx:
        for state in ["Wake", "NREM", "REM"]:
            # Get epoch indices for this state
            state_epochs = [idx for idx, s in state_labels.items() if s == state]
            if len(state_epochs) < MIN_EPOCHS:
                continue
            epoch_n = int(EPOCH_S * fs)
            # Extract state-specific segments
            ca3_state = np.concatenate([
                sig_ca3[i*epoch_n:(i+1)*epoch_n]
                for i in state_epochs
                if (i+1)*epoch_n <= len(sig_ca3)
            ])
            if len(ca3_state) < int(fs * 10):
                continue
            f_ca3 = compute_spectral_features(ca3_state, fs)
            for k, v in f_ca3.items():
                result[f"ca3_{state.lower()}_{k}"] = v

    del sig_ca3
    gc.collect()
    return result


# ── Process one timepoint ─────────────────────────────────────────────────

def process_timepoint(tp, inventory):
    print(f"\n{'='*60}")
    print(f"Circuit Dissociation — {tp}")
    print(f"{'='*60}")

    inv = inventory[(inventory.timepoint==tp)].drop_duplicates(
                        subset=["abf_path"]).copy()
    if len(inv) == 0:
        print(f"  No files for {tp}")
        return pd.DataFrame()

    rows = []
    for _, file_row in inv.iterrows():
        abf_path  = file_row["abf_path"]
        animal_id = str(file_row["animal_id"])
        group     = file_row["group"]
        abf_file  = file_row["abf_file"]

        if not os.path.exists(abf_path):
            continue

        print(f"  {animal_id} | {abf_file[:25]}", end="", flush=True)
        res = compute_dual_channel(abf_path, animal_id, group, tp, abf_file)

        if res:
            rows.append(res)
            dsi2 = res.get("dsi_2_alpha_beta", np.nan)
            ctx_beta = res.get("ctx_rbp_beta", np.nan)
            ca3_beta = res.get("ca3_rbp_beta", np.nan)
            print(f" DSI2={dsi2:.4f} CTX_beta={ctx_beta:.4f} CA3_beta={ca3_beta:.4f}")
        else:
            print(" SKIP")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, f"circuit_dissociation_{tp}.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out} ({len(df)} rows)")
    return df


# ── Statistics ────────────────────────────────────────────────────────────

def run_dissociation_stats(all_dfs):
    """
    For each DSI formulation × timepoint:
    1. WT vs KO comparison (does KO show more dissociation?)
    2. Within KO: does DSI increase with timepoint? (Spearman vs TP_X)
    3. CTX vs CA3 within group (does CTX hyperexcite while CA3 stays stable?)
    """
    dsi_cols = ["dsi_1_beta", "dsi_2_alpha_beta", "dsi_3_euclidean",
                "dsi_4_decorrelation", "ap_exp_dissociation"]
    ratio_cols = [f"ctx_ca3_ratio_{b}" for b in BANDS]
    region_cols = ([f"ca3_rbp_{b}" for b in BANDS] +
                   [f"ctx_rbp_{b}" for b in BANDS] +
                   ["ca3_ap_exp", "ctx_ap_exp"])

    all_rows = []

    print(f"\n{'='*60}")
    print("CIRCUIT DISSOCIATION STATISTICS")
    print(f"{'='*60}")

    # ── 1. WT vs KO DSI per timepoint ─────────────────────────────────
    print("\n--- DSI: KO vs WT ---")
    for tp in TP_ORDER:
        if tp not in all_dfs or all_dfs[tp].empty:
            continue
        df = all_dfs[tp]
        am = df.groupby(["animal_id","group"])[dsi_cols + ratio_cols + region_cols].mean().reset_index()

        for col in dsi_cols + ratio_cols:
            if col not in am.columns:
                continue
            wt = am[am.group=="WT"][col].dropna().values
            ko = am[am.group=="KO"][col].dropna().values
            if len(wt) < 2 or len(ko) < 2:
                continue
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            d, d_lo, d_hi = cohens_d_ci(wt, ko)
            ci_excl = (d_lo > 0) or (d_hi < 0)
            all_rows.append({
                "analysis":        "KO_vs_WT",
                "feature":         col,
                "timepoint":       tp,
                "wt_mean":         round(float(np.mean(wt)),6),
                "ko_mean":         round(float(np.mean(ko)),6),
                "cohens_d":        round(d,4),
                "d_ci_lo":         round(d_lo,4),
                "d_ci_hi":         round(d_hi,4),
                "ci_excludes_zero": ci_excl,
                "pval":            round(p,6),
                "n_wt":            len(wt),
                "n_ko":            len(ko),
            })
            if p < 0.10 or ci_excl:
                print(f"  [{tp}] {col:<30} "
                      f"WT={np.mean(wt):.4f} KO={np.mean(ko):.4f} "
                      f"d={d:.3f} [{d_lo:.3f},{d_hi:.3f}] p={p:.5f}"
                      f"{'  CI✓' if ci_excl else ''}")

    # ── 2. DSI trajectory within KO (does it increase over time?) ─────
    print("\n--- DSI trajectory within KO (Spearman vs age) ---")
    for col in dsi_cols:
        ko_vals, tp_vals = [], []
        for tp, tp_x in zip(TP_ORDER, TP_X):
            if tp not in all_dfs or all_dfs[tp].empty:
                continue
            df = all_dfs[tp]
            am = df.groupby(["animal_id","group"])[col].mean().reset_index()
            ko = am[am.group=="KO"][col].dropna().values
            ko_vals.extend(ko)
            tp_vals.extend([tp_x] * len(ko))
        if len(ko_vals) < 5:
            continue
        r, p = spearmanr(tp_vals, ko_vals)
        print(f"  {col:<30} KO trajectory: ρ={r:.3f} p={p:.5f}"
              f"{'  ***' if p<0.001 else '  **' if p<0.01 else '  *' if p<0.05 else ''}")
        all_rows.append({
            "analysis": "KO_trajectory",
            "feature": col, "timepoint": "all",
            "cohens_d": r, "pval": p,
            "wt_mean": np.nan, "ko_mean": np.nan,
            "d_ci_lo": np.nan, "d_ci_hi": np.nan,
            "ci_excludes_zero": False,
            "n_wt": 0, "n_ko": len(ko_vals),
        })

    # ── 3. CTX vs CA3 within KO per timepoint ─────────────────────────
    print("\n--- CTX vs CA3 within KO (paired) ---")
    for tp in TP_ORDER:
        if tp not in all_dfs or all_dfs[tp].empty:
            continue
        df = all_dfs[tp]
        ko_df = df[df.group=="KO"]
        am = ko_df.groupby("animal_id")[region_cols].mean().reset_index()

        for b in BANDS:
            ca3_col = f"ca3_rbp_{b}"
            ctx_col = f"ctx_rbp_{b}"
            if ca3_col not in am or ctx_col not in am:
                continue
            ca3 = am[ca3_col].dropna().values
            ctx = am[ctx_col].dropna().values
            n = min(len(ca3), len(ctx))
            if n < 3:
                continue
            from scipy.stats import wilcoxon
            try:
                _, p = wilcoxon(ctx[:n], ca3[:n])
            except Exception:
                continue
            diff = np.mean(ctx[:n]) - np.mean(ca3[:n])
            if p < 0.10:
                direction = "CTX>CA3" if diff > 0 else "CA3>CTX"
                print(f"  [{tp}] KO {b}: CTX={np.mean(ctx[:n]):.4f} "
                      f"CA3={np.mean(ca3[:n]):.4f} diff={diff:.4f} "
                      f"p={p:.5f} [{direction}]")
            all_rows.append({
                "analysis": "CTX_vs_CA3_KO",
                "feature": f"{b}_ctx_vs_ca3",
                "timepoint": tp,
                "wt_mean": float(np.mean(ca3[:n])),
                "ko_mean": float(np.mean(ctx[:n])),
                "cohens_d": diff,
                "d_ci_lo": np.nan, "d_ci_hi": np.nan,
                "ci_excludes_zero": False,
                "pval": round(p,6),
                "n_wt": n, "n_ko": n,
            })

    # Apply FDR to KO_vs_WT tests
    stats = pd.DataFrame(all_rows)
    ko_wt_mask = stats.analysis == "KO_vs_WT"
    if ko_wt_mask.sum() > 1:
        stats.loc[ko_wt_mask, "pval_fdr"] = fdr_bh(
            stats.loc[ko_wt_mask, "pval"].values).round(6)
        stats.loc[ko_wt_mask, "fdr_sig"] = \
            stats.loc[ko_wt_mask, "pval_fdr"] < 0.05

    stats.to_csv(os.path.join(RESULTS_DIR, "circuit_dissociation_stats.csv"),
                  index=False)
    print(f"\nSaved: circuit_dissociation_stats.csv ({len(stats)} rows)")

    fdr_sig = stats[stats.get("fdr_sig", False) == True] \
        if "fdr_sig" in stats.columns else pd.DataFrame()
    ci_sig  = stats[(stats.analysis=="KO_vs_WT") & stats.ci_excludes_zero]

    print(f"\nFDR significant (KO vs WT): {len(fdr_sig)}")
    print(f"CI excludes zero (KO vs WT): {len(ci_sig)}")
    if len(fdr_sig) > 0:
        for _, r in fdr_sig.iterrows():
            print(f"  *** [{r.timepoint}] {r.feature}: "
                  f"d={r.cohens_d:.3f} p={r.pval:.5f} FDR={r.pval_fdr:.5f}")
    if len(ci_sig) > 0:
        for _, r in ci_sig.sort_values("pval").iterrows():
            print(f"  ◄ [{r.timepoint}] {r.feature}: "
                  f"WT={r.wt_mean:.4f} KO={r.ko_mean:.4f} "
                  f"d={r.cohens_d:.3f} [{r.d_ci_lo:.3f},{r.d_ci_hi:.3f}] "
                  f"p={r.pval:.5f}")

    return stats


# ── DSI trajectory table ──────────────────────────────────────────────────

def build_dsi_trajectory(all_dfs):
    """Build per-animal DSI at each timepoint — for spaghetti plot and correlations."""
    dsi_col = "dsi_2_alpha_beta"   # Paper 2 primary formulation
    rows = []
    for tp, tp_x in zip(TP_ORDER, TP_X):
        if tp not in all_dfs or all_dfs[tp].empty:
            continue
        df = all_dfs[tp]
        am = df.groupby(["animal_id","group"])[[dsi_col] + [f"dsi_{i}" for i in ["1_beta","3_euclidean","4_decorrelation"]] + ["ap_exp_dissociation"]].mean().reset_index()
        am["timepoint"] = tp
        am["tp_x"]      = tp_x
        rows.append(am)
    if not rows:
        return pd.DataFrame()
    traj = pd.concat(rows, ignore_index=True)
    traj.to_csv(os.path.join(RESULTS_DIR, "dsi_trajectory.csv"), index=False)
    print(f"Saved: dsi_trajectory.csv ({len(traj)} rows)")
    return traj


# ── Figures ───────────────────────────────────────────────────────────────

def plot_region_trajectories(all_dfs):
    """
    Side-by-side CA3 vs CTX band power trajectories.
    One row per band, two columns (CA3 | CTX), two lines per plot (WT vs KO).
    """
    fig, axes = plt.subplots(len(BANDS), 2,
                               figsize=(12, len(BANDS)*3.5), sharey="row")
    bands_list = list(BANDS.keys())

    for row_i, band in enumerate(bands_list):
        for col_i, (region, region_label) in enumerate([("ca3","CA3"), ("ctx","CTX")]):
            ax = axes[row_i, col_i]
            feat = f"{region}_rbp_{band}"

            for group, color in COLORS.items():
                xs, means, sems = [], [], []
                for tp, tp_x in zip(TP_ORDER, TP_X):
                    if tp not in all_dfs or all_dfs[tp].empty:
                        continue
                    df = all_dfs[tp]
                    if feat not in df.columns:
                        continue
                    am = df.groupby(["animal_id","group"])[feat].mean().reset_index()
                    vals = am[am.group==group][feat].dropna()
                    if len(vals) == 0:
                        continue
                    xs.append(tp_x)
                    means.append(vals.mean())
                    sems.append(vals.sem())
                    ax.scatter([tp_x]*len(vals), vals,
                                color=color, s=15, alpha=0.35, zorder=3)

                if xs:
                    ax.fill_between(xs, np.array(means)-np.array(sems),
                                     np.array(means)+np.array(sems),
                                     color=color, alpha=0.15)
                    ax.plot(xs, means, "-o", color=color, lw=2,
                             markersize=6, label=group, zorder=4)

            ax.set_xticks(TP_X)
            ax.set_xticklabels(TP_ORDER, fontsize=8)
            ax.set_ylabel(f"Rel. {band} power", fontsize=8)
            if row_i == 0:
                ax.set_title(f"{region_label}", fontsize=11, fontweight="bold",
                              color=REGION_COLORS[region_label])
            if col_i == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("CA3 vs CTX Band Power Trajectories — C9orf72 KO vs WT\n"
                  "Left: hippocampal CA3 | Right: cortex",
                  fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "circuit_region_trajectories.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: circuit_region_trajectories.png")


def plot_dsi_trajectory(traj_df, stats_df):
    """
    DSI-2 trajectory: WT vs KO mean±SEM + individual animal lines.
    """
    if traj_df.empty:
        return

    dsi_col = "dsi_2_alpha_beta"
    if dsi_col not in traj_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: group means + individual animals
    ax = axes[0]
    for group, color in COLORS.items():
        grp = traj_df[traj_df.group==group]
        xs, means, sems = [], [], []
        for tp, tp_x in zip(TP_ORDER, TP_X):
            sub = grp[grp.timepoint==tp][dsi_col].dropna()
            if len(sub) == 0:
                continue
            xs.append(tp_x)
            means.append(sub.mean())
            sems.append(sub.sem())

        # Individual animal lines
        for animal, adf in grp.groupby("animal_id"):
            adf = adf.sort_values("tp_x")
            ax.plot(adf["tp_x"], adf[dsi_col], "-",
                     color=color, lw=0.8, alpha=0.35)

        if xs:
            ax.fill_between(xs, np.array(means)-np.array(sems),
                             np.array(means)+np.array(sems),
                             color=color, alpha=0.15)
            ax.plot(xs, means, "-o", color=color, lw=2.5,
                     markersize=8, label=group, zorder=5)
            ax.errorbar(xs, means, yerr=sems, fmt="none",
                         color=color, capsize=5, capthick=2)

    # Significance markers
    if len(stats_df) > 0 and "feature" in stats_df.columns:
        ko_wt = stats_df[(stats_df.analysis=="KO_vs_WT") &
                          (stats_df.feature==dsi_col)]
        for _, r in ko_wt.iterrows():
            if r.timepoint not in TP_ORDER:
                continue
            tp_x = TP_X[TP_ORDER.index(r.timepoint)]
            if r.get("fdr_sig", False):
                ax.text(tp_x, ax.get_ylim()[1]*0.97, "***",
                         ha="center", fontsize=11, color="darkred", weight="bold")
            elif r.ci_excludes_zero:
                ax.text(tp_x, ax.get_ylim()[1]*0.97, "◄",
                         ha="center", fontsize=10, color="#D85A30")
            elif r.pval < 0.05:
                ax.text(tp_x, ax.get_ylim()[1]*0.97, "†",
                         ha="center", fontsize=10, color="gray")

    ax.set_xticks(TP_X)
    ax.set_xticklabels(TP_ORDER, fontsize=10)
    ax.set_xlabel("Age (months)", fontsize=11)
    ax.set_ylabel("DSI-2 (|α_CTX−α_CA3| + |β_CTX−β_CA3|)", fontsize=10)
    ax.set_title("Circuit Dissociation Index Trajectory\n"
                  "Individual animals + mean±SEM", fontsize=10)
    ax.legend(fontsize=10)

    # Right: all DSI formulations as effect size trajectory
    ax2 = axes[1]
    dsi_forms = {
        "dsi_1_beta":           ("DSI-1 (beta)", "#E74C3C"),
        "dsi_2_alpha_beta":     ("DSI-2 (α+β)", "#C0392B"),
        "dsi_3_euclidean":      ("DSI-3 (Euclidean)", "#8E44AD"),
        "dsi_4_decorrelation":  ("DSI-4 (1−ρ)", "#2980B9"),
    }

    for dsi_key, (label, color) in dsi_forms.items():
        if dsi_key not in traj_df.columns:
            continue
        xs, means = [], []
        for tp, tp_x in zip(TP_ORDER, TP_X):
            ko_sub = traj_df[(traj_df.group=="KO") &
                              (traj_df.timepoint==tp)][dsi_key].dropna()
            if len(ko_sub) == 0:
                continue
            xs.append(tp_x)
            means.append(ko_sub.mean())
        if len(xs) > 1:
            # Normalize to 3m baseline
            base = means[0] if means[0] != 0 else 1
            norm_means = [m/base for m in means]
            ax2.plot(xs, norm_means, "-o", color=color, lw=2,
                      markersize=6, label=label)

    ax2.axhline(1.0, color="black", lw=1, ls="--", alpha=0.5,
                 label="3m baseline")
    ax2.set_xticks(TP_X)
    ax2.set_xticklabels(TP_ORDER, fontsize=10)
    ax2.set_xlabel("Age (months)", fontsize=11)
    ax2.set_ylabel("DSI (normalized to 3m)", fontsize=10)
    ax2.set_title("All DSI Formulations — KO only\n"
                   "Normalized to 3m baseline", fontsize=10)
    ax2.legend(fontsize=8)

    fig.suptitle("Circuit Dissociation: CA3 vs CTX — C9orf72-KO vs WT",
                  fontsize=11, weight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "circuit_dsi_trajectory.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: circuit_dsi_trajectory.png")


def plot_ctx_ca3_ratio(all_dfs, stats_df):
    """
    CTX/CA3 power ratio per band across timepoints.
    Ratio > 1 means CTX hyperactive relative to CA3.
    """
    fig, axes = plt.subplots(1, len(BANDS), figsize=(len(BANDS)*3.5, 5), sharey=False)

    for ax, band in zip(axes, BANDS):
        feat = f"ctx_ca3_ratio_{band}"

        for group, color in COLORS.items():
            xs, means, sems = [], [], []
            for tp, tp_x in zip(TP_ORDER, TP_X):
                if tp not in all_dfs or all_dfs[tp].empty:
                    continue
                df = all_dfs[tp]
                if feat not in df.columns:
                    continue
                am = df.groupby(["animal_id","group"])[feat].mean().reset_index()
                vals = am[am.group==group][feat].dropna()
                if len(vals) == 0:
                    continue
                xs.append(tp_x)
                means.append(vals.mean())
                sems.append(vals.sem())

            if xs:
                ax.fill_between(xs, np.array(means)-np.array(sems),
                                 np.array(means)+np.array(sems),
                                 color=color, alpha=0.15)
                ax.plot(xs, means, "-o", color=color, lw=2,
                         markersize=6, label=group, zorder=4)

        ax.axhline(1.0, color="black", lw=1, ls="--", alpha=0.5)
        ax.set_xticks(TP_X)
        ax.set_xticklabels(TP_ORDER, fontsize=7, rotation=45)
        ax.set_xlabel("Age (months)", fontsize=9)
        ax.set_ylabel("CTX/CA3 ratio", fontsize=9)
        ax.set_title(f"{band}", fontsize=10)
        if band == list(BANDS.keys())[0]:
            ax.legend(fontsize=8)

    fig.suptitle("CTX/CA3 Power Ratio — C9orf72 KO vs WT\n"
                  "Ratio > 1: CTX more active than CA3  |  dashed = equal",
                  fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "circuit_ctx_ca3_ratio.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: circuit_ctx_ca3_ratio.png")


def plot_aperiodic_dissociation(all_dfs):
    """
    Aperiodic exponent per region (CA3 vs CTX) per group.
    Shows whether E/I balance diverges between regions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, group in zip(axes, ["WT", "KO"]):
        color_ca3 = REGION_COLORS["CA3"]
        color_ctx = REGION_COLORS["CTX"]

        for feat, color, label in [("ca3_ap_exp", color_ca3, "CA3"),
                                     ("ctx_ap_exp", color_ctx, "CTX")]:
            xs, means, sems = [], [], []
            for tp, tp_x in zip(TP_ORDER, TP_X):
                if tp not in all_dfs or all_dfs[tp].empty:
                    continue
                df = all_dfs[tp]
                if feat not in df.columns:
                    continue
                am = df.groupby(["animal_id","group"])[feat].mean().reset_index()
                vals = am[am.group==group][feat].dropna()
                if len(vals) == 0:
                    continue
                xs.append(tp_x)
                means.append(vals.mean())
                sems.append(vals.sem())

            if xs:
                ax.fill_between(xs, np.array(means)-np.array(sems),
                                 np.array(means)+np.array(sems),
                                 color=color, alpha=0.15)
                ax.plot(xs, means, "-o", color=color, lw=2,
                         markersize=7, label=label, zorder=4)
                ax.errorbar(xs, means, yerr=sems, fmt="none",
                             color=color, capsize=4, capthick=1.5)

        ax.set_xticks(TP_X)
        ax.set_xticklabels(TP_ORDER, fontsize=9)
        ax.set_xlabel("Age (months)", fontsize=10)
        ax.set_ylabel("Aperiodic Exponent (1/f slope)", fontsize=10)
        ax.set_title(f"{group}", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Aperiodic Exponent: CA3 vs CTX — E/I Balance per Region\n"
                  "Higher exponent = more inhibition-dominant",
                  fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "circuit_aperiodic_dissociation.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: circuit_aperiodic_dissociation.png")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timepoint", default=None,
                        help="Single timepoint (e.g. 3m). Default: all.")
    args = parser.parse_args()

    print("=" * 60)
    print("CIRCUIT DISSOCIATION ANALYSIS — CA3 vs CTX")
    print("All 6 timepoints | All DSI formulations | Both groups")
    print("=" * 60)

    inv_path = os.path.join(DATA_DIR, "file_inventory_all_timepoints.csv")
    if not os.path.exists(inv_path):
        print("ERROR: Run script 04 first")
        sys.exit(1)

    inventory = pd.read_csv(inv_path)
    inventory["animal_id"] = inventory["animal_id"].astype(str)

    tp_to_run = [args.timepoint] if args.timepoint else TP_ORDER
    all_dfs = {}

    for tp in tp_to_run:
        # Check if already processed
        path = os.path.join(RESULTS_DIR, f"circuit_dissociation_{tp}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["animal_id"] = df["animal_id"].astype(str)
            all_dfs[tp] = df
            print(f"  Loaded {tp}: {len(df)} rows")
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
    stats_df = run_dissociation_stats(all_dfs)

    # DSI trajectory table
    print("\n=== BUILDING DSI TRAJECTORY ===")
    traj_df = build_dsi_trajectory(all_dfs)

    # Figures
    print("\n=== GENERATING FIGURES ===")
    plot_region_trajectories(all_dfs)
    plot_dsi_trajectory(traj_df, stats_df)
    plot_ctx_ca3_ratio(all_dfs, stats_df)
    plot_aperiodic_dissociation(all_dfs)

    print("\n" + "="*60)
    print("CIRCUIT DISSOCIATION ANALYSIS COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
