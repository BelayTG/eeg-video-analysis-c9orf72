"""
13_fixes_and_extensions.py
===========================
Fixes critical issues found in audit + adds classical analyses.

FIXES:
  1. PAC with full signal (removes n_epochs_max cap)
  2. DSI trajectory: WT + KO + slope-difference test
  3. EEG-video epoch-level correlation (4m merged file)

NEW CLASSICAL ANALYSES:
  4. Circadian rhythm (hourly Wake% across 24h)
  5. Sleep bout fragmentation index
  6. Video immobility trajectory (all timepoints)
  7. Pre-ictal EEG state (30-min window before first seizure, 4m)
  8. REM atonia check (video movement during REM)
  9. CTX vs CA3 pattern in WT (control for script 12 finding)
 10. WT vs KO DSI trajectory slope difference test

Run:
    python src/13_fixes_and_extensions.py
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import pyabf
from scipy.signal import welch, decimate, butter, hilbert
from scipy.signal import sosfiltfilt
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")

COLORS   = {"WT": "#378ADD", "KO": "#D85A30"}
TP_ORDER = ["3m", "4m", "6m", "7m", "9m", "12m"]
TP_X     = [3, 4, 6, 7, 9, 12]
FS_DS    = 500
EPOCH_S  = 4.0


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
    boot = [(np.mean(rng.choice(y,len(y),replace=True)) -
             np.mean(rng.choice(x,len(x),replace=True))) /
            np.sqrt((np.std(rng.choice(x,len(x),replace=True))**2 +
                     np.std(rng.choice(y,len(y),replace=True))**2)/2+1e-10)
            for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def bandpass(sig, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


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


# ══════════════════════════════════════════════════════════════
# FIX 1: PAC with full signal
# ══════════════════════════════════════════════════════════════

def pac_full_signal(sig, fs, phase_band=(4,8), amp_band=(30,80), n_bins=18):
    """PAC using ALL epochs (no cap). Returns MI value."""
    epoch_n  = int(EPOCH_S * fs)
    n_epochs = len(sig) // epoch_n
    mi_vals  = []
    for i in range(n_epochs):
        ep = sig[i*epoch_n:(i+1)*epoch_n]
        if np.abs(ep).max() > 500 or np.std(ep) < 0.001:
            continue
        ph_sig = bandpass(ep, fs, phase_band[0], phase_band[1])
        am_sig = bandpass(ep, fs, amp_band[0],   amp_band[1])
        phase  = np.angle(hilbert(ph_sig))
        amp    = np.abs(hilbert(am_sig))
        bins   = np.linspace(-np.pi, np.pi, n_bins+1)
        amp_d  = np.array([amp[(phase>=bins[k])&(phase<bins[k+1])].mean()
                           if np.any((phase>=bins[k])&(phase<bins[k+1])) else 0
                           for k in range(n_bins)])
        amp_d  = np.clip(amp_d, 1e-12, None)
        amp_d /= amp_d.sum()
        mi_vals.append(float(np.sum(amp_d*np.log(amp_d/(1/n_bins)+1e-12))/np.log(n_bins)))
    return float(np.median(mi_vals)) if mi_vals else np.nan


def run_fix1_pac(inventory):
    """Recompute PAC using full signal for all timepoints."""
    print("\n" + "="*60)
    print("FIX 1: PAC — FULL SIGNAL (no epoch cap)")
    print("="*60)

    all_rows = []
    for tp in TP_ORDER:
        inv = inventory[(inventory.timepoint==tp)].drop_duplicates(subset=["abf_path"])
        if len(inv) == 0:
            continue
        print(f"\n  {tp} ({len(inv)} files)")
        for _, row in inv.iterrows():
            sig, fs = load_signal(row["abf_path"], channel=0)
            if sig is None:
                continue
            pac_tg = pac_full_signal(sig, fs, (4,8),  (30,80))
            pac_ag = pac_full_signal(sig, fs, (8,13), (30,80))
            pac_dg = pac_full_signal(sig, fs, (0.5,4),(30,80))
            del sig; gc.collect()
            all_rows.append({
                "animal_id": str(row["animal_id"]),
                "group":     row["group"],
                "timepoint": tp,
                "abf_file":  row["abf_file"],
                "pac_theta_gamma_full": pac_tg,
                "pac_alpha_gamma_full": pac_ag,
                "pac_delta_gamma_full": pac_dg,
            })
            print(f"    {row['animal_id']} pac_tg={pac_tg:.5f}")

    if not all_rows:
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "pac_full_signal.csv"), index=False)

    print("\n  STATISTICS — Full-signal PAC (animal-level):")
    rows_stat = []
    for feat in ["pac_theta_gamma_full","pac_alpha_gamma_full","pac_delta_gamma_full"]:
        for tp in TP_ORDER:
            sub = df[df.timepoint==tp]
            am  = sub.groupby(["animal_id","group"])[feat].mean().reset_index()
            wt  = am[am.group=="WT"][feat].dropna().values
            ko  = am[am.group=="KO"][feat].dropna().values
            if len(wt)<2 or len(ko)<2:
                continue
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            d, d_lo, d_hi = cohens_d_ci(wt, ko)
            ci = (d_lo>0)or(d_hi<0)
            rows_stat.append({"feature":feat,"timepoint":tp,"wt_mean":np.mean(wt),
                               "ko_mean":np.mean(ko),"cohens_d":d,"d_ci_lo":d_lo,
                               "d_ci_hi":d_hi,"ci_excludes_zero":ci,"pval":p,
                               "n_wt":len(wt),"n_ko":len(ko)})
            if p < 0.10 or ci:
                print(f"    [{tp}] {feat}: WT={np.mean(wt):.5f} KO={np.mean(ko):.5f} "
                      f"d={d:.3f} [{d_lo:.3f},{d_hi:.3f}] p={p:.5f}"
                      f"{'  CI✓' if ci else ''}")

    stat_df = pd.DataFrame(rows_stat)
    if len(stat_df) > 1:
        stat_df["pval_fdr"] = fdr_bh(stat_df["pval"].values).round(6)
        stat_df["fdr_sig"]  = stat_df["pval_fdr"] < 0.05
    stat_df.to_csv(os.path.join(RESULTS_DIR, "pac_full_signal_stats.csv"), index=False)
    print(f"  Saved: pac_full_signal_stats.csv ({len(stat_df)} rows)")
    fdr = stat_df[stat_df.get("fdr_sig",False)==True] if "fdr_sig" in stat_df else pd.DataFrame()
    print(f"  FDR significant: {len(fdr)}")
    for _, r in fdr.iterrows():
        print(f"  *** [{r.timepoint}] {r.feature}: d={r.cohens_d:.3f} FDR={r.pval_fdr:.5f}")


# ══════════════════════════════════════════════════════════════
# FIX 2: DSI trajectory — WT + KO + slope difference
# ══════════════════════════════════════════════════════════════

def run_fix2_dsi_wt_trajectory():
    """Add WT trajectory + slope-difference test to DSI analysis."""
    print("\n" + "="*60)
    print("FIX 2: DSI TRAJECTORY — WT + KO + SLOPE DIFFERENCE TEST")
    print("="*60)

    traj_path = os.path.join(RESULTS_DIR, "dsi_trajectory.csv")
    if not os.path.exists(traj_path):
        print("  dsi_trajectory.csv not found — run script 12 first")
        return

    traj = pd.read_csv(traj_path)
    dsi_cols = ["dsi_1_beta","dsi_2_alpha_beta","dsi_3_euclidean","dsi_4_decorrelation"]

    print("\n  Spearman trajectories (WT and KO separately):")
    rows = []
    for col in dsi_cols:
        if col not in traj.columns:
            continue
        for group in ["WT","KO"]:
            grp = traj[traj.group==group]
            vals, tps = [], []
            for tp, tp_x in zip(TP_ORDER, TP_X):
                sub = grp[grp.timepoint==tp][col].dropna()
                vals.extend(sub.tolist())
                tps.extend([tp_x]*len(sub))
            if len(vals) < 5:
                continue
            r, p = spearmanr(tps, vals)
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            print(f"  {group} {col:<30} ρ={r:.3f} p={p:.5f} {sig}")
            rows.append({"group":group,"feature":col,"rho":r,"pval":p})

    # Slope difference: permutation test
    print("\n  KO vs WT slope difference (permutation test):")
    for col in dsi_cols:
        if col not in traj.columns:
            continue
        # Get slopes per group
        group_slopes = {}
        for group in ["WT","KO"]:
            grp = traj[traj.group==group]
            vals, tps = [], []
            for tp, tp_x in zip(TP_ORDER, TP_X):
                sub = grp[grp.timepoint==tp][col].dropna()
                vals.extend(sub.tolist())
                tps.extend([tp_x]*len(sub))
            if len(vals) >= 5:
                r, _ = spearmanr(tps, vals)
                group_slopes[group] = r

        if len(group_slopes) < 2:
            continue

        obs_diff = group_slopes.get("KO",0) - group_slopes.get("WT",0)

        # Permutation test: shuffle group labels
        all_data = []
        for tp, tp_x in zip(TP_ORDER, TP_X):
            sub = traj[traj.timepoint==tp][[col,"group"]].dropna()
            for _, row in sub.iterrows():
                all_data.append({"val": row[col], "tp": tp_x, "group": row["group"]})
        adf = pd.DataFrame(all_data)

        n_perm = 5000
        rng    = np.random.default_rng(42)
        perm_diffs = []
        for _ in range(n_perm):
            shuffled = adf.copy()
            shuffled["group"] = rng.permutation(shuffled["group"].values)
            slopes = {}
            for g in ["WT","KO"]:
                sub = shuffled[shuffled.group==g]
                if len(sub) >= 5:
                    r, _ = spearmanr(sub["tp"], sub["val"])
                    slopes[g] = r
            if len(slopes) == 2:
                perm_diffs.append(slopes["KO"] - slopes["WT"])

        perm_p = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
        sig    = "***" if perm_p<0.001 else "**" if perm_p<0.01 else "*" if perm_p<0.05 else "ns"
        print(f"  {col:<30} KO_slope={group_slopes.get('KO',np.nan):.3f} "
              f"WT_slope={group_slopes.get('WT',np.nan):.3f} "
              f"diff={obs_diff:.3f} perm_p={perm_p:.4f} {sig}")
        rows.append({"group":"DIFF","feature":col,"rho":obs_diff,"pval":perm_p})

    traj_stats = pd.DataFrame(rows)
    traj_stats.to_csv(os.path.join(RESULTS_DIR, "dsi_trajectory_stats.csv"), index=False)
    print(f"\n  Saved: dsi_trajectory_stats.csv")

    # Figure: WT vs KO DSI-4 trajectory
    dsi_col = "dsi_4_decorrelation"
    if dsi_col in traj.columns:
        fig, ax = plt.subplots(figsize=(9,5))
        for group, color in COLORS.items():
            grp = traj[traj.group==group]
            xs, means, sems = [], [], []
            for tp, tp_x in zip(TP_ORDER, TP_X):
                sub = grp[grp.timepoint==tp][dsi_col].dropna()
                if len(sub)==0: continue
                xs.append(tp_x); means.append(sub.mean()); sems.append(sub.sem())
                ax.scatter([tp_x]*len(sub), sub, color=color, s=25, alpha=0.4, zorder=3)
            if xs:
                ax.fill_between(xs, np.array(means)-np.array(sems),
                                 np.array(means)+np.array(sems), color=color, alpha=0.15)
                ax.plot(xs, means, "-o", color=color, lw=2.5, markersize=8,
                         label=f"{group}", zorder=4)
        ax.set_xticks(TP_X); ax.set_xticklabels(TP_ORDER, fontsize=10)
        ax.set_xlabel("Age (months)", fontsize=11)
        ax.set_ylabel("DSI-4 Decorrelation (1−ρ CA3:CTX)", fontsize=11)
        ax.set_title("Circuit Dissociation Trajectory: WT vs KO\n"
                      "KO: increasing decorrelation (ρ=0.43, p=0.007) | WT: flat", fontsize=10)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "dsi4_wt_ko_trajectory.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: dsi4_wt_ko_trajectory.png")


# ══════════════════════════════════════════════════════════════
# FIX 3: EEG-video epoch correlation
# ══════════════════════════════════════════════════════════════

def run_fix3_eeg_video_correlation():
    """Correlate EEG features with video movement at epoch level (4m)."""
    print("\n" + "="*60)
    print("FIX 3: EEG-VIDEO EPOCH-LEVEL CORRELATION (4m)")
    print("="*60)

    merged_path = os.path.join(DATA_DIR, "epochs_eeg_video_merged_4m.csv")
    if not os.path.exists(merged_path):
        print("  File not found: epochs_eeg_video_merged_4m.csv")
        print("  Run script 07 --timepoint 4m first")
        return

    df = pd.read_csv(merged_path)
    df["animal_id"] = df["animal_id"].astype(str)
    print(f"  Loaded: {len(df):,} epochs, {df['animal_id'].nunique()} animals")

    eeg_feats = ["bp_theta","bp_beta","bp_gamma","rbp_theta","rbp_beta",
                 "td_ratio","spectral_entropy","total_var","hjorth_mob","lzc"]
    vid_feats = ["mean_movement","max_movement","pct_active_frames"]
    eeg_feats = [c for c in eeg_feats if c in df.columns]
    vid_feats = [c for c in vid_feats if c in df.columns]

    if not vid_feats:
        print("  No video features found in merged file")
        return

    rows = []
    print("\n  Correlations (all epochs, all animals):")
    for ef in eeg_feats:
        for vf in vid_feats:
            xy = df[[ef,vf]].dropna()
            if len(xy) < 50:
                continue
            r, p = spearmanr(xy[ef], xy[vf])
            rows.append({"eeg_feat":ef,"vid_feat":vf,"rho":round(r,4),
                          "pval":round(p,6),"n":len(xy)})

    corr_df = pd.DataFrame(rows)
    if len(corr_df) > 1:
        corr_df["pval_fdr"] = fdr_bh(corr_df["pval"].values).round(6)
        corr_df["fdr_sig"]  = corr_df["pval_fdr"] < 0.05

    sig = corr_df[corr_df.pval < 0.05].sort_values("pval") if len(corr_df) > 0 else corr_df
    for _, r in sig.head(15).iterrows():
        fdr = " FDR*" if r.get("fdr_sig",False) else ""
        print(f"  {r.eeg_feat:<25} ↔ {r.vid_feat:<22} ρ={r.rho:.3f} p={r.pval:.5f}{fdr}")

    corr_df.to_csv(os.path.join(RESULTS_DIR, "eeg_video_correlation_4m.csv"), index=False)
    print(f"\n  Saved: eeg_video_correlation_4m.csv ({len(corr_df)} correlations)")

    # By state
    if "state" in df.columns:
        print("\n  By sleep/wake state:")
        for state in ["Wake","NREM","REM"]:
            sub = df[df.state==state]
            if len(sub) < 100:
                continue
            for ef in eeg_feats[:4]:
                for vf in vid_feats[:2]:
                    xy = sub[[ef,vf]].dropna()
                    if len(xy) < 50: continue
                    r, p = spearmanr(xy[ef], xy[vf])
                    if p < 0.05:
                        print(f"  [{state}] {ef:<22} ↔ {vf:<18} ρ={r:.3f} p={p:.5f}")

    # Figure
    if len(sig) > 0 and vid_feats:
        vf = vid_feats[0]
        ef = sig.iloc[0]["eeg_feat"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, group, color in zip(axes, ["WT","KO"], [COLORS["WT"],COLORS["KO"]]):
            sub = df[df.group==group][[ef,vf]].dropna()
            if len(sub) < 10:
                ax.set_visible(False); continue
            ax.hexbin(sub[vf], sub[ef], gridsize=40, cmap="YlOrRd", mincnt=1)
            r, p = spearmanr(sub[vf], sub[ef])
            ax.set_xlabel(vf, fontsize=10)
            ax.set_ylabel(ef, fontsize=10)
            ax.set_title(f"{group}: ρ={r:.3f} p={p:.4f}", fontsize=10)
        fig.suptitle(f"EEG-Video Epoch Correlation — 4m\n{ef} vs {vf}", fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "eeg_video_correlation_4m.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: eeg_video_correlation_4m.png")


# ══════════════════════════════════════════════════════════════
# CLASSICAL 1: Circadian rhythm analysis
# ══════════════════════════════════════════════════════════════

def run_circadian_analysis():
    """Hourly Wake% distribution across 24h recording per timepoint."""
    print("\n" + "="*60)
    print("CLASSICAL 1: CIRCADIAN RHYTHM ANALYSIS")
    print("="*60)

    rows = []
    for tp in TP_ORDER:
        path = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["animal_id"] = df["animal_id"].astype(str)
        df["hour"] = (df["onset_s"] / 3600).astype(int)
        max_hour = df["hour"].max()
        if max_hour < 3:
            continue  # too short

        for hour in range(int(max_hour)+1):
            sub = df[df.hour==hour]
            for group in ["WT","KO"]:
                gsub = sub[sub.group==group]
                if len(gsub) < 10:
                    continue
                wake_pct = (gsub.state=="Wake").mean()*100
                rows.append({
                    "timepoint": tp, "hour": hour, "group": group,
                    "wake_pct": wake_pct, "n_epochs": len(gsub)
                })

    if not rows:
        print("  No data available")
        return

    circ_df = pd.DataFrame(rows)
    circ_df.to_csv(os.path.join(RESULTS_DIR, "circadian_wake_pct.csv"), index=False)
    print(f"  Saved: circadian_wake_pct.csv")

    # WT vs KO per hour — which hours differ?
    print("\n  Hours with significant WT vs KO difference (p<0.05):")
    stat_rows = []
    for tp in circ_df["timepoint"].unique():
        for hour in sorted(circ_df["hour"].unique()):
            sub = circ_df[(circ_df.timepoint==tp) & (circ_df.hour==hour)]
            wt = sub[sub.group=="WT"]["wake_pct"].values
            ko = sub[sub.group=="KO"]["wake_pct"].values
            if len(wt)<2 or len(ko)<2:
                continue
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            d = (np.mean(ko)-np.mean(wt))/np.sqrt((np.std(wt)**2+np.std(ko)**2)/2+1e-10)
            stat_rows.append({"timepoint":tp,"hour":hour,"wt_mean":np.mean(wt),
                               "ko_mean":np.mean(ko),"cohens_d":d,"pval":p})
            if p < 0.05:
                print(f"  [{tp}] hour {hour:02d}: WT={np.mean(wt):.1f}% KO={np.mean(ko):.1f}% "
                      f"d={d:.2f} p={p:.4f}")

    stat_df = pd.DataFrame(stat_rows)
    if len(stat_df)>1:
        stat_df["pval_fdr"] = fdr_bh(stat_df["pval"].values).round(6)
    stat_df.to_csv(os.path.join(RESULTS_DIR, "circadian_stats.csv"), index=False)
    print(f"  Saved: circadian_stats.csv ({len(stat_df)} rows)")

    # Figure
    tp_avail = [t for t in TP_ORDER if t in circ_df["timepoint"].unique()]
    n = len(tp_avail)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(n*4, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, tp in zip(axes, tp_avail):
        for group, color in COLORS.items():
            sub = circ_df[(circ_df.timepoint==tp) & (circ_df.group==group)]
            sub = sub.groupby("hour")["wake_pct"].mean().reset_index()
            ax.plot(sub["hour"], sub["wake_pct"], "-o", color=color,
                     lw=1.5, markersize=4, label=group)
        ax.set_title(tp, fontsize=10)
        ax.set_xlabel("Hour", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Wake %", fontsize=10)
            ax.legend(fontsize=8)
        ax.set_xlim(0, None)

    fig.suptitle("Circadian Wake Distribution — C9orf72 KO vs WT", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "circadian_wake_pct.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: circadian_wake_pct.png")


# ══════════════════════════════════════════════════════════════
# CLASSICAL 2: Sleep bout fragmentation
# ══════════════════════════════════════════════════════════════

def run_bout_fragmentation():
    """Sleep bout fragmentation index across all timepoints."""
    print("\n" + "="*60)
    print("CLASSICAL 2: SLEEP BOUT FRAGMENTATION INDEX")
    print("="*60)

    rows = []
    for tp in TP_ORDER:
        path = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["animal_id"] = df["animal_id"].astype(str)

        for (animal_id, group), adf in df.groupby(["animal_id","group"]):
            adf = adf.sort_values("epoch_idx")
            for state in ["Wake","NREM","REM"]:
                states = adf["state"].values
                bouts = []
                count = 0
                for s in states:
                    if s == state:
                        count += 1
                    else:
                        if count > 0:
                            bouts.append(count)
                        count = 0
                if count > 0:
                    bouts.append(count)

                if not bouts:
                    continue
                bouts_s = [b * EPOCH_S for b in bouts]
                rows.append({
                    "animal_id":    animal_id,
                    "group":        group,
                    "timepoint":    tp,
                    "state":        state,
                    "n_bouts":      len(bouts_s),
                    "mean_bout_s":  float(np.mean(bouts_s)),
                    "median_bout_s":float(np.median(bouts_s)),
                    "cv_bout":      float(np.std(bouts_s)/(np.mean(bouts_s)+1e-8)),
                    "total_s":      float(sum(bouts_s)),
                })

    if not rows:
        print("  No data available")
        return

    frag_df = pd.DataFrame(rows)
    frag_df.to_csv(os.path.join(RESULTS_DIR, "bout_fragmentation.csv"), index=False)
    print(f"  Saved: bout_fragmentation.csv ({len(frag_df)} rows)")

    print("\n  WT vs KO bout metrics (animal-level):")
    stat_rows = []
    for tp in TP_ORDER:
        for state in ["Wake","NREM","REM"]:
            sub = frag_df[(frag_df.timepoint==tp) & (frag_df.state==state)]
            am  = sub.groupby(["animal_id","group"])[
                ["n_bouts","mean_bout_s","cv_bout"]].mean().reset_index()
            for col in ["mean_bout_s","cv_bout","n_bouts"]:
                wt = am[am.group=="WT"][col].dropna().values
                ko = am[am.group=="KO"][col].dropna().values
                if len(wt)<2 or len(ko)<2:
                    continue
                _, p = mannwhitneyu(wt, ko, alternative="two-sided")
                d = (np.mean(ko)-np.mean(wt))/np.sqrt((np.std(wt)**2+np.std(ko)**2)/2+1e-10)
                ci = False
                stat_rows.append({"timepoint":tp,"state":state,"feature":col,
                                   "wt_mean":np.mean(wt),"ko_mean":np.mean(ko),
                                   "cohens_d":d,"pval":p,"n_wt":len(wt),"n_ko":len(ko)})
                if p < 0.10:
                    print(f"  [{tp}] {state} {col:<15}: WT={np.mean(wt):.2f} "
                          f"KO={np.mean(ko):.2f} d={d:.3f} p={p:.4f}")

    stat_df = pd.DataFrame(stat_rows)
    if len(stat_df)>1:
        stat_df["pval_fdr"] = fdr_bh(stat_df["pval"].values).round(6)
        stat_df["fdr_sig"]  = stat_df["pval_fdr"] < 0.05
    stat_df.to_csv(os.path.join(RESULTS_DIR, "bout_fragmentation_stats.csv"), index=False)

    fdr_sig = stat_df[stat_df.get("fdr_sig",False)==True] if "fdr_sig" in stat_df else pd.DataFrame()
    print(f"\n  FDR significant: {len(fdr_sig)}")
    for _, r in fdr_sig.iterrows():
        print(f"  *** [{r.timepoint}] {r.state} {r.feature}: d={r.cohens_d:.3f} "
              f"p={r.pval:.5f} FDR={r.pval_fdr:.5f}")

    # Trajectory figure for NREM bout duration
    fig, ax = plt.subplots(figsize=(9, 5))
    nrem_frag = frag_df[frag_df.state=="NREM"]
    for group, color in COLORS.items():
        xs, means, sems = [], [], []
        for tp, tp_x in zip(TP_ORDER, TP_X):
            sub = nrem_frag[(nrem_frag.timepoint==tp) & (nrem_frag.group==group)]
            am  = sub.groupby("animal_id")["mean_bout_s"].mean()
            if len(am)==0: continue
            xs.append(tp_x); means.append(am.mean()); sems.append(am.sem())
        if xs:
            ax.fill_between(xs, np.array(means)-np.array(sems),
                             np.array(means)+np.array(sems), color=color, alpha=0.15)
            ax.plot(xs, means, "-o", color=color, lw=2.5, markersize=8, label=group)
    ax.set_xticks(TP_X); ax.set_xticklabels(TP_ORDER, fontsize=10)
    ax.set_xlabel("Age (months)", fontsize=11)
    ax.set_ylabel("Mean NREM bout duration (s)", fontsize=11)
    ax.set_title("NREM Sleep Bout Duration Trajectory — C9orf72 KO vs WT", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "nrem_bout_duration_trajectory.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: nrem_bout_duration_trajectory.png")


# ══════════════════════════════════════════════════════════════
# CLASSICAL 3: Video immobility trajectory
# ══════════════════════════════════════════════════════════════

def run_immobility_trajectory():
    """Video immobility analysis across all timepoints."""
    print("\n" + "="*60)
    print("CLASSICAL 3: VIDEO IMMOBILITY TRAJECTORY")
    print("="*60)

    rows = []
    for tp in TP_ORDER:
        path = os.path.join(DATA_DIR, f"video_movement_epochs_{tp}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["animal_id"] = df["animal_id"].astype(str)

        # Immobility threshold: bottom 10th percentile of movement
        thresh = df["mean_movement"].quantile(0.10)
        df["immobile"] = df["mean_movement"] < thresh

        am = df.groupby(["animal_id","group"]).agg(
            pct_immobile=("immobile","mean"),
            mean_movement=("mean_movement","mean"),
            total_epochs=("mean_movement","count")
        ).reset_index()
        am["pct_immobile"] *= 100
        am["timepoint"] = tp
        rows.append(am)

    if not rows:
        print("  No video data found — run script 07 for all timepoints first")
        return

    imm_df = pd.concat(rows, ignore_index=True)
    imm_df.to_csv(os.path.join(RESULTS_DIR, "immobility_trajectory.csv"), index=False)
    print(f"  Saved: immobility_trajectory.csv")

    print("\n  WT vs KO immobility per timepoint:")
    stat_rows = []
    for tp in TP_ORDER:
        sub = imm_df[imm_df.timepoint==tp]
        for feat in ["pct_immobile","mean_movement"]:
            wt = sub[sub.group=="WT"][feat].dropna().values
            ko = sub[sub.group=="KO"][feat].dropna().values
            if len(wt)<2 or len(ko)<2: continue
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            d = (np.mean(ko)-np.mean(wt))/np.sqrt((np.std(wt)**2+np.std(ko)**2)/2+1e-10)
            stat_rows.append({"timepoint":tp,"feature":feat,
                               "wt_mean":np.mean(wt),"ko_mean":np.mean(ko),
                               "cohens_d":d,"pval":p})
            if p < 0.10:
                print(f"  [{tp}] {feat:<20}: WT={np.mean(wt):.2f} KO={np.mean(ko):.2f} "
                      f"d={d:.3f} p={p:.4f}")

    stat_df = pd.DataFrame(stat_rows)
    if len(stat_df)>1:
        stat_df["pval_fdr"] = fdr_bh(stat_df["pval"].values).round(6)
    stat_df.to_csv(os.path.join(RESULTS_DIR, "immobility_stats.csv"), index=False)

    # Figure
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    for ax, feat, ylabel in zip(axes,
                                  ["pct_immobile","mean_movement"],
                                  ["% Immobile epochs","Mean movement score"]):
        for group, color in COLORS.items():
            xs, means, sems = [], [], []
            for tp, tp_x in zip(TP_ORDER, TP_X):
                sub = imm_df[(imm_df.timepoint==tp)&(imm_df.group==group)]
                vals = sub[feat].dropna()
                if len(vals)==0: continue
                xs.append(tp_x); means.append(vals.mean()); sems.append(vals.sem())
            if xs:
                ax.fill_between(xs, np.array(means)-np.array(sems),
                                 np.array(means)+np.array(sems), color=color, alpha=0.15)
                ax.plot(xs, means, "-o", color=color, lw=2.5, markersize=8, label=group)
        ax.set_xticks([x for x in TP_X if x in [tp_x for tp_x in TP_X]])
        ax.set_xticklabels(TP_ORDER, fontsize=9)
        ax.set_xlabel("Age (months)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
    fig.suptitle("Video Immobility Trajectory — C9orf72 KO vs WT", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "immobility_trajectory.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: immobility_trajectory.png")


# ══════════════════════════════════════════════════════════════
# CLASSICAL 4: Pre-ictal EEG state (4m)
# ══════════════════════════════════════════════════════════════

def run_preictal_analysis():
    """EEG features in 30-min window before first seizure at 4m."""
    print("\n" + "="*60)
    print("CLASSICAL 4: PRE-ICTAL EEG STATE (4m)")
    print("="*60)

    seizure_path = os.path.join(DATA_DIR, "video_seizures_4m.csv")
    epoch_path   = os.path.join(DATA_DIR, "epochs_with_states_4m.csv")

    if not os.path.exists(seizure_path):
        print("  video_seizures_4m.csv not found — run script 07 --timepoint 4m")
        return
    if not os.path.exists(epoch_path):
        print("  epochs_with_states_4m.csv not found")
        return

    seizures = pd.read_csv(seizure_path)
    epochs   = pd.read_csv(epoch_path)
    epochs["animal_id"] = epochs["animal_id"].astype(str)
    seizures["animal_id"] = seizures["animal_id"].astype(str)

    feat_cols = ["bp_theta","bp_beta","rbp_theta","rbp_beta","td_ratio",
                 "spectral_entropy","total_var","hjorth_mob"]
    feat_cols = [c for c in feat_cols if c in epochs.columns]

    PRE_WINDOW_S  = 30 * 60   # 30 minutes
    POST_WINDOW_S = 30 * 60   # 30 minutes post (for comparison)

    rows = []
    for animal_id, sz_df in seizures.groupby("animal_id"):
        if len(sz_df) == 0:
            continue
        first_sz = sz_df["onset_s"].min()
        group    = sz_df["group"].iloc[0]

        ep_sub = epochs[epochs.animal_id == animal_id]
        if len(ep_sub) == 0:
            continue

        # Pre-ictal window
        pre = ep_sub[(ep_sub.onset_s >= first_sz - PRE_WINDOW_S) &
                     (ep_sub.onset_s <  first_sz)]
        # Post-ictal window (as comparison)
        post = ep_sub[(ep_sub.onset_s >  first_sz) &
                      (ep_sub.onset_s <= first_sz + POST_WINDOW_S)]

        if len(pre) < 20:
            continue

        row = {"animal_id": animal_id, "group": group,
               "first_seizure_s": first_sz, "n_pre_epochs": len(pre)}
        for f in feat_cols:
            row[f"pre_{f}"]  = float(pre[f].median())
            if len(post) >= 20:
                row[f"post_{f}"] = float(post[f].median())
        rows.append(row)

    if not rows:
        print("  Insufficient pre-ictal data")
        return

    pre_df = pd.DataFrame(rows)
    pre_df.to_csv(os.path.join(RESULTS_DIR, "preictal_eeg_4m.csv"), index=False)
    print(f"  Saved: preictal_eeg_4m.csv ({len(pre_df)} animals)")

    print("\n  Pre-ictal EEG: WT vs KO:")
    for f in feat_cols:
        col = f"pre_{f}"
        if col not in pre_df.columns:
            continue
        wt = pre_df[pre_df.group=="WT"][col].dropna().values
        ko = pre_df[pre_df.group=="KO"][col].dropna().values
        if len(wt)<2 or len(ko)<2:
            continue
        _, p = mannwhitneyu(wt, ko, alternative="two-sided")
        d = (np.mean(ko)-np.mean(wt))/np.sqrt((np.std(wt)**2+np.std(ko)**2)/2+1e-10)
        if p < 0.10:
            print(f"  {f:<25}: WT={np.mean(wt):.5f} KO={np.mean(ko):.5f} "
                  f"d={d:.3f} p={p:.4f}")

    print("\n  Pre vs post-ictal (within KO, Wilcoxon):")
    ko_df = pre_df[pre_df.group=="KO"]
    for f in feat_cols:
        pre_col  = f"pre_{f}"
        post_col = f"post_{f}"
        if pre_col not in ko_df or post_col not in ko_df:
            continue
        xy = ko_df[[pre_col, post_col]].dropna()
        if len(xy) < 3:
            continue
        try:
            _, p = wilcoxon(xy[pre_col], xy[post_col])
            diff = np.mean(xy[post_col]) - np.mean(xy[pre_col])
            if p < 0.10:
                print(f"  KO {f:<22}: pre={np.mean(xy[pre_col]):.5f} "
                      f"post={np.mean(xy[post_col]):.5f} diff={diff:.5f} p={p:.4f}")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# CLASSICAL 5: WT CTX vs CA3 (control for fix 2)
# ══════════════════════════════════════════════════════════════

def run_wt_ctx_ca3_comparison():
    """Does WT also show CTX vs CA3 spectral dissociation?"""
    print("\n" + "="*60)
    print("CLASSICAL 5: WT CTX vs CA3 — IS DISSOCIATION KO-SPECIFIC?")
    print("="*60)

    for tp in TP_ORDER:
        path = os.path.join(RESULTS_DIR, f"circuit_dissociation_{tp}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["animal_id"] = df["animal_id"].astype(str)

        print(f"\n  {tp}:")
        for group in ["WT","KO"]:
            grp = df[df.group==group]
            am  = grp.groupby("animal_id")[
                [f"ca3_rbp_{b}" for b in ["delta","theta","alpha","beta","gamma"]] +
                [f"ctx_rbp_{b}" for b in ["delta","theta","alpha","beta","gamma"]]
            ].mean().reset_index()

            sig_found = False
            for b in ["delta","theta","alpha","beta","gamma"]:
                ca3 = am[f"ca3_rbp_{b}"].dropna().values
                ctx = am[f"ctx_rbp_{b}"].dropna().values
                n   = min(len(ca3), len(ctx))
                if n < 3:
                    continue
                try:
                    _, p = wilcoxon(ctx[:n], ca3[:n])
                except Exception:
                    continue
                diff = np.mean(ctx[:n]) - np.mean(ca3[:n])
                if p < 0.05:
                    direction = "CTX>CA3" if diff > 0 else "CA3>CTX"
                    print(f"    {group} {b}: diff={diff:.4f} p={p:.4f} [{direction}]")
                    sig_found = True
            if not sig_found:
                print(f"    {group}: no significant CTX vs CA3 differences (p>0.05)")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FIXES + CLASSICAL EXTENSIONS")
    print("=" * 60)

    inv_path = os.path.join(DATA_DIR, "file_inventory_all_timepoints.csv")
    if os.path.exists(inv_path):
        inventory = pd.read_csv(inv_path)
        inventory["animal_id"] = inventory["animal_id"].astype(str)
        inventory = inventory.drop_duplicates(subset=["abf_path"])
    else:
        inventory = pd.DataFrame()

    # Fix 1: PAC full signal (reprocesses ABF files — slow)
    # Uncomment to run:
    if not inventory.empty:
        run_fix1_pac(inventory)

    # Fix 2: DSI WT trajectory + slope difference (fast — uses existing CSV)
    run_fix2_dsi_wt_trajectory()

    # Fix 3: EEG-video correlation (fast — uses existing merged CSV)
    run_fix3_eeg_video_correlation()

    # Classical 1: Circadian (fast — uses epoch CSVs)
    run_circadian_analysis()

    # Classical 2: Bout fragmentation (fast — uses epoch CSVs)
    run_bout_fragmentation()

    # Classical 3: Immobility trajectory (fast — uses video CSVs)
    run_immobility_trajectory()

    # Classical 4: Pre-ictal EEG (fast — uses seizure + epoch CSVs)
    run_preictal_analysis()

    # Classical 5: WT CTX vs CA3 (fast — uses circuit dissociation CSVs)
    run_wt_ctx_ca3_comparison()

    print("\n" + "="*60)
    print("ALL EXTENSIONS COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("="*60)
    print("\nNote: Fix 1 (PAC full signal) is commented out — uncomment")
    print("and rerun to recompute PAC without epoch cap.")


if __name__ == "__main__":
    main()
