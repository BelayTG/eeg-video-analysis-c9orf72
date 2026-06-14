"""
11_portfolio_integration_analysis.py
=====================================
Integrates analyses from all previous C9orf72 EEG portfolios into
the current 6-timepoint framework (3m, 4m, 6m, 7m, 9m, 12m).

NEW ANALYSES from previous portfolios:
  1. FOOOF 1/f aperiodic exponent (E/I balance biomarker)
     Source: eeg-network-vulnerability-c9orf72
     Previous finding: 12m alpha p=0.008 d=0.87; 3m predicts 12m r=0.503 p=0.004

  2. Permutation entropy (PeEn) — signal complexity
     Source: vulnerability paper + six-portfolio synthesis
     Previous finding: 3m p=0.008 d=0.73 (latent vulnerability)

  3. Lempel-Ziv complexity (LZC) trajectory
     Source: vulnerability paper
     Previous finding: 4m p=0.002 (complexity collapse)

  4. Phase-amplitude coupling (PAC) trajectory
     Source: six-portfolio synthesis
     Previous finding: 12m alpha-gamma PAC FDR p=0.0004

  5. Recovery index (unique to 6-timepoint design)
     = (value at 9m or 12m) / (value at 3m baseline)
     Tests whether KO networks recover to baseline after KA insult

  6. Predictive modeling: 3m features → 9m/12m outcome
     Source: vulnerability paper
     Previous finding: r=0.601 p=0.0004 for gamma

  7. wPLI connectivity trajectory (CA3-CTX)
     Source: vulnerability paper
     More robust than coherence (rejects volume conduction)

  8. Linear mixed effects model: Genotype × Time interaction
     Across all 6 timepoints for primary features

Statistical approach:
  - All analyses exploratory (not pre-specified)
  - Report uncorrected p-values + Cohen's d + 95% bootstrap CI
  - Flag findings where CI excludes zero
  - Cross-reference with previous portfolio findings

Outputs:
  - results/portfolio_integration_{tp}.csv
  - results/recovery_index.csv
  - results/predictive_3m_to_outcome.csv
  - results/genotype_time_interaction.csv
  - figures/portfolio_*.png

Run:
    python src/11_portfolio_integration_analysis.py
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import pyabf
from scipy.signal import welch, butter, decimate, hilbert
from scipy.signal import sosfiltfilt
from scipy.stats import mannwhitneyu, spearmanr, permutation_test
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


# ── Helpers ────────────────────────────────────────────────────────────────

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
        boot.append((np.mean(by)-np.mean(bx)) / np.sqrt((np.std(bx)**2+np.std(by)**2)/2+1e-10))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def load_signal(abf_path, channel=0, target_fs=FS_DS):
    try:
        abf = pyabf.ABF(abf_path)
        fs  = float(abf.dataRate)
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


def bandpass(sig, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


# ── 1. FOOOF proxy — aperiodic exponent ───────────────────────────────────

def aperiodic_exponent_robust(sig, fs, lo=2, hi=40, nperseg_s=4):
    """
    Robust 1/f aperiodic exponent using median of epoch-level estimates.
    More stable than single-epoch estimates.
    """
    epoch_n  = int(EPOCH_S * fs)
    n_epochs = len(sig) // epoch_n
    exps = []
    for i in range(n_epochs):
        ep = sig[i*epoch_n:(i+1)*epoch_n]
        if np.abs(ep).max() > 500 or np.std(ep) < 0.001:
            continue
        f, psd = welch(ep, fs=fs, nperseg=min(int(fs*2), epoch_n))
        m = (f >= lo) & (f <= hi) & (psd > 0)
        if m.sum() < 5:
            continue
        slope, _ = np.polyfit(np.log10(f[m]), np.log10(psd[m]), 1)
        exps.append(-slope)
    return float(np.median(exps)) if exps else np.nan


# ── 2. Permutation entropy ─────────────────────────────────────────────────

def permutation_entropy(sig, m=3, delay=1):
    """
    Ordinal permutation entropy (Bandt & Pompe 2002).
    Normalized to [0,1]: higher = more complex/random.
    """
    n = len(sig)
    if n < m * delay:
        return np.nan
    # Generate all ordinal patterns
    from itertools import permutations
    perms = list(permutations(range(m)))
    perm_idx = {p: i for i, p in enumerate(perms)}
    counts = np.zeros(len(perms))
    for i in range(n - (m-1)*delay):
        pattern = tuple(np.argsort(sig[i:i+m*delay:delay]))
        if pattern in perm_idx:
            counts[perm_idx[pattern]] += 1
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)) / np.log2(len(perms)))


def peen_epochs(sig, fs, epoch_s=EPOCH_S, m=3):
    """Compute permutation entropy for each epoch, return median."""
    epoch_n  = int(epoch_s * fs)
    n_epochs = len(sig) // epoch_n
    vals = []
    for i in range(n_epochs):
        ep = sig[i*epoch_n:(i+1)*epoch_n]
        if np.abs(ep).max() > 500 or np.std(ep) < 0.001:
            continue
        # Downsample for speed
        ep_ds = ep[::4]
        pe = permutation_entropy(ep_ds, m=m)
        if not np.isnan(pe):
            vals.append(pe)
    return float(np.median(vals)) if vals else np.nan


# ── 3. Phase-amplitude coupling (modulation index) ────────────────────────

def pac_modulation_index(sig, fs, phase_band=(4,8), amp_band=(30,80),
                          n_bins=18, n_epochs_max=120):
    """
    Theta-gamma PAC using Modulation Index (Tort et al. 2010).
    Uses first n_epochs_max epochs for speed.
    """
    epoch_n = int(EPOCH_S * fs)
    n_epochs = min(len(sig) // epoch_n, n_epochs_max)
    mi_vals = []
    for i in range(n_epochs):
        ep = sig[i*epoch_n:(i+1)*epoch_n]
        if np.abs(ep).max() > 500 or np.std(ep) < 0.001:
            continue
        phase_sig = bandpass(ep, fs, phase_band[0], phase_band[1])
        amp_sig   = bandpass(ep, fs, amp_band[0],   amp_band[1])
        phase = np.angle(hilbert(phase_sig))
        amp   = np.abs(hilbert(amp_sig))
        bin_edges = np.linspace(-np.pi, np.pi, n_bins+1)
        amp_dist = np.array([
            amp[(phase >= bin_edges[k]) & (phase < bin_edges[k+1])].mean()
            if np.any((phase >= bin_edges[k]) & (phase < bin_edges[k+1])) else 0
            for k in range(n_bins)
        ])
        amp_dist = np.clip(amp_dist, 1e-12, None)
        amp_dist /= amp_dist.sum()
        mi = np.sum(amp_dist * np.log(amp_dist / (1/n_bins) + 1e-12)) / np.log(n_bins)
        mi_vals.append(float(mi))
    return float(np.median(mi_vals)) if mi_vals else np.nan


# ── 4. wPLI connectivity ──────────────────────────────────────────────────

def wpli_epoch(sig1, sig2, fs, band=(4,8), epoch_n=None):
    """Weighted Phase Lag Index for one epoch pair."""
    if epoch_n is None:
        epoch_n = len(sig1)
    nperseg = min(int(fs*2), epoch_n)
    from scipy.signal import csd
    f, Pxy = csd(sig1, sig2, fs=fs, nperseg=nperseg)
    lo, hi = band
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return np.nan
    imag = np.imag(Pxy[m])
    wpli = np.abs(np.mean(np.abs(imag) * np.sign(imag))) / np.mean(np.abs(imag) + 1e-12)
    return float(wpli)


def wpli_trajectory(sig1, sig2, fs, bands=None):
    """Compute wPLI per band across epochs."""
    if bands is None:
        bands = {"theta": (4,8), "alpha": (8,13), "beta": (13,30), "gamma": (30,80)}
    epoch_n = int(EPOCH_S * fs)
    n_epochs = min(len(sig1)//epoch_n, len(sig2)//epoch_n, 120)
    results = {b: [] for b in bands}
    for i in range(n_epochs):
        ep1 = sig1[i*epoch_n:(i+1)*epoch_n]
        ep2 = sig2[i*epoch_n:(i+1)*epoch_n]
        if np.abs(ep1).max() > 500 or np.abs(ep2).max() > 500:
            continue
        for band_name, band_range in bands.items():
            w = wpli_epoch(ep1, ep2, fs, band_range, epoch_n)
            if not np.isnan(w):
                results[band_name].append(w)
    return {b: float(np.median(v)) if v else np.nan for b, v in results.items()}


# ── Main per-file computation ──────────────────────────────────────────────

def compute_portfolio_features(abf_path, animal_id, group, tp):
    """Compute all portfolio features for one ABF file."""
    sig_ca3, fs = load_signal(abf_path, channel=0)
    if sig_ca3 is None:
        return None

    result = {
        "animal_id": animal_id,
        "group":     group,
        "timepoint": tp,
        "abf_file":  os.path.basename(abf_path),
    }

    # 1. FOOOF proxy
    result["ap_exp_robust"] = aperiodic_exponent_robust(sig_ca3, fs)

    # 2. Permutation entropy (fast: downsample first)
    result["peen"] = peen_epochs(sig_ca3, fs)

    # 3. PAC — theta-gamma
    n_pac = min(len(sig_ca3), int(30*60*fs))
    result["pac_theta_gamma"] = pac_modulation_index(
        sig_ca3[:n_pac], fs, phase_band=(4,8), amp_band=(30,80))

    # 4. PAC — alpha-gamma (strong finding in previous portfolio)
    result["pac_alpha_gamma"] = pac_modulation_index(
        sig_ca3[:n_pac], fs, phase_band=(8,13), amp_band=(30,80))

    # 5. PAC — delta-gamma
    result["pac_delta_gamma"] = pac_modulation_index(
        sig_ca3[:n_pac], fs, phase_band=(0.5,4), amp_band=(30,80))

    # 6. wPLI connectivity (if CTX channel available)
    sig_ctx, _ = load_signal(abf_path, channel=1)
    if sig_ctx is not None:
        n_wpli = min(len(sig_ca3), len(sig_ctx), int(30*60*fs))
        wpli = wpli_trajectory(sig_ca3[:n_wpli], sig_ctx[:n_wpli], fs)
        for band_name, val in wpli.items():
            result[f"wpli_{band_name}"] = val
        del sig_ctx

    del sig_ca3
    gc.collect()
    return result


# ── Process one timepoint ─────────────────────────────────────────────────

def process_timepoint(tp, inventory):
    print(f"\n{'='*60}")
    print(f"Portfolio Integration — {tp}")
    print(f"{'='*60}")

    inv = inventory[(inventory.timepoint==tp) &
                    (inventory.scenario=="A")].drop_duplicates(
                        subset=["abf_path"]).copy()
    if len(inv) == 0:
        print(f"  No files for {tp}")
        return pd.DataFrame()

    rows = []
    for _, file_row in inv.iterrows():
        abf_path  = file_row["abf_path"]
        animal_id = str(file_row["animal_id"])
        group     = file_row["group"]
        if not os.path.exists(abf_path):
            continue
        print(f"  {animal_id} | {file_row['abf_file'][:25]}", end="", flush=True)
        res = compute_portfolio_features(abf_path, animal_id, group, tp)
        if res:
            rows.append(res)
            print(f" ap={res.get('ap_exp_robust',np.nan):.2f} "
                  f"pe={res.get('peen',np.nan):.3f} "
                  f"pac_tg={res.get('pac_theta_gamma',np.nan):.5f}")
        else:
            print(" SKIP")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f"portfolio_integration_{tp}.csv"),
               index=False)
    print(f"\n  Saved: portfolio_integration_{tp}.csv ({len(df)} rows)")
    return df


# ── Statistics across all timepoints ─────────────────────────────────────

def run_statistics_all(all_dfs):
    """Animal-level Mann-Whitney U for all portfolio features across timepoints."""
    feat_cols = ["ap_exp_robust", "peen", "pac_theta_gamma",
                 "pac_alpha_gamma", "pac_delta_gamma",
                 "wpli_theta", "wpli_alpha", "wpli_beta", "wpli_gamma"]

    rows = []
    print(f"\n{'='*60}")
    print("PORTFOLIO INTEGRATION STATISTICS")
    print(f"{'='*60}")

    for tp in TP_ORDER:
        if tp not in all_dfs or all_dfs[tp].empty:
            continue
        df = all_dfs[tp]
        # Animal-level means
        am = df.groupby(["animal_id","group"])[feat_cols].mean().reset_index()

        for feat in feat_cols:
            if feat not in am.columns:
                continue
            wt = am[am.group=="WT"][feat].dropna().values
            ko = am[am.group=="KO"][feat].dropna().values
            if len(wt) < 2 or len(ko) < 2:
                continue
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            d, d_lo, d_hi = cohens_d_ci(wt, ko)
            ci_excl = (d_lo > 0) or (d_hi < 0)
            rows.append({
                "feature":         feat,
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
                print(f"  [{tp}] {feat:<25} WT={np.mean(wt):.4f} KO={np.mean(ko):.4f} "
                      f"d={d:.3f} [{d_lo:.3f},{d_hi:.3f}] p={p:.5f}"
                      f"{'  CI✓' if ci_excl else ''}")

    stats = pd.DataFrame(rows)
    if len(stats) > 1:
        stats["pval_fdr"] = fdr_bh(stats["pval"].values).round(6)
        stats["fdr_sig"]  = stats["pval_fdr"] < 0.05

    stats.to_csv(os.path.join(RESULTS_DIR, "portfolio_integration_stats.csv"),
                  index=False)
    print(f"\nSaved: portfolio_integration_stats.csv ({len(stats)} rows)")

    fdr_sig = stats[stats.get("fdr_sig", False) == True] if "fdr_sig" in stats else pd.DataFrame()
    ci_sig  = stats[stats.ci_excludes_zero & ~stats.get("fdr_sig", False)] if "fdr_sig" in stats else stats[stats.ci_excludes_zero]

    print(f"\nFDR significant: {len(fdr_sig)}")
    print(f"CI excludes zero: {len(ci_sig) + len(fdr_sig)}")
    if len(fdr_sig) > 0:
        for _, r in fdr_sig.iterrows():
            print(f"  *** [{r.timepoint}] {r.feature}: d={r.cohens_d:.3f} "
                  f"p={r.pval:.5f} FDR={r.pval_fdr:.5f}")

    return stats


# ── Recovery index ────────────────────────────────────────────────────────

def compute_recovery_index(all_dfs):
    """
    Recovery index = value_at_9m (or 12m) / value_at_3m per animal.
    Tests whether KO networks recover to baseline after the KA insult.
    Unique to this 6-timepoint design.
    """
    print(f"\n{'='*60}")
    print("RECOVERY INDEX ANALYSIS")
    print(f"{'='*60}")

    feat_cols = ["ap_exp_robust", "peen", "pac_theta_gamma", "pac_alpha_gamma"]
    # Also include band power from script 05
    bp_path = os.path.join(DATA_DIR, "state_specific_features_3m_A.csv")
    bp_cols = []
    if os.path.exists(bp_path):
        bp_df = pd.read_csv(bp_path)
        bp_cols = ["nrem_rbp_theta", "nrem_td_ratio", "wake_rbp_beta",
                   "wake_spectral_entropy", "nrem_lzc", "wake_ap_exp"]
        bp_cols = [c for c in bp_cols if c in bp_df.columns]

    rows = []
    for animal_id in set().union(*[
            set(all_dfs[tp]["animal_id"].tolist())
            for tp in ["3m","9m","12m"] if tp in all_dfs and not all_dfs[tp].empty]):

        for feat in feat_cols:
            # Get baseline
            if "3m" not in all_dfs or all_dfs["3m"].empty:
                continue
            base_df = all_dfs["3m"]
            base_row = base_df[base_df.animal_id == animal_id]
            if base_row.empty or feat not in base_row.columns:
                continue
            base_val = base_row[feat].mean()
            if np.isnan(base_val) or base_val == 0:
                continue

            group = base_row["group"].iloc[0]

            for tp_late in ["9m","12m"]:
                if tp_late not in all_dfs or all_dfs[tp_late].empty:
                    continue
                late_df = all_dfs[tp_late]
                late_row = late_df[late_df.animal_id == animal_id]
                if late_row.empty or feat not in late_row.columns:
                    continue
                late_val = late_row[feat].mean()
                if np.isnan(late_val):
                    continue
                ri = late_val / base_val
                rows.append({
                    "animal_id": animal_id,
                    "group":     group,
                    "feature":   feat,
                    "timepoint": tp_late,
                    "baseline":  base_val,
                    "late_val":  late_val,
                    "recovery_index": ri,
                })

    if not rows:
        print("  Insufficient data for recovery index")
        return pd.DataFrame()

    ri_df = pd.DataFrame(rows)

    print("\nRecovery Index WT vs KO (value_late / value_3m):")
    for feat in feat_cols:
        for tp in ["9m","12m"]:
            sub = ri_df[(ri_df.feature==feat) & (ri_df.timepoint==tp)]
            wt = sub[sub.group=="WT"]["recovery_index"].dropna().values
            ko = sub[sub.group=="KO"]["recovery_index"].dropna().values
            if len(wt) < 2 or len(ko) < 2:
                continue
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            d, d_lo, d_hi = cohens_d_ci(wt, ko)
            ci_excl = (d_lo > 0) or (d_hi < 0)
            flag = "  CI✓" if ci_excl else ""
            print(f"  [{tp}] {feat:<25} WT_ri={np.mean(wt):.3f} "
                  f"KO_ri={np.mean(ko):.3f} d={d:.3f} p={p:.5f}{flag}")

    ri_df.to_csv(os.path.join(RESULTS_DIR, "recovery_index.csv"), index=False)
    print(f"\nSaved: recovery_index.csv ({len(ri_df)} rows)")
    return ri_df


# ── Predictive analysis: 3m → 9m/12m ────────────────────────────────────

def predictive_analysis(all_dfs):
    """
    Test whether 3m portfolio features predict 9m/12m band power outcomes.
    Replication of the vulnerability paper's key finding (r=0.601, p=0.0004).
    """
    print(f"\n{'='*60}")
    print("PREDICTIVE ANALYSIS: 3m features → 9m/12m outcomes")
    print(f"{'='*60}")

    if "3m" not in all_dfs or all_dfs["3m"].empty:
        print("  No 3m data")
        return pd.DataFrame()

    # 3m predictors from portfolio integration
    pred_df_3m = all_dfs["3m"].groupby(["animal_id","group"])[
        ["ap_exp_robust","peen","pac_theta_gamma","pac_alpha_gamma"]
    ].mean().reset_index()

    # Outcomes from band power (script 05)
    rows = []
    for tp_out in ["9m","12m"]:
        bp_path = os.path.join(DATA_DIR, f"state_specific_features_{tp_out}_A.csv")
        if not os.path.exists(bp_path):
            continue
        bp = pd.read_csv(bp_path)
        bp["animal_id"] = bp["animal_id"].astype(str)
        outcome_cols = ["nrem_rbp_theta","nrem_td_ratio","wake_rbp_beta",
                        "nrem_spectral_entropy","nrem_lzc"]
        outcome_cols = [c for c in outcome_cols if c in bp.columns]
        bp_animal = bp.groupby(["animal_id","group"])[outcome_cols].mean().reset_index()

        merged = pred_df_3m.merge(bp_animal, on=["animal_id","group"], suffixes=("_3m","_out"))

        for pred in ["ap_exp_robust","peen","pac_theta_gamma","pac_alpha_gamma"]:
            if pred not in merged.columns:
                continue
            for outcome in outcome_cols:
                if outcome not in merged.columns:
                    continue
                xy = merged[[pred, outcome]].dropna()
                if len(xy) < 5:
                    continue
                r, p = spearmanr(xy[pred], xy[outcome])
                rows.append({
                    "predictor":  pred,
                    "outcome":    outcome,
                    "tp_outcome": tp_out,
                    "rho":        round(r, 4),
                    "pval":       round(p, 5),
                    "n":          len(xy),
                })
                if abs(r) > 0.4 and p < 0.10:
                    print(f"  3m {pred:<22} → [{tp_out}] {outcome:<22} "
                          f"ρ={r:.3f} p={p:.5f}")

    pred_results = pd.DataFrame(rows)
    if len(pred_results) > 1:
        pred_results["pval_fdr"] = fdr_bh(pred_results["pval"].values).round(6)
        fdr_sig = pred_results[pred_results.pval_fdr < 0.05]
        print(f"\nFDR significant predictors: {len(fdr_sig)}")
        for _, r in fdr_sig.iterrows():
            print(f"  *** 3m {r.predictor} → [{r.tp_outcome}] {r.outcome}: "
                  f"ρ={r.rho:.3f} p={r.pval:.5f} FDR={r.pval_fdr:.5f}")

    pred_results.to_csv(os.path.join(RESULTS_DIR, "predictive_3m_to_outcome.csv"),
                         index=False)
    print(f"Saved: predictive_3m_to_outcome.csv ({len(pred_results)} rows)")
    return pred_results


# ── Trajectory plot ────────────────────────────────────────────────────────

def plot_portfolio_trajectories(all_dfs, stats_df):
    """Trajectory plot for key portfolio features across all 6 timepoints."""
    key_feats = [
        ("ap_exp_robust",    "1/f Aperiodic Exponent\n(E/I balance proxy)"),
        ("peen",             "Permutation Entropy\n(signal complexity)"),
        ("pac_theta_gamma",  "Theta-Gamma PAC\n(cross-frequency coupling)"),
        ("pac_alpha_gamma",  "Alpha-Gamma PAC"),
        ("wpli_theta",       "wPLI Theta\n(CA3-CTX connectivity)"),
    ]

    n = len(key_feats)
    fig, axes = plt.subplots(1, n, figsize=(n*4, 5))
    if n == 1:
        axes = [axes]

    for ax, (feat, ylabel) in zip(axes, key_feats):
        for group, color in COLORS.items():
            xs, means, sems = [], [], []
            for tp, tp_x in zip(TP_ORDER, TP_X):
                if tp not in all_dfs or all_dfs[tp].empty:
                    continue
                am = all_dfs[tp].groupby(["animal_id","group"])[feat].mean().reset_index()
                vals = am[am.group==group][feat].dropna()
                if len(vals) == 0:
                    continue
                xs.append(tp_x)
                means.append(vals.mean())
                sems.append(vals.sem())
                ax.scatter([tp_x]*len(vals), vals,
                            color=color, s=20, alpha=0.4, zorder=3)

            if xs:
                ax.fill_between(xs, np.array(means)-np.array(sems),
                                 np.array(means)+np.array(sems),
                                 color=color, alpha=0.15)
                ax.plot(xs, means, "-o", color=color, lw=2,
                         markersize=7, label=group, zorder=4)

        # Mark significant timepoints
        if len(stats_df) > 0 and "feature" in stats_df.columns:
            for tp, tp_x in zip(TP_ORDER, TP_X):
                sub = stats_df[(stats_df.feature==feat) & (stats_df.timepoint==tp)]
                if len(sub) == 0:
                    continue
                r = sub.iloc[0]
                if r.get("fdr_sig", False):
                    ax.text(tp_x, ax.get_ylim()[1]*0.97, "***",
                             ha="center", fontsize=9, color="darkred", weight="bold")
                elif r.ci_excludes_zero:
                    ax.text(tp_x, ax.get_ylim()[1]*0.97, "◄",
                             ha="center", fontsize=9, color="#D85A30")
                elif r.pval < 0.05:
                    ax.text(tp_x, ax.get_ylim()[1]*0.97, "†",
                             ha="center", fontsize=9, color="gray")

        ax.set_xticks(TP_X)
        ax.set_xticklabels(TP_ORDER, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Age (months)", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Portfolio Integration — Advanced Features Longitudinal Trajectory\n"
                  "C9orf72-KO vs WT | † p<0.05  ◄ CI excl. 0  *** FDR",
                  fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "portfolio_trajectory.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: portfolio_trajectory.png")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PORTFOLIO INTEGRATION ANALYSIS")
    print("Adding: FOOOF 1/f, PeEn, PAC, wPLI, Recovery Index, Predictive")
    print("=" * 60)

    # Load inventory
    inv_path = os.path.join(DATA_DIR, "file_inventory_all_timepoints.csv")
    if not os.path.exists(inv_path):
        print("ERROR: Run script 04 first")
        return
    inventory = pd.read_csv(inv_path)
    inventory["animal_id"] = inventory["animal_id"].astype(str)

    # Process each timepoint
    all_dfs = {}
    for tp in TP_ORDER:
        df = process_timepoint(tp, inventory)
        if not df.empty:
            all_dfs[tp] = df

    if not all_dfs:
        print("No data processed")
        return

    # Statistics
    stats_df = run_statistics_all(all_dfs)

    # Recovery index
    ri_df = compute_recovery_index(all_dfs)

    # Predictive analysis
    pred_df = predictive_analysis(all_dfs)

    # Figures
    print("\n=== GENERATING FIGURES ===")
    plot_portfolio_trajectories(all_dfs, stats_df)

    print("\n" + "="*60)
    print("PORTFOLIO INTEGRATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
