"""
09_behavioral_eeg_integration.py
==================================
Integrates behavioral test scores (10m timepoint) with
longitudinal EEG features to test predictive relationships.

Behavioral tests:
  - NOR (Novel Object Recognition): DI = (novel-familiar)/(novel+familiar)
    → Hippocampal-dependent memory. Correlate with theta power, PAC, spindles.
  - Open Field: total distance, center time, rearing count
    → Anxiety, locomotion, hyperactivity. Correlate with gamma, beta.
  - Fear Conditioning: % freezing (cued and contextual)
    → Amygdala-PFC. Correlate with theta-gamma PAC.
  - Grip Strength: average force (g)
    → Motor neuron function. Correlate with late-stage EEG collapse.

EEG predictors (from multiple timepoints):
  - 3m baseline features (pre-symptomatic predictors)
  - 9m features (concurrent)
  - Trajectory features (rate of change 3m→9m)

Analyses:
  1. Cross-sectional: 9m EEG vs 10m behavior
  2. Predictive: 3m EEG vs 10m behavior
  3. Trajectory: delta(3m→9m) EEG vs 10m behavior
  4. Multivariate: behavioral test battery → clustering
  5. Mediation: does EEG mediate genotype → behavior?

Input:
  - data/behavioral_scores_10m.csv  (you fill this in)
  - results/longitudinal_effect_sizes.csv
  - data/state_specific_features_{tp}.csv

Output:
  - results/behavioral_eeg_correlations.csv
  - figures/behavioral_*.png

Run:
    python src/09_behavioral_eeg_integration.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")

COLORS   = {"WT": "#378ADD", "KO": "#D85A30"}
TP_ORDER = ["3m", "4m", "6m", "7m", "9m", "12m"]


# ── Create behavioral template if missing ──────────────────────────────────

def create_behavioral_template(animal_ids_wt, animal_ids_ko):
    """
    Create a fillable template for behavioral scores.
    Designed for 10m behavioral battery.
    """
    template_path = os.path.join(DATA_DIR, "behavioral_scores_10m.csv")
    if os.path.exists(template_path):
        print(f"Template already exists: {template_path}")
        return pd.read_csv(template_path)

    rows = []
    for aid in animal_ids_wt:
        rows.append({"animal_id": aid, "group": "WT",
                     # NOR
                     "nor_di": np.nan,                  # discrimination index (-1 to 1; healthy ~0.2-0.4)
                     "nor_novel_time_s": np.nan,
                     "nor_familiar_time_s": np.nan,
                     "nor_total_exploration_s": np.nan,
                     # Open field
                     "of_distance_m": np.nan,           # total distance (m)
                     "of_center_pct": np.nan,           # % time in center zone
                     "of_rearing_n": np.nan,            # number of rears
                     "of_velocity_cm_s": np.nan,        # mean velocity
                     # Fear conditioning
                     "fc_baseline_freeze_pct": np.nan,  # pre-CS baseline
                     "fc_cued_freeze_pct": np.nan,      # cued (CS+) freezing
                     "fc_contextual_freeze_pct": np.nan, # contextual (24h)
                     # Grip strength
                     "grip_forelimb_g": np.nan,         # forelimb grip (g)
                     "grip_hindlimb_g": np.nan,         # hindlimb grip (g)
                     "grip_composite": np.nan,          # normalized by body weight
                     # Body weight
                     "body_weight_g": np.nan,
                     # Notes
                     "notes": "",
                     })
    for aid in animal_ids_ko:
        rows.append({"animal_id": aid, "group": "KO",
                     "nor_di": np.nan, "nor_novel_time_s": np.nan,
                     "nor_familiar_time_s": np.nan, "nor_total_exploration_s": np.nan,
                     "of_distance_m": np.nan, "of_center_pct": np.nan,
                     "of_rearing_n": np.nan, "of_velocity_cm_s": np.nan,
                     "fc_baseline_freeze_pct": np.nan, "fc_cued_freeze_pct": np.nan,
                     "fc_contextual_freeze_pct": np.nan,
                     "grip_forelimb_g": np.nan, "grip_hindlimb_g": np.nan,
                     "grip_composite": np.nan, "body_weight_g": np.nan, "notes": ""
                     })

    df = pd.DataFrame(rows)
    df.to_csv(template_path, index=False)
    print(f"Behavioral template created: {template_path}")
    print(f"  {len(df)} animals — fill in and rerun")
    print(f"\n  Columns:")
    for c in df.columns:
        print(f"    {c}")
    return df


# ── Load EEG features at specified timepoints ──────────────────────────────

def load_eeg_animal_means(tp, scenario="A"):
    """Load animal-level mean EEG features for one timepoint (Scenario A = animal-level)."""
    path = os.path.join(DATA_DIR, f"state_specific_features_{tp}_{scenario}.csv")
    if not os.path.exists(path):
        # Fallback to un-suffixed name for backward compatibility
        alt = os.path.join(DATA_DIR, f"state_specific_features_{tp}.csv")
        path = alt if os.path.exists(alt) else path
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["animal_id"] = df["animal_id"].astype(str)
    # Only numeric feature columns (exclude ids, group, timepoint, text columns)
    exclude = {"animal_id", "session_id", "group", "timepoint", "abf_file", "notes"}
    feat_cols = [c for c in df.columns
                 if c not in exclude
                 and not c.endswith("_n_epochs") and not c.endswith("_std")
                 and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        return pd.DataFrame()
    animal_means = df.groupby(["animal_id", "group"])[feat_cols].mean().reset_index()
    animal_means["timepoint"] = tp
    return animal_means


# ── Behavioral group comparison ────────────────────────────────────────────

def behavioral_group_comparison(behav_df):
    """WT vs KO comparison for all behavioral outcomes."""
    outcomes = [c for c in behav_df.columns
                if c not in ["animal_id","group","notes","body_weight_g"]]
    rows = []
    print("\nBEHAVIORAL GROUP COMPARISONS (WT vs KO):")
    for col in outcomes:
        wt = behav_df[behav_df.group=="WT"][col].dropna()
        ko = behav_df[behav_df.group=="KO"][col].dropna()
        if len(wt) < 2 or len(ko) < 2:
            continue
        _, p = mannwhitneyu(wt, ko, alternative="two-sided")
        d = (ko.mean()-wt.mean())/np.sqrt((wt.std()**2+ko.std()**2)/2+1e-10)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        rows.append({"outcome": col, "wt_mean": wt.mean(), "ko_mean": ko.mean(),
                     "cohens_d": d, "pval": p, "sig": sig})
        print(f"  {col:<30}: WT={wt.mean():.3f} KO={ko.mean():.3f} d={d:.3f} p={p:.4f} {sig}")
    return pd.DataFrame(rows)


# ── EEG-Behavior correlations ─────────────────────────────────────────────

def eeg_behavior_correlations(eeg_df, behav_df, eeg_tp_label,
                                eeg_cols_of_interest=None):
    """
    Compute Spearman correlations between EEG features and behavioral outcomes.
    Tests all-animals, then WT-only and KO-only separately.
    """
    if eeg_cols_of_interest is None:
        eeg_cols_of_interest = [c for c in eeg_df.columns
                                 if c not in ["animal_id","group","timepoint"]]

    behav_cols = [c for c in behav_df.columns
                  if c not in ["animal_id","group","notes","body_weight_g"]]

    merged = behav_df.merge(eeg_df[["animal_id","group"] + eeg_cols_of_interest],
                             on=["animal_id","group"], how="inner")
    print(f"\n  Merged n={len(merged)} animals for {eeg_tp_label} EEG vs 10m behavior")

    rows = []
    for eeg_col in eeg_cols_of_interest:
        for beh_col in behav_cols:
            xy = merged[[eeg_col, beh_col]].dropna()
            if len(xy) < 5:
                continue
            r, p = spearmanr(xy[eeg_col], xy[beh_col])
            rows.append({
                "eeg_feature":  eeg_col,
                "eeg_timepoint": eeg_tp_label,
                "behavior":     beh_col,
                "rho":          round(r, 4),
                "pval":         round(p, 5),
                "n":            len(xy),
            })

    corr_df = pd.DataFrame(rows)
    if len(corr_df) > 1:
        _, pfd, _, _ = multipletests(corr_df["pval"], method="fdr_bh")
        corr_df["pval_fdr"] = pfd
    corr_df["sig"] = corr_df["pval"].apply(
        lambda x: "***" if x<0.001 else "**" if x<0.01 else "*" if x<0.05 else "ns")

    sig = corr_df[corr_df.pval < 0.05].sort_values("pval")
    print(f"  Significant correlations (p<0.05): {len(sig)}/{len(corr_df)}")
    for _, r in sig.head(15).iterrows():
        print(f"    [{r.eeg_timepoint}] {r.eeg_feature:<30} ↔ {r.behavior:<25} "
              f"ρ={r.rho:.3f} p={r.pval:.4f}")

    return corr_df


# ── Trajectory predictor analysis ─────────────────────────────────────────

def trajectory_predictors(eeg_3m, eeg_late, behav_df, label="3m→9m_delta"):
    """
    Test whether the rate of EEG change (3m→late) predicts 10m behavior.
    `label` names the contrast (e.g. '3m→9m_delta' or '3m→12m_delta').
    """
    common_animals = (set(eeg_3m.animal_id) &
                      set(eeg_late.animal_id) &
                      set(behav_df.animal_id))
    if len(common_animals) < 4:
        print(f"  Insufficient overlap for trajectory analysis ({label})")
        return pd.DataFrame()

    feat_cols = [c for c in eeg_3m.columns
                 if c not in ["animal_id","group","timepoint"]
                 and c in eeg_late.columns]

    # Compute delta (late - 3m)
    m3 = eeg_3m[eeg_3m.animal_id.isin(common_animals)].set_index("animal_id")
    mL = eeg_late[eeg_late.animal_id.isin(common_animals)].set_index("animal_id")

    delta = (mL[feat_cols] - m3[feat_cols]).reset_index()
    delta.columns = ["animal_id"] + [f"delta_{c}" for c in feat_cols]
    delta["group"] = m3.loc[delta.animal_id, "group"].values

    delta_cols = [c for c in delta.columns if c.startswith("delta_")]
    corr_df = eeg_behavior_correlations(
        delta.rename(columns={c: c for c in delta_cols}),
        behav_df, label, delta_cols
    )
    safe = label.replace("→", "_to_").replace("_delta", "")
    corr_df.to_csv(os.path.join(RESULTS_DIR, f"trajectory_behavior_correlations_{safe}.csv"),
                   index=False)
    # keep legacy filename for the primary 3m→9m contrast
    if label == "3m→9m_delta":
        corr_df.to_csv(os.path.join(RESULTS_DIR, "trajectory_behavior_correlations.csv"),
                       index=False)
    return corr_df


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Behavioral-EEG Integration Analysis")
    print("=" * 65)

    # Load all animal IDs from available EEG data
    eeg_3m = load_eeg_animal_means("3m")
    all_wt = eeg_3m[eeg_3m.group=="WT"]["animal_id"].tolist() if not eeg_3m.empty else []
    all_ko = eeg_3m[eeg_3m.group=="KO"]["animal_id"].tolist() if not eeg_3m.empty else []

    # Load or create behavioral template
    behav_path = os.path.join(DATA_DIR, "behavioral_scores_10m.csv")
    if not os.path.exists(behav_path):
        print("\nCreating behavioral template...")
        create_behavioral_template(all_wt, all_ko)
        print("\nFill in data/behavioral_scores_10m.csv and rerun.")
        return

    behav = pd.read_csv(behav_path)
    behav["animal_id"] = behav["animal_id"].astype(str)
    behav_filled = behav.dropna(subset=["nor_di","grip_forelimb_g"], how="all")
    print(f"\nLoaded behavioral data: {len(behav_filled)}/{len(behav)} animals filled")

    if len(behav_filled) < 4:
        print("Not enough behavioral data filled — run after scoring.")
        return

    # ── Group comparisons ─────────────────────────────────────────────────
    grp_stats = behavioral_group_comparison(behav_filled)
    grp_stats.to_csv(os.path.join(RESULTS_DIR, "behavioral_group_comparison.csv"),
                     index=False)

    # ── EEG-Behavior correlations (concurrent 9m and predictive 3m) ───────
    all_corr = []

    # Key EEG features to test
    key_eeg_feats = [
        # REM features — the paper's primary findings
        "rem_rbp_theta", "rem_rbp_beta", "rem_td_ratio", "rem_theta_alpha",
        "rem_zcr",
        # Wake/NREM
        "wake_rbp_theta", "nrem_rbp_theta", "wake_rbp_beta",
        "wake_td_ratio", "nrem_td_ratio",
        # Complexity / aperiodic (pre-symptomatic predictors)
        "wake_spectral_entropy", "nrem_spectral_entropy",
        "wake_lzc", "nrem_lzc", "rem_lzc",
        "wake_ap_exp", "ap_exp", "aperiodic_exponent",
        "wake_hjorth_mob", "wake_rbp_gamma", "wake_rbp_delta",
        # Spindles
        "spindle_duration_mean",
    ]

    for tp, label in [("3m", "3m_baseline"), ("9m", "9m_concurrent"), ("12m", "12m_endstage")]:
        eeg = load_eeg_animal_means(tp)
        if eeg.empty:
            print(f"  [{tp}] no EEG features found (state_specific_features_{tp}.csv) — skipping")
            continue
        eeg_feats = [c for c in key_eeg_feats if c in eeg.columns]
        corr = eeg_behavior_correlations(eeg, behav_filled, label, eeg_feats)
        all_corr.append(corr)

    # ── Trajectory predictors ──────────────────────────────────────────────
    eeg_9m = load_eeg_animal_means("9m")
    if not eeg_3m.empty and not eeg_9m.empty:
        traj_corr = trajectory_predictors(eeg_3m, eeg_9m, behav_filled)
        if not traj_corr.empty:
            all_corr.append(traj_corr)

    # 3m→12m trajectory (pre-specified motor anchor: late-stage beta change vs grip)
    eeg_12m = load_eeg_animal_means("12m")
    if not eeg_3m.empty and not eeg_12m.empty:
        traj_corr_12 = trajectory_predictors(eeg_3m, eeg_12m, behav_filled,
                                             label="3m→12m_delta")
        if not traj_corr_12.empty:
            all_corr.append(traj_corr_12)

    # ── Save all correlations ──────────────────────────────────────────────
    if all_corr:
        full_corr = pd.concat(all_corr, ignore_index=True)
        full_corr.to_csv(os.path.join(RESULTS_DIR, "behavioral_eeg_correlations.csv"),
                         index=False)
        print(f"\nSaved: behavioral_eeg_correlations.csv ({len(full_corr)} tests)")

    # ── Plot: key scatter plots ────────────────────────────────────────────
    if all_corr and len(all_corr[0]) > 0:
        sig_corrs = pd.concat(all_corr).nsmallest(6, "pval")
        _plot_scatter_grid(sig_corrs, behav_filled,
                           {tp: load_eeg_animal_means(tp)
                            for tp in ["3m","9m","12m"] if not load_eeg_animal_means(tp).empty})

    print("\nBEHAVIORAL-EEG INTEGRATION COMPLETE")


def _plot_scatter_grid(sig_corrs, behav_df, eeg_by_tp):
    """Plot grid of top EEG-behavior scatter plots."""
    if len(sig_corrs) == 0:
        return
    n = min(len(sig_corrs), 6)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    axes = axes.flat

    for ax, (_, row) in zip(axes, sig_corrs.head(n).iterrows()):
        tp_key = row["eeg_timepoint"].replace("_baseline","").replace("_concurrent","").replace("_endstage","")
        if tp_key not in eeg_by_tp:
            ax.set_visible(False); continue
        eeg = eeg_by_tp[tp_key]
        merged = behav_df.merge(eeg[["animal_id","group",row["eeg_feature"]]],
                                 on=["animal_id","group"], how="inner")
        xy = merged[[row["eeg_feature"], row["behavior"], "group"]].dropna()
        if len(xy) < 3:
            ax.set_visible(False); continue

        for g, color, marker in [("WT","#378ADD","o"),("KO","#D85A30","s")]:
            sub = xy[xy.group==g]
            ax.scatter(sub[row["eeg_feature"]], sub[row["behavior"]],
                       color=color, marker=marker, s=60, alpha=0.8, label=g)

        x_all = xy[row["eeg_feature"]].values
        y_all = xy[row["behavior"]].values
        slope, intercept = np.polyfit(x_all, y_all, 1)
        xl = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(xl, slope*xl+intercept, "k--", lw=1.5, alpha=0.5)

        ax.set_xlabel(f"{row['eeg_feature']}\n({row['eeg_timepoint']})", fontsize=8)
        ax.set_ylabel(row["behavior"].replace("_"," "), fontsize=8)
        ax.set_title(f"ρ={row['rho']:.3f} p={row['pval']:.4f}", fontsize=9)
        ax.legend(fontsize=7)

    for ax in axes: ax.set_visible(ax.get_visible())
    fig.suptitle("EEG Features Predict Behavioral Outcomes\nC9orf72-KO vs WT", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "behavioral_eeg_scatterplots.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: behavioral_eeg_scatterplots.png")


if __name__ == "__main__":
    main()
