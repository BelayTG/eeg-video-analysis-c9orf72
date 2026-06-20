"""
06_longitudinal_trajectory.py
==============================
Builds the longitudinal EEG feature trajectory across all 6 timepoints.
This is the core analysis for the Nature-level paper.

TWO SCENARIOS run in parallel throughout:

  Scenario A — Animal-level (5-digit animal_id, n ≈ 9 WT, 9 KO per TP)
      All ABF files across both sessions averaged per animal.
      Most conservative; primary statistical unit for publication.

  Scenario B — Session-level (6-digit session_id, n ≈ 18 WT, 18 KO per TP)
      Files within each session averaged; two data points per animal.
      Greater power; used as confirmatory / sensitivity analysis.

Both scenarios are computed identically and saved as separate CSVs.
Trajectory figures show Scenario A means ± SEM with individual animal lines.

Analyses:
  1. Longitudinal band power trajectory (WT vs KO, all timepoints)
  2. Sleep architecture evolution over time
  3. Effect size (Cohen's d) trajectory per feature × timepoint
  4. Individual animal trajectories (spaghetti plots)
  5. Multivariate trajectory (PCA of all features × timepoints)
  6. Rate of change between adjacent timepoints (delta analysis)
  7. Predictive feature identification: which 3m features predict group?
  8. Effect size heatmap

Inputs:
  - data/state_specific_features_{tp}.csv (from script 05, one row per ABF file)

Outputs:
  - results/longitudinal_effect_sizes_scenA.csv
  - results/longitudinal_effect_sizes_scenB.csv
  - figures/longitudinal_*.png

Run:
    python src/06_longitudinal_trajectory.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
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
TP_X     = [3, 4, 6, 7, 9, 12]   # numeric x-axis months


# ── Load all timepoints ────────────────────────────────────────────────────

def load_all_timepoints(scenario="A"):
    """
    Load state_specific_features for all timepoints for one scenario.
    Scenario A: state_specific_features_{tp}_A.csv  (unit = animal_id)
    Scenario B: state_specific_features_{tp}_B.csv  (unit = session_id)
    """
    unit_col = "animal_id" if scenario == "A" else "session_id"
    dfs = []
    for tp in TP_ORDER:
        path = os.path.join(DATA_DIR, f"state_specific_features_{tp}_{scenario}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["timepoint"] = tp
            df["animal_id"] = df["animal_id"].astype(str)
            if "session_id" in df.columns:
                df["session_id"] = df["session_id"].astype(str)
            df["unit_id"] = df[unit_col].astype(str)
            dfs.append(df)
            n_units = df["unit_id"].nunique()
            print(f"  [Sc {scenario}] {tp}: {len(df)} file-rows | {n_units} {unit_col}s")
        else:
            print(f"  MISSING: state_specific_features_{tp}_{scenario}.csv "
                  f"— run script 05 first")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_sleep_summaries(scenario="A"):
    """Load sleep summaries for all timepoints."""
    dfs = []
    for tp in TP_ORDER:
        path = os.path.join(DATA_DIR, f"sleep_session_summary_{tp}_{scenario}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["timepoint"] = tp
            df["animal_id"] = df["animal_id"].astype(str)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ── Scenario-aware aggregation ─────────────────────────────────────────────

def get_scenario_df(raw_df, feat_cols, scenario="A"):
    """
    Aggregate the per-file table to the correct statistical unit level.
    unit_id is already set by load_all_timepoints(); we just group by it.
    """
    if "unit_id" not in raw_df.columns:
        raw_df = raw_df.copy()
        raw_df["unit_id"] = raw_df["animal_id"] if scenario == "A" else raw_df.get("session_id", raw_df["animal_id"])
    grp_cols = ["unit_id", "animal_id", "group", "timepoint"]
    grp_cols = [c for c in grp_cols if c in raw_df.columns]
    # Keep only numeric feature columns that exist in the dataframe
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in feat_cols if c in raw_df.columns and c in numeric_cols]
    grp = raw_df.groupby(grp_cols)[feat_cols].mean().reset_index()
    return grp


# ── Effect size trajectory ─────────────────────────────────────────────────

def compute_effect_size_trajectory(scenario_df, feat_cols, scenario_label):
    """
    For each feature × timepoint: compute Cohen's d (KO vs WT) and p-value.
    scenario_df must already be aggregated to the correct unit (A or B).
    Returns DataFrame indexed by (feature, timepoint).
    """
    rows = []
    for tp in TP_ORDER:
        sub = scenario_df[scenario_df.timepoint == tp]
        if len(sub) < 4:
            continue
        for feat in feat_cols:
            if feat not in sub.columns:
                continue
            wt = sub[sub.group == "WT"][feat].dropna()
            ko = sub[sub.group == "KO"][feat].dropna()
            if len(wt) < 2 or len(ko) < 2:
                continue
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            d = ((ko.mean() - wt.mean()) /
                 np.sqrt((wt.std()**2 + ko.std()**2) / 2 + 1e-10))
            rows.append({
                "feature":   feat,
                "timepoint": tp,
                "tp_x":      TP_X[TP_ORDER.index(tp)],
                "wt_mean":   float(wt.mean()),
                "wt_sem":    float(wt.sem()),
                "ko_mean":   float(ko.mean()),
                "ko_sem":    float(ko.sem()),
                "cohens_d":  float(d),
                "pval":      float(p),
                "n_wt":      len(wt),
                "n_ko":      len(ko),
                "scenario":  scenario_label,
            })

    traj = pd.DataFrame(rows)
    if len(traj) > 1:
        _, pfd, _, _ = multipletests(traj["pval"], method="fdr_bh")
        traj["pval_fdr"] = pfd
        traj["sig"] = traj["pval"].apply(
            lambda x: "***" if x<0.001 else "**" if x<0.01 else "*" if x<0.05 else "ns")
    return traj


# ── Plotting helpers ───────────────────────────────────────────────────────

def plot_trajectory(animal_df, feat, title, ylabel, fname, log_scale=False):
    """
    Longitudinal trajectory plot for a single feature.
    Mean ± SEM per group per timepoint + individual animal lines.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for group, color in COLORS.items():
        grp = animal_df[animal_df.group == group]
        xs, means, sems = [], [], []
        for tp, tp_x in zip(TP_ORDER, TP_X):
            sub = grp[grp.timepoint == tp][feat].dropna()
            if len(sub) == 0:
                continue
            xs.append(tp_x)
            means.append(sub.mean())
            sems.append(sub.sem())

        xs    = np.array(xs)
        means = np.array(means)
        sems  = np.array(sems)

        ax.fill_between(xs, means - sems, means + sems,
                        color=color, alpha=0.15)
        ax.plot(xs, means, "-o", color=color, lw=2,
                markersize=7, label=group, zorder=4)
        ax.errorbar(xs, means, yerr=sems, fmt="none",
                    color=color, capsize=4, capthick=1.5, elinewidth=1.5)

        # Individual animal trajectories (thin lines)
        for animal, adf in grp.groupby("animal_id"):
            adf = adf.sort_values("timepoint",
                                   key=lambda s: s.map({t: i for i, t in enumerate(TP_ORDER)}))
            ax_xs = [TP_X[TP_ORDER.index(t)] for t in adf["timepoint"] if t in TP_ORDER]
            ax_ys = [adf[adf.timepoint == t][feat].mean() for t in adf["timepoint"] if t in TP_ORDER]
            ax.plot(ax_xs, ax_ys, "-", color=color, lw=0.5, alpha=0.35)

    ax.set_xticks(TP_X[:len([t for t in TP_ORDER])])
    ax.set_xticklabels(TP_ORDER, fontsize=9)
    ax.set_xlabel("Timepoint (months)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    if log_scale:
        ax.set_yscale("log")

    # Add significance markers, positioned just above each timepoint's data
    def _sig_label(p):
        if p < 0.001: return "***", "p<0.001"
        if p < 0.01:  return "**",  f"p={p:.3f}"
        if p < 0.05:  return "*",   f"p={p:.3f}"
        return "ns", f"p={p:.2f}"

    def _err_top(v):
        # top of the error bar = mean + SEM (matches the errorbar drawn on the plot)
        m = v.mean()
        sem = v.std(ddof=1) / max(len(v) ** 0.5, 1)
        return m + sem

    y0, y1 = ax.get_ylim()
    yspan = y1 - y0
    top_label_y = y1

    for tp, tp_x in zip(TP_ORDER, TP_X):
        sub = animal_df[animal_df.timepoint == tp]
        wt  = sub[sub.group == "WT"][feat].dropna()
        ko  = sub[sub.group == "KO"][feat].dropna()
        if len(wt) < 2 or len(ko) < 2:
            continue
        _, p = mannwhitneyu(wt, ko, alternative="two-sided")
        star, ptxt = _sig_label(p)
        is_sig = p < 0.05
        col = "red" if is_sig else "gray"

        # local ceiling: higher group's mean + SEM at THIS timepoint
        y_data = max(_err_top(wt), _err_top(ko))
        y_lab  = y_data + yspan * 0.05          # star sits just above the error bar
        y_ptxt = y_lab  + yspan * 0.045         # p-value sits above the star
        top_label_y = max(top_label_y, y_ptxt + yspan * 0.04)

        ax.text(tp_x, y_lab, star, ha="center", va="bottom",
                fontsize=10, color=col, weight="bold" if is_sig else "normal")
        ax.text(tp_x, y_ptxt, ptxt, ha="center", va="bottom",
                fontsize=6.5, color=col)

    # expand the top only if the tallest label needs the room
    if top_label_y > y1:
        ax.set_ylim(y0, top_label_y)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

def plot_effect_size_heatmap(traj_df, top_n=20, suffix=""):
    """
    Heatmap: features (rows) × timepoints (cols), colored by Cohen's d.
    Shows top N features by max |d| across timepoints.
    """
    # Select top features
    max_d = traj_df.groupby("feature")["cohens_d"].apply(lambda x: x.abs().max())
    top_feats = max_d.nlargest(top_n).index.tolist()

    tps_avail = [tp for tp in TP_ORDER if tp in traj_df["timepoint"].unique()]

    matrix = np.full((len(top_feats), len(tps_avail)), np.nan)
    pmat   = np.full_like(matrix, 1.0)

    for i, feat in enumerate(top_feats):
        for j, tp in enumerate(tps_avail):
            sub = traj_df[(traj_df.feature == feat) & (traj_df.timepoint == tp)]
            if len(sub) > 0:
                matrix[i, j] = sub["cohens_d"].values[0]
                pmat[i, j]   = sub["pval"].values[0]

    fig, ax = plt.subplots(figsize=(max(8, len(tps_avail) * 1.3), max(8, top_n * 0.35)))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label="Cohen's d (KO − WT)")

    # Significance overlays
    for i in range(len(top_feats)):
        for j in range(len(tps_avail)):
            if pmat[i, j] < 0.05:
                marker = "***" if pmat[i,j] < 0.001 else "**" if pmat[i,j] < 0.01 else "*"
                ax.text(j, i, marker, ha="center", va="center",
                        fontsize=6, color="black", weight="bold")

    ax.set_xticks(range(len(tps_avail)))
    ax.set_xticklabels(tps_avail, fontsize=9)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=7)
    ax.set_xlabel("Timepoint", fontsize=10)
    ax.set_title(f"EEG Feature Effect Size (Cohen's d) — Top {top_n} Features\n"
                 "Positive = KO > WT | * p<0.05 ** p<0.01 *** p<0.001", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"longitudinal_effect_size_heatmap_{suffix}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: longitudinal_effect_size_heatmap_{suffix}.png")


def plot_pca_trajectory(animal_df, feat_cols, suffix="scenA"):
    """
    PCA of all features per animal × timepoint.
    Color by group, marker by timepoint.
    Draws mean trajectory lines per group.
    """
    sub = animal_df[feat_cols + ["group", "timepoint", "animal_id"]].dropna()
    if len(sub) < 4:
        return

    X = sub[feat_cols].values
    y_group = sub["group"].values
    y_tp    = sub["timepoint"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=2)
    X_pca  = pca.fit_transform(X_sc)

    markers = {tp: m for tp, m in zip(TP_ORDER, ["o","s","^","D","v","P"])}

    fig, ax = plt.subplots(figsize=(8, 7))

    for group, color in COLORS.items():
        for tp in TP_ORDER:
            mask = (y_group == group) & (y_tp == tp)
            if not mask.any():
                continue
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       color=color, marker=markers.get(tp, "o"),
                       s=60, alpha=0.7, label=f"{group} {tp}", zorder=4)

        # Mean trajectory per group
        xs, ys = [], []
        for tp in TP_ORDER:
            mask = (y_group == group) & (y_tp == tp)
            if not mask.any():
                continue
            xs.append(X_pca[mask, 0].mean())
            ys.append(X_pca[mask, 1].mean())
        if len(xs) > 1:
            ax.plot(xs, ys, "-", color=color, lw=2.5, alpha=0.8)
            for i, tp in enumerate([t for t in TP_ORDER if
                                     any((y_group==group) & (y_tp==t))]):
                ax.annotate(tp, (xs[i], ys[i]), fontsize=7,
                            color=color, ha="center", va="bottom",
                            xytext=(0, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    ax.set_title("PCA Trajectory — EEG Feature Space\n"
                 "Each point = one animal × timepoint", fontsize=11)

    handles = [plt.Line2D([0],[0], color=COLORS["WT"], marker="o", linestyle="-",
                          label="WT"),
               plt.Line2D([0],[0], color=COLORS["KO"], marker="o", linestyle="-",
                          label="KO")]
    ax.legend(handles=handles, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"longitudinal_pca_trajectory_{suffix}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: longitudinal_pca_trajectory_{suffix}.png")


def plot_sleep_architecture_longitudinal(sleep_df):
    """
    Stacked area chart: sleep state proportions over time per group.
    """
    animal_sleep = (sleep_df.groupby(["animal_id","group","timepoint"])
                   [["pct_wake","pct_nrem","pct_rem"]].mean().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    state_colors = {"pct_wake": "#E67E22", "pct_nrem": "#3498DB", "pct_rem": "#9B59B6"}
    state_labels = {"pct_wake": "Wake", "pct_nrem": "NREM", "pct_rem": "REM"}

    for ax, group in zip(axes, ["WT", "KO"]):
        grp = animal_sleep[animal_sleep.group == group]
        tp_avail = [tp for tp in TP_ORDER if tp in grp.timepoint.unique()]
        tp_xs    = [TP_X[TP_ORDER.index(t)] for t in tp_avail]

        bottom = np.zeros(len(tp_avail))
        for col, color in state_colors.items():
            means = [grp[grp.timepoint == tp][col].mean() for tp in tp_avail]
            sems  = [grp[grp.timepoint == tp][col].sem()  for tp in tp_avail]
            ax.bar(tp_xs, means, bottom=bottom, color=color, alpha=0.8,
                   label=state_labels[col], width=0.6)
            bottom += np.array(means)

        ax.set_xticks(tp_xs)
        ax.set_xticklabels(tp_avail, fontsize=9)
        ax.set_xlabel("Timepoint (months)", fontsize=10)
        ax.set_ylabel("% Time", fontsize=10)
        ax.set_title(f"{group}", fontsize=11)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Sleep Architecture Longitudinal Trajectory — C9orf72-KO vs WT",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "longitudinal_sleep_architecture.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: longitudinal_sleep_architecture.png")


def predictive_feature_analysis(animal_df, feat_cols):
    """
    Test whether 3m baseline EEG features predict group membership at later timepoints.
    Uses logistic regression with LOO-CV.
    """
    print("\n=== PREDICTIVE FEATURE ANALYSIS ===")
    results = []

    baseline = animal_df[animal_df.timepoint == "3m"][["animal_id","group"] + feat_cols].dropna()
    if len(baseline) < 6:
        print("  Insufficient 3m data for predictive analysis")
        return pd.DataFrame()

    X = baseline[feat_cols].values
    y = (baseline.group == "KO").astype(int).values

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(C=0.1, max_iter=500, random_state=42))])
    scores = cross_val_score(pipe, X, y, cv=LeaveOneOut(), scoring="accuracy")
    print(f"  3m → group classification (LOO-CV): "
          f"{scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    # Individual feature predictive power
    feature_scores = []
    for feat in feat_cols:
        x = baseline[[feat]].values
        if np.isnan(x).all():
            continue
        try:
            p1 = Pipeline([("scaler", StandardScaler()),
                            ("lr", LogisticRegression(C=1, max_iter=500, random_state=42))])
            s = cross_val_score(p1, x, y, cv=LeaveOneOut(), scoring="accuracy")
            feature_scores.append({"feature": feat, "accuracy": s.mean()})
        except Exception:
            pass

    fs_df = pd.DataFrame(feature_scores).sort_values("accuracy", ascending=False)
    print("  Top 10 individually predictive 3m features:")
    print(fs_df.head(10).to_string(index=False))

    return fs_df


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Longitudinal EEG Trajectory Analysis — Both Scenarios")
    print("=" * 65)

    # Load raw per-file data
    print("\nLoading state-specific features (raw, per ABF file)...")
    raw_df = load_all_timepoints()
    if raw_df.empty:
        print("ERROR: No feature data found. Run script 05 first.")
        return

    print("\nLoading sleep summaries...")
    sleep_df = load_sleep_summaries()

    # Feature columns (exclude metadata, n_epochs, std columns)
    meta = {"animal_id", "session_id", "unit_id", "group", "timepoint", "abf_file",
            "n_epochs_total", "pct_wake", "pct_nrem", "pct_rem"}
    feat_cols = [c for c in raw_df.columns
                 if c not in meta
                 and not c.endswith("_n_epochs")
                 and not c.endswith("_std")
                 and not c.endswith("_pct")
                 and raw_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    print(f"\nFeature columns: {len(feat_cols)}")

    # ── Aggregate to both scenarios ───────────────────────────────────────
    print("\nAggregating to Scenario A (animal-level) ...")
    animal_df = get_scenario_df(raw_df, feat_cols, scenario="A")
    print(f"  Shape: {animal_df.shape}  (animals × timepoints)")

    print("Aggregating to Scenario B (session-level) ...")
    session_df = get_scenario_df(raw_df, feat_cols, scenario="B")
    print(f"  Shape: {session_df.shape}  (sessions × timepoints)")

    # ── 1. Effect size trajectories (both scenarios) ──────────────────────
    print("\n=== EFFECT SIZE TRAJECTORIES ===")
    traj_A = compute_effect_size_trajectory(animal_df,  feat_cols, "A_animal")
    traj_B = compute_effect_size_trajectory(session_df, feat_cols, "B_session")

    traj_A.to_csv(os.path.join(RESULTS_DIR, "longitudinal_effect_sizes_scenA.csv"), index=False)
    traj_B.to_csv(os.path.join(RESULTS_DIR, "longitudinal_effect_sizes_scenB.csv"), index=False)

    # Combined table for convenience
    traj_both = pd.concat([traj_A, traj_B], ignore_index=True)
    traj_both.to_csv(os.path.join(RESULTS_DIR, "longitudinal_effect_sizes_both.csv"), index=False)
    print(f"  Saved: longitudinal_effect_sizes_scenA/B.csv")

    # Print top findings per timepoint, per scenario
    for scen_label, traj in [("Scenario A (animal)", traj_A),
                               ("Scenario B (session)", traj_B)]:
        print(f"\n  {scen_label}:")
        for tp in TP_ORDER:
            sub = traj[traj.timepoint == tp].sort_values("pval")
            sig = sub[sub.pval < 0.05]
            if len(sig) > 0:
                top = sig.head(3)
                print(f"    {tp} — {len(sig)} sig features: "
                      + " | ".join(f"{r.feature} d={r.cohens_d:+.2f} p={r.pval:.4f}"
                                   for _, r in top.iterrows()))

    # ── 2. Scenario comparison — do conclusions differ? ───────────────────
    print("\n=== SCENARIO A vs B COMPARISON ===")
    sig_A = set(zip(traj_A[traj_A.pval < 0.05].feature,
                    traj_A[traj_A.pval < 0.05].timepoint))
    sig_B = set(zip(traj_B[traj_B.pval < 0.05].feature,
                    traj_B[traj_B.pval < 0.05].timepoint))
    both  = sig_A & sig_B
    only_A = sig_A - sig_B
    only_B = sig_B - sig_A
    print(f"  Significant in BOTH scenarios:      {len(both)}")
    print(f"  Significant in Scenario A only:     {len(only_A)}")
    print(f"  Significant in Scenario B only:     {len(only_B)}")
    if only_A:
        print(f"  A-only findings: {sorted(only_A)[:10]}")
    if only_B:
        print(f"  B-only findings (higher power): {sorted(only_B)[:10]}")
    # Save comparison table
    merged_traj = traj_A.merge(
        traj_B[["feature","timepoint","cohens_d","pval","sig"]],
        on=["feature","timepoint"], suffixes=("_A","_B"), how="outer"
    )
    merged_traj.to_csv(
        os.path.join(RESULTS_DIR, "longitudinal_scenAB_comparison.csv"),
        index=False)
    print(f"  Saved: longitudinal_scenAB_comparison.csv")

    # ── 3. Key feature trajectories (use Scenario A for figures) ──────────
    print("\n=== PLOTTING KEY FEATURE TRAJECTORIES (Scenario A) ===")
    key_features = [
        ("wake_rbp_theta",        "Wake Relative Theta Power",      "Relative Theta (Wake)"),
        ("nrem_rbp_theta",        "NREM Relative Theta Power",      "Relative Theta (NREM)"),
        ("rem_rbp_theta",         "REM Relative Theta Power",       "Relative Theta (REM)"),
        ("wake_td_ratio",         "Wake Theta/Delta Ratio",         "Theta/Delta Ratio (Wake)"),
        ("nrem_td_ratio",         "NREM Theta/Delta Ratio",         "Theta/Delta Ratio (NREM)"),
        ("wake_spectral_entropy", "Wake Spectral Entropy",          "Spectral Entropy (Wake)"),
        ("nrem_spectral_entropy", "NREM Spectral Entropy",          "Spectral Entropy (NREM)"),
        ("wake_lzc",              "Wake Lempel-Ziv Complexity",     "LZC (Wake)"),
        ("nrem_lzc",              "NREM Lempel-Ziv Complexity",     "LZC (NREM)"),
        ("wake_ap_exp",           "Wake Aperiodic Exponent",        "Aperiodic Exponent (Wake)"),
        ("wake_hjorth_mob",       "Wake Hjorth Mobility",           "Hjorth Mobility (Wake)"),
        ("wake_rbp_delta",        "Wake Relative Delta Power",      "Relative Delta (Wake)"),
        ("nrem_rbp_delta",        "NREM Relative Delta Power",      "Relative Delta (NREM)"),
        ("all_rbp_gamma",         "All-state Relative Gamma",       "Relative Gamma (All)"),
        ("wake_total_var",        "Wake Signal Variance",           "Signal Variance (Wake)"),
    ]

    for feat, title, ylabel in key_features:
        if feat not in animal_df.columns:
            continue
        plot_trajectory(animal_df, feat, title, ylabel,
                        f"longitudinal_{feat}.png")

    # ── 4. Effect size heatmap ─────────────────────────────────────────────
    print("\n=== EFFECT SIZE HEATMAPS ===")
    for scen_label, traj in [("scenA", traj_A), ("scenB", traj_B)]:
        plot_effect_size_heatmap(traj, top_n=30, suffix=scen_label)

    # ── 5. PCA trajectory (Scenario A) ────────────────────────────────────
    print("\n=== PCA TRAJECTORY (Scenario A) ===")
    pca_feats = [f for f in feat_cols if f in animal_df.columns
                 and animal_df[f].notna().sum() > 10][:40]
    plot_pca_trajectory(animal_df, pca_feats, suffix="scenA")

    # ── 6. Sleep architecture longitudinal ────────────────────────────────
    if not sleep_df.empty:
        print("\n=== SLEEP ARCHITECTURE LONGITUDINAL ===")
        plot_sleep_architecture_longitudinal(sleep_df)

    # ── 7. Delta analysis (rate of change, Scenario A) ────────────────────
    print("\n=== RATE OF CHANGE ANALYSIS (Scenario A) ===")
    tp_pairs   = list(zip(TP_ORDER[:-1], TP_ORDER[1:]))
    delta_rows = []
    for feat in feat_cols:
        if feat not in animal_df.columns:
            continue
        for tp1, tp2 in tp_pairs:
            for group in ["WT", "KO"]:
                s1 = animal_df[(animal_df.timepoint==tp1)&(animal_df.group==group)][["animal_id",feat]]
                s2 = animal_df[(animal_df.timepoint==tp2)&(animal_df.group==group)][["animal_id",feat]]
                merged = s1.merge(s2, on="animal_id", suffixes=("_t1","_t2"))
                if len(merged) < 3:
                    continue
                delta = (merged[f"{feat}_t2"] - merged[f"{feat}_t1"]).mean()
                delta_rows.append({
                    "feature": feat, "tp1": tp1, "tp2": tp2,
                    "group": group, "mean_delta": delta,
                    "transition": f"{tp1}→{tp2}",
                    "scenario": "A",
                })

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(os.path.join(RESULTS_DIR, "longitudinal_delta_analysis.csv"),
                    index=False)
    print(f"  Saved: longitudinal_delta_analysis.csv ({len(delta_df)} rows)")

    # ── 8. Predictive analysis (Scenario A — 3m features → group) ─────────
    pca_feats_avail = [f for f in pca_feats if f in animal_df.columns]
    pred_df = predictive_feature_analysis(animal_df, pca_feats_avail)
    if not pred_df.empty:
        pred_df.to_csv(os.path.join(RESULTS_DIR, "predictive_feature_scores.csv"),
                       index=False)

    print("\n" + "="*65)
    print("LONGITUDINAL TRAJECTORY ANALYSIS COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("="*65)


if __name__ == "__main__":
    main()
