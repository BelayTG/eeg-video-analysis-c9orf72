"""
10_prespecified_analysis.py
============================
Pre-specified hypothesis testing for the C9orf72 EEG paper.

PRIMARY FEATURES (4, biologically justified a priori):
  1. nrem_td_ratio          — Theta/delta ratio during NREM
                              Rationale: established ALS EEG biomarker;
                              reflects fast-frequency failure
  2. nrem_rbp_theta         — Relative theta power during NREM
                              Rationale: slow oscillation dominance;
                              progressive network collapse marker
  3. spindle_duration_mean  — Sleep spindle duration (seconds)
                              Rationale: thalamocortical integrity;
                              spindle quality > quantity for PV function
  4. wake_rbp_beta          — Relative beta power during Wake
                              Rationale: corticospinal excitability;
                              biphasic hyper→hypo transition marker

PRIMARY TIMEPOINTS (4, excluding known-null 6m):
  3m, 4m, 9m, 12m
  6m excluded from primary analysis (pre-specified null: normalization window)
  7m included as secondary/exploratory only

Statistical approach:
  - Mann-Whitney U test (WT vs KO) per feature per timepoint
  - FDR correction (Benjamini-Hochberg) across 4 × 4 = 16 primary tests
  - Effect size: Cohen's d with 95% bootstrap CI
  - Findings reported as significant if:
      (a) FDR q < 0.05, OR
      (b) 95% bootstrap CI for d excludes zero (effect size evidence)
  - Both Scenario A (animal-level) and Scenario B (session-level)
  - 7m and CA3-CTX coherence reported separately as secondary findings

Outputs:
  - results/prespecified_results.csv
  - results/prespecified_paper_table.csv
  - figures/prespecified_*.png

Run:
    python src/10_prespecified_analysis.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")

COLORS   = {"WT": "#378ADD", "KO": "#D85A30"}
TP_ALL   = ["3m", "4m", "6m", "7m", "9m", "12m"]
TP_X_ALL = [3, 4, 6, 7, 9, 12]

# Primary timepoints (6m excluded as pre-specified null)
TP_PRIMARY = ["3m", "4m", "9m", "12m"]
TP_X_PRI   = [3, 4, 9, 12]

# Secondary timepoints (exploratory only)
TP_SECONDARY = ["6m", "7m"]

# ── Primary features ───────────────────────────────────────────────────────
PRIMARY_FEATURES = [
    {
        "name":      "rem_td_ratio",
        "label":     "REM Theta/Delta Ratio",
        "source":    "band_power",
        "rationale": "REM hyperexcitability; Scekic-Zahirovic PAC state; E/I balance during REM",
        "direction": "+",   # KO > WT expected
    },
    {
        "name":      "rem_rbp_theta",
        "label":     "REM Relative Theta Power",
        "source":    "band_power",
        "rationale": "REM theta dominance; progressive slow-wave invasion of REM",
        "direction": "+",
    },
    {
        "name":      "wake_rbp_beta",
        "label":     "Wake Relative Beta Power",
        "source":    "band_power",
        "rationale": "Corticospinal excitability; biphasic hyper→hypo transition",
        "direction": "±",   # + at 4m, - at 12m
    },
    {
        "name":      "spindle_duration_mean",
        "label":     "Sleep Spindle Duration (s)",
        "source":    "advanced_eeg",
        "rationale": "Thalamocortical circuit quality; PV interneuron integrity",
        "direction": "-",   # KO < WT expected
    },
    {
        "name":      "rem_rbp_beta",
        "label":     "REM Relative Beta Power",
        "source":    "band_power",
        "rationale": "REM beta sign-flip; hyperexcitability (4m) → hypoexcitability (12m)",
        "direction": "±",
    },
]

# Secondary features reported separately
SECONDARY_FEATURES = [
    {
        "name":      "nrem_td_ratio",
        "label":     "NREM Theta/Delta Ratio",
        "source":    "band_power",
        "rationale": "Established ALS EEG biomarker; moved to secondary after REM classifier fix",
        "direction": "+",
    },
    {
        "name":      "nrem_rbp_theta",
        "label":     "NREM Relative Theta Power",
        "source":    "band_power",
        "rationale": "NREM slow oscillation dominance; secondary to REM findings",
        "direction": "+",
    },
    {
        "name":      "ca3_ctx_coh_delta",
        "label":     "CA3-CTX Delta Coherence",
        "source":    "advanced_eeg",
        "rationale": "Hippocampal-cortical connectivity; memory consolidation",
        "direction": "-",
    },
]


# ── FDR correction (BH) ────────────────────────────────────────────────────

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


# ── Bootstrap Cohen's d CI ─────────────────────────────────────────────────

def cohens_d_bootstrap(x, y, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    obs_d = (np.mean(y) - np.mean(x)) / np.sqrt(
        (np.std(x)**2 + np.std(y)**2) / 2 + 1e-10)
    boot_ds = []
    for _ in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        d = (np.mean(by) - np.mean(bx)) / np.sqrt(
            (np.std(bx)**2 + np.std(by)**2) / 2 + 1e-10)
        boot_ds.append(d)
    lo = np.percentile(boot_ds, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_ds, (1 + ci) / 2 * 100)
    return float(obs_d), float(lo), float(hi)


# ── Data loading ───────────────────────────────────────────────────────────

def load_feature(feat_info, tp, scenario="A"):
    """Load one primary feature for one timepoint, return animal-level means."""
    unit_col = "animal_id" if scenario == "A" else "session_id"

    if feat_info["source"] == "band_power":
        path = os.path.join(DATA_DIR,
                            f"state_specific_features_{tp}_{scenario}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df["animal_id"] = df["animal_id"].astype(str)
        feat = feat_info["name"]
        if feat not in df.columns:
            return None
        # Average across files within unit
        uid = unit_col if unit_col in df.columns else "animal_id"
        means = df.groupby([uid, "group"])[feat].mean().reset_index()
        means.columns = ["unit_id", "group", "value"]

    elif feat_info["source"] == "advanced_eeg":
        path = os.path.join(RESULTS_DIR, f"advanced_eeg_{tp}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df["animal_id"] = df["animal_id"].astype(str)
        feat = feat_info["name"]
        if feat not in df.columns:
            return None
        # Advanced EEG: average by animal regardless of scenario
        means = df.groupby(["animal_id", "group"])[feat].mean().reset_index()
        means.columns = ["unit_id", "group", "value"]

    means["timepoint"] = tp
    means["feature"]   = feat_info["name"]
    return means


# ── Run all tests ──────────────────────────────────────────────────────────

def run_prespecified(scenario="A", features=None, timepoints=None, label="PRIMARY"):
    """Run Mann-Whitney + FDR for a given feature/timepoint set."""
    if features is None:
        features = PRIMARY_FEATURES
    if timepoints is None:
        timepoints = TP_PRIMARY

    print(f"\n{'='*65}")
    print(f"{label} ANALYSIS — Scenario {scenario}")
    print(f"Features: {len(features)} | Timepoints: {timepoints}")
    print(f"Total tests: {len(features)} × {len(timepoints)} = "
          f"{len(features)*len(timepoints)}")
    print(f"{'='*65}")

    all_data = {}
    rows = []

    for feat_info in features:
        for tp in timepoints:
            data = load_feature(feat_info, tp, scenario)
            if data is None or len(data) < 4:
                continue

            all_data[(feat_info["name"], tp)] = data

            wt = data[data.group == "WT"]["value"].dropna().values
            ko = data[data.group == "KO"]["value"].dropna().values

            if len(wt) < 2 or len(ko) < 2:
                continue

            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            d, d_lo, d_hi = cohens_d_bootstrap(wt, ko)
            ci_excludes_zero = (d_lo > 0) or (d_hi < 0)

            rows.append({
                "feature":          feat_info["name"],
                "label":            feat_info["label"],
                "timepoint":        tp,
                "scenario":         scenario,
                "analysis":         label,
                "wt_mean":          round(float(np.mean(wt)), 6),
                "wt_sd":            round(float(np.std(wt)),  6),
                "wt_sem":           round(float(np.std(wt) / np.sqrt(len(wt))), 6),
                "ko_mean":          round(float(np.mean(ko)), 6),
                "ko_sd":            round(float(np.std(ko)),  6),
                "ko_sem":           round(float(np.std(ko) / np.sqrt(len(ko))), 6),
                "cohens_d":         round(d,    4),
                "d_ci_lo":          round(d_lo, 4),
                "d_ci_hi":          round(d_hi, 4),
                "ci_excludes_zero": ci_excludes_zero,
                "pval":             round(p, 6),
                "n_wt":             len(wt),
                "n_ko":             len(ko),
            })

    if not rows:
        print("  No data found")
        return pd.DataFrame(), {}

    results = pd.DataFrame(rows)

    # FDR across this test set only
    results["pval_fdr"] = fdr_bh(results["pval"].values).round(6)
    results["fdr_sig"]  = results["pval_fdr"] < 0.05
    results["sig_raw"]  = results["pval"].apply(
        lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns")
    results["sig_fdr"]  = results["pval_fdr"].apply(
        lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns")

    # Print
    print(f"\n{'Feature':<28} {'TP':<5} {'WT':>10} {'KO':>10} "
          f"{'d':>7} {'95%CI':>16} {'p_raw':>9} {'p_FDR':>9} {'note'}")
    print("-" * 105)

    for feat_info in features:
        feat = feat_info["name"]
        sub  = results[results.feature == feat].sort_values(
            "timepoint", key=lambda s: s.map(
                {t: i for i, t in enumerate(TP_ALL)}))
        print(f"\n  {feat_info['label']}")
        for _, r in sub.iterrows():
            notes = []
            if r.fdr_sig:
                notes.append("FDR*")
            if r.ci_excludes_zero:
                notes.append("CI✓")
            note_str = " ".join(notes)
            print(f"  {'':26} {r.timepoint:<5} {r.wt_mean:>10.5f} "
                  f"{r.ko_mean:>10.5f} {r.cohens_d:>7.3f} "
                  f"[{r.d_ci_lo:>6.3f},{r.d_ci_hi:>6.3f}] "
                  f"{r.pval:>9.5f} {r.pval_fdr:>9.5f}  {note_str}")

    # Summary
    fdr_sig   = results[results.fdr_sig]
    ci_sig    = results[results.ci_excludes_zero & ~results.fdr_sig]
    nom_sig   = results[(results.pval < 0.05) & ~results.fdr_sig & ~results.ci_excludes_zero]

    n_tests = len(results)
    print(f"\n{'='*65}")
    print(f"SUMMARY — {label} | Scenario {scenario} | {n_tests} tests")
    print(f"{'='*65}")
    print(f"  FDR significant (q<0.05):                {len(fdr_sig)}")
    print(f"  Effect size evidence (CI excludes zero): {len(ci_sig) + len(fdr_sig)}")
    print(f"  Nominally significant only (p<0.05):     {len(nom_sig)}")

    if len(fdr_sig) > 0:
        print(f"\n  FDR-SIGNIFICANT FINDINGS:")
        for _, r in fdr_sig.iterrows():
            print(f"    *** [{r.timepoint}] {r.label}: "
                  f"d={r.cohens_d:.3f} [{r.d_ci_lo:.3f},{r.d_ci_hi:.3f}] "
                  f"p={r.pval:.5f} FDR_p={r.pval_fdr:.5f}")

    if len(ci_sig) > 0:
        print(f"\n  CI EXCLUDES ZERO (effect size evidence, FDR not met):")
        for _, r in ci_sig.sort_values("pval").iterrows():
            print(f"    ◄ [{r.timepoint}] {r.label}: "
                  f"d={r.cohens_d:.3f} [{r.d_ci_lo:.3f},{r.d_ci_hi:.3f}] "
                  f"p={r.pval:.5f} FDR_p={r.pval_fdr:.5f}")

    return results, all_data


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_primary_trajectories(results_A, results_B, all_data):
    """
    Trajectory figure per primary feature across ALL timepoints.
    Primary timepoints shown with solid markers, secondary with open markers.
    """
    for feat_info in PRIMARY_FEATURES:
        feat   = feat_info["name"]
        sub_A  = results_A[results_A.feature == feat] if len(results_A) > 0 else pd.DataFrame()

        fig = plt.figure(figsize=(14, 5))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)
        ax_traj = fig.add_subplot(gs[0])
        ax_eff  = fig.add_subplot(gs[1])

        # Collect data for all timepoints
        tp_avail = [tp for tp in TP_ALL if (feat, tp) in all_data]
        tp_x_avail = [TP_X_ALL[TP_ALL.index(t)] for t in tp_avail]

        for group, color in COLORS.items():
            means, sems, xs = [], [], []
            for tp, tp_x in zip(tp_avail, tp_x_avail):
                data = all_data.get((feat, tp))
                if data is None:
                    continue
                vals = data[data.group == group]["value"].dropna()
                if len(vals) == 0:
                    continue
                means.append(vals.mean())
                sems.append(vals.sem())
                xs.append(tp_x)
                # Individual points
                marker = "o" if tp in TP_PRIMARY else "^"
                ax_traj.scatter([tp_x] * len(vals), vals,
                                 color=color, s=25, alpha=0.35,
                                 marker=marker, zorder=3)

            if not xs:
                continue
            ax_traj.fill_between(xs,
                                  np.array(means) - np.array(sems),
                                  np.array(means) + np.array(sems),
                                  color=color, alpha=0.15)
            ax_traj.plot(xs, means, "-o", color=color, lw=2.5,
                          markersize=8, label=group, zorder=4)
            ax_traj.errorbar(xs, means, yerr=sems, fmt="none",
                              color=color, capsize=5, capthick=2)

        # Mark primary vs secondary timepoints
        for tp, tp_x in zip(tp_avail, tp_x_avail):
            ax_traj.axvline(tp_x, color="gray", lw=0.5, alpha=0.3,
                             ls="--" if tp in TP_SECONDARY else "-")

        # Significance markers from primary results
        if len(sub_A) > 0:
            ymax = ax_traj.get_ylim()[1] if ax_traj.get_ylim()[1] != 0 else 1
            for _, r in sub_A.iterrows():
                if r.timepoint not in tp_avail:
                    continue
                tp_x = TP_X_ALL[TP_ALL.index(r.timepoint)]
                if r.fdr_sig:
                    ax_traj.text(tp_x, ymax*0.97, "***",
                                  ha="center", fontsize=11,
                                  color="darkred", weight="bold")
                elif r.ci_excludes_zero:
                    ax_traj.text(tp_x, ymax*0.97, "◄",
                                  ha="center", fontsize=10, color="#D85A30")
                elif r.pval < 0.05:
                    ax_traj.text(tp_x, ymax*0.97, "†",
                                  ha="center", fontsize=11, color="gray")

        ax_traj.set_xticks(tp_x_avail)
        ax_traj.set_xticklabels(
            [f"{t}{'*' if t in TP_PRIMARY else ''}" for t in tp_avail],
            fontsize=9)
        ax_traj.set_xlabel("Age (months) [* = primary timepoint]", fontsize=10)
        ax_traj.set_ylabel(feat_info["label"], fontsize=11)
        ax_traj.legend(fontsize=10)
        ax_traj.set_title(f"{feat_info['label']}\n"
                           "† p<0.05  ◄ CI excludes 0  *** FDR*",
                           fontsize=10)

        # Effect size panel (primary timepoints only)
        ds, d_los, d_his, tp_labels = [], [], [], []
        for tp in TP_PRIMARY:
            data = all_data.get((feat, tp))
            if data is None:
                continue
            wt = data[data.group=="WT"]["value"].dropna().values
            ko = data[data.group=="KO"]["value"].dropna().values
            if len(wt) < 2 or len(ko) < 2:
                continue
            d, d_lo, d_hi = cohens_d_bootstrap(wt, ko)
            ds.append(d); d_los.append(d_lo); d_his.append(d_hi)
            tp_labels.append(tp)

        if ds:
            colors_d = ["#D85A30" if d > 0 else "#378ADD" for d in ds]
            ax_eff.barh(range(len(tp_labels)), ds, color=colors_d,
                         alpha=0.75, height=0.6)
            ax_eff.errorbar(ds, range(len(tp_labels)),
                             xerr=[np.array(ds)-np.array(d_los),
                                   np.array(d_his)-np.array(ds)],
                             fmt="none", color="black",
                             capsize=4, capthick=1.5)
            ax_eff.axvline(0,    color="black", lw=1.2)
            ax_eff.axvline( 0.8, color="gray", lw=1, ls="--", alpha=0.5)
            ax_eff.axvline(-0.8, color="gray", lw=1, ls="--", alpha=0.5)
            ax_eff.set_yticks(range(len(tp_labels)))
            ax_eff.set_yticklabels(tp_labels, fontsize=10)
            ax_eff.set_xlabel("Cohen's d (KO − WT)", fontsize=10)
            ax_eff.set_title("Effect size ± 95% CI\n(primary TPs only)", fontsize=9)

            # Star annotations
            if len(sub_A) > 0:
                for i, tp in enumerate(tp_labels):
                    r_sub = sub_A[sub_A.timepoint == tp]
                    if len(r_sub) == 0:
                        continue
                    r = r_sub.iloc[0]
                    if r.fdr_sig:
                        ax_eff.text(ds[i] + 0.05, i, "***",
                                     va="center", fontsize=9, color="darkred")
                    elif r.ci_excludes_zero:
                        ax_eff.text(ds[i] + 0.05, i, "◄",
                                     va="center", fontsize=9, color="#D85A30")

        fig.suptitle(f"{feat_info['label']} — C9orf72 KO vs WT",
                      fontsize=11, weight="bold")
        plt.tight_layout()
        fname = f"prespecified_{feat}.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


def plot_summary_heatmap(results_A, results_B):
    """Summary heatmap: 4 features × 4 primary timepoints."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, results, sc_label in zip(
            axes,
            [results_A, results_B],
            ["Scenario A (animal-level, n=animals)",
             "Scenario B (session-level, n=sessions)"]):
        if results is None or len(results) == 0:
            continue

        feat_labels = [f["label"] for f in PRIMARY_FEATURES]
        feat_names  = [f["name"]  for f in PRIMARY_FEATURES]
        tp_avail    = [t for t in TP_PRIMARY
                       if t in results["timepoint"].values]

        matrix   = np.full((len(feat_names), len(tp_avail)), np.nan)
        pmat     = np.ones_like(matrix)
        fdr_mat  = np.zeros_like(matrix, dtype=bool)
        ci_mat   = np.zeros_like(matrix, dtype=bool)

        for i, feat in enumerate(feat_names):
            for j, tp in enumerate(tp_avail):
                sub = results[(results.feature == feat) &
                               (results.timepoint == tp)]
                if len(sub) > 0:
                    matrix[i, j]  = sub["cohens_d"].values[0]
                    pmat[i, j]    = sub["pval"].values[0]
                    fdr_mat[i, j] = sub["fdr_sig"].values[0]
                    ci_mat[i, j]  = sub["ci_excludes_zero"].values[0]

        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                        vmin=-2.5, vmax=2.5)
        plt.colorbar(im, ax=ax, label="Cohen's d (KO − WT)", shrink=0.8)

        for i in range(len(feat_names)):
            for j in range(len(tp_avail)):
                if np.isnan(matrix[i, j]):
                    continue
                # Annotation
                if fdr_mat[i, j]:
                    ax.text(j, i-0.2, "***", ha="center", va="center",
                             fontsize=11, color="black", weight="bold")
                elif ci_mat[i, j]:
                    ax.text(j, i-0.2, "◄", ha="center", va="center",
                             fontsize=10, color="black")
                elif pmat[i, j] < 0.05:
                    ax.text(j, i-0.2, "†", ha="center", va="center",
                             fontsize=11, color="black")
                ax.text(j, i+0.25, f"d={matrix[i,j]:.2f}",
                         ha="center", va="center", fontsize=7,
                         color="black", alpha=0.8)

        ax.set_xticks(range(len(tp_avail)))
        ax.set_xticklabels(tp_avail, fontsize=11)
        ax.set_yticks(range(len(feat_labels)))
        ax.set_yticklabels(feat_labels, fontsize=9)
        ax.set_xlabel("Primary Timepoints", fontsize=10)
        ax.set_title(f"{sc_label}\n"
                      "† p<0.05  ◄ CI excl. 0  *** FDR q<0.05",
                      fontsize=9)

    fig.suptitle("Pre-specified EEG Features — C9orf72-KO vs WT\n"
                  "16 primary tests (4 features × 4 timepoints)",
                  fontsize=11, weight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "prespecified_summary_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: prespecified_summary_heatmap.png")


def build_paper_table(results_A, results_B):
    rows = []
    for feat_info in PRIMARY_FEATURES:
        feat = feat_info["name"]
        for tp in TP_PRIMARY:
            r_A = results_A[(results_A.feature == feat) &
                             (results_A.timepoint == tp)]
            r_B = results_B[(results_B.feature == feat) &
                             (results_B.timepoint == tp)] \
                  if results_B is not None else pd.DataFrame()
            if len(r_A) == 0:
                continue
            r = r_A.iloc[0]
            row = {
                "Feature":        feat_info["label"],
                "Timepoint":      tp,
                "WT mean±SD":     f"{r.wt_mean:.4f}±{r.wt_sd:.4f}",
                "KO mean±SD":     f"{r.ko_mean:.4f}±{r.ko_sd:.4f}",
                "n_WT":           int(r.n_wt),
                "n_KO":           int(r.n_ko),
                "Cohen's d":      f"{r.cohens_d:.3f} [{r.d_ci_lo:.3f},{r.d_ci_hi:.3f}]",
                "p (raw)":        f"{r.pval:.5f}",
                "p (FDR)":        f"{r.pval_fdr:.5f}",
                "CI excludes 0":  "Yes" if r.ci_excludes_zero else "No",
                "Sig (FDR)":      r.sig_fdr,
            }
            if len(r_B) > 0:
                rb = r_B.iloc[0]
                row["p_FDR_ScB"]   = f"{rb.pval_fdr:.5f}"
                row["Sig_FDR_ScB"] = "Yes" if rb.fdr_sig else "No"
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "prespecified_paper_table.csv"),
                  index=False)
    print(f"  Saved: prespecified_paper_table.csv")
    return table


def plot_beta_signflip(all_data):
    """
    Dedicated figure showing the beta sign-flip across timepoints.
    This is the most novel single finding — deserves its own figure.
    """
    feat = "wake_rbp_beta"
    tp_avail = [tp for tp in TP_ALL if (feat, tp) in all_data]
    if len(tp_avail) < 3:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: raw values trajectory
    ax = axes[0]
    for group, color in COLORS.items():
        means, sems, xs = [], [], []
        for tp in tp_avail:
            data = all_data.get((feat, tp))
            if data is None:
                continue
            vals = data[data.group == group]["value"].dropna()
            if len(vals) == 0:
                continue
            tp_x = TP_X_ALL[TP_ALL.index(tp)]
            means.append(vals.mean())
            sems.append(vals.sem())
            xs.append(tp_x)
            # Individual animals
            ax.scatter([tp_x] * len(vals), vals,
                        color=color, s=30, alpha=0.4, zorder=3)

        ax.fill_between(xs, np.array(means)-np.array(sems),
                         np.array(means)+np.array(sems),
                         color=color, alpha=0.15)
        ax.plot(xs, means, "-o", color=color, lw=2.5,
                 markersize=8, label=group, zorder=4)
        ax.errorbar(xs, means, yerr=sems, fmt="none",
                     color=color, capsize=5, capthick=2)

    # Shade the sign-flip zone
    ax.axvspan(4.5, 8.5, alpha=0.06, color="gray",
                label="Normalization window")
    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.3)
    ax.set_xticks([TP_X_ALL[TP_ALL.index(t)] for t in tp_avail])
    ax.set_xticklabels(tp_avail, fontsize=10)
    ax.set_xlabel("Age (months)", fontsize=11)
    ax.set_ylabel("Wake Relative Beta Power", fontsize=11)
    ax.set_title("Biphasic Beta Trajectory\n"
                  "Hyperexcitability (4m) → Hypoexcitability (12m)",
                  fontsize=10)
    ax.legend(fontsize=10)

    # Right: Cohen's d trajectory with zero line
    ax2 = axes[1]
    ds, d_los, d_his, xs2 = [], [], [], []
    for tp in tp_avail:
        data = all_data.get((feat, tp))
        if data is None:
            continue
        wt = data[data.group == "WT"]["value"].dropna().values
        ko = data[data.group == "KO"]["value"].dropna().values
        if len(wt) < 2 or len(ko) < 2:
            continue
        d, d_lo, d_hi = cohens_d_bootstrap(wt, ko)
        ds.append(d); d_los.append(d_lo); d_his.append(d_hi)
        xs2.append(TP_X_ALL[TP_ALL.index(tp)])

    colors_d = ["#D85A30" if d > 0 else "#378ADD" for d in ds]
    ax2.axhline(0, color="black", lw=1.5)
    ax2.axhline( 0.8, color="gray", lw=1, ls="--", alpha=0.5)
    ax2.axhline(-0.8, color="gray", lw=1, ls="--", alpha=0.5)
    for x, d, lo, hi, c in zip(xs2, ds, d_los, d_his, colors_d):
        ax2.plot([x, x], [lo, hi], color=c, lw=2.5, solid_capstyle="round")
        ax2.scatter(x, d, color=c, s=80, zorder=5)
    ax2.fill_between(xs2, d_los, d_his, alpha=0.12, color="gray")
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(tp_avail, fontsize=10)
    ax2.set_xlabel("Age (months)", fontsize=11)
    ax2.set_ylabel("Cohen's d (KO − WT)", fontsize=11)
    ax2.set_title("Effect Size Trajectory\n"
                   "95% bootstrap CI | dashed = |d|=0.8",
                   fontsize=10)

    fig.suptitle("Wake Relative Beta Power — Biphasic Sign Flip\n"
                  "C9orf72-KO: Early Hyperexcitability → Late Hypoexcitability",
                  fontsize=11, weight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "prespecified_beta_signflip.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: prespecified_beta_signflip.png")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("PRE-SPECIFIED HYPOTHESIS TESTING — REVISED v2")
    print("5 REM/Wake features × 4 timepoints = 20 primary tests")
    print("REM features added after classifier fix (v1 had NREM only)")
    print("FDR: Benjamini-Hochberg across 20 tests")
    print("=" * 65)
    print("\nPrimary features:")
    for i, f in enumerate(PRIMARY_FEATURES, 1):
        print(f"  {i}. {f['label']} — {f['rationale']}")
    print(f"\nPrimary timepoints: {TP_PRIMARY}")
    print(f"Secondary timepoints: {TP_SECONDARY}")

    all_results = []

    # ── PRIMARY ANALYSIS ──────────────────────────────────────────────────
    res_A, all_data_A = run_prespecified("A", PRIMARY_FEATURES,
                                          TP_PRIMARY, "PRIMARY")
    res_B, _          = run_prespecified("B", PRIMARY_FEATURES,
                                          TP_PRIMARY, "PRIMARY")
    all_results.extend([res_A, res_B])

    # ── SECONDARY: 6m + 7m (exploratory, uncorrected) ────────────────────
    print("\n" + "="*65)
    print("SECONDARY / EXPLORATORY (6m + 7m, uncorrected, no FDR claim)")
    print("="*65)
    res_sec_A, all_data_sec = run_prespecified(
        "A", PRIMARY_FEATURES + SECONDARY_FEATURES,
        TP_SECONDARY, "SECONDARY")
    res_sec_B, _ = run_prespecified(
        "B", PRIMARY_FEATURES + SECONDARY_FEATURES,
        TP_SECONDARY, "SECONDARY")
    all_results.extend([res_sec_A, res_sec_B])

    # ── Save all results ──────────────────────────────────────────────────
    combined = pd.concat([r for r in all_results if len(r) > 0],
                          ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "prespecified_results.csv"),
                    index=False)
    print(f"\nSaved: prespecified_results.csv ({len(combined)} rows)")

    # ── Paper table (primary only, Scenario A) ───────────────────────────
    print("\n=== PAPER-READY TABLE (Primary, Scenario A) ===")
    table = build_paper_table(res_A, res_B)
    cols  = ["Feature","Timepoint","WT mean±SD","KO mean±SD",
             "Cohen's d","p (raw)","p (FDR)","CI excludes 0","Sig (FDR)"]
    print(table[cols].to_string(index=False))

    # ── Figures ───────────────────────────────────────────────────────────
    print("\n=== GENERATING FIGURES ===")

    # Trajectory plots across ALL timepoints (primary + secondary)
    all_data_combined = {**all_data_A, **all_data_sec}
    plot_primary_trajectories(res_A, res_B, all_data_combined)

    # Summary heatmap — primary timepoints
    plot_summary_heatmap(res_A, res_B)

    # Beta sign-flip figure (special)
    plot_beta_signflip(all_data_combined)

    print("\n" + "="*65)
    print("PRE-SPECIFIED ANALYSIS COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("="*65)


if __name__ == "__main__":
    main()
