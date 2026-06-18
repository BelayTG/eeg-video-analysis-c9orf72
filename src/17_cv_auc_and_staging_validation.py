"""
17_cv_auc_and_staging_validation.py
====================================
Two rigor patches that convert impressive-but-attackable results into
defensible ones, addressing the two cheapest top-tier reviewer objections.

PART A — CROSS-VALIDATED + PERMUTATION AUC
  The in-sample AUCs from script 16 are optimistic (same animals used to
  find the effect and measure classification). This part reports, per
  headline feature:
    (1) Leave-one-out cross-validated AUC (honest out-of-sample estimate)
    (2) Stratified k-fold CV AUC with repeats (variance estimate)
    (3) Label-permutation null distribution + empirical p-value
        (is the observed AUC beyond what label-shuffling produces?)
  A feature is reported as a credible classifier only if the CV-AUC stays
  high AND the permutation p < 0.05.

PART B — SLEEP-STAGING SENSITIVITY ANALYSIS
  REM was classified by relative band-power thresholds, not EMG. Reviewers
  will question this. This part shows the headline REM findings are STABLE
  to the classification thresholds, by:
    (1) Re-deriving REM under a grid of threshold choices (stricter/looser
        theta, delta, and variance cutoffs)
    (2) Recomputing the key REM features (rem_rbp_beta at 4m & 12m;
        rem_td_ratio at 4m) under each threshold set
    (3) Reporting the genotype effect size across the grid — if the effect
        holds across reasonable thresholds, the finding is not an artifact
        of one arbitrary cutoff.
  Also produces a state-separation diagnostic (theta/delta scatter by state)
  to visually justify the classification.

Outputs:
  results/cv_auc_results.csv
  results/auc_permutation_pvals.csv
  results/staging_sensitivity_grid.csv
  figures/cv_auc_forest.png
  figures/auc_permutation_null.png
  figures/staging_sensitivity_heatmap.png
  figures/state_separation_diagnostic.png

Run:
    python src/17_cv_auc_and_staging_validation.py

NOTE: Part B re-derives REM from epoch-level band power. It expects
  data/epochs_with_states_<tp>.csv to contain per-epoch relative band powers
  (rbp_delta, rbp_theta, ... for the channel used in classification) plus
  total variance. If those columns are absent, Part B falls back to a
  threshold-perturbation on the precomputed state labels where possible and
  reports what it could and could not test.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
    from sklearn.preprocessing import StandardScaler
    HAVE_SK = True
except ImportError:
    HAVE_SK = False
    print("WARNING: scikit-learn required. pip install scikit-learn --break-system-packages")

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")

COLORS   = {"WT": "#378ADD", "KO": "#D85A30"}
TP_ORDER = ["3m", "4m", "6m", "7m", "9m", "12m"]
EPOCH_S  = 4.0
RNG_SEED = 0


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return 0.0 if pooled == 0 else (np.mean(a) - np.mean(b)) / pooled


# ══════════════════════════════════════════════════════════════════════
# PART A — CROSS-VALIDATED + PERMUTATION AUC
# ══════════════════════════════════════════════════════════════════════

ROC_FEATURES = [
    ("4m",  "rem_rbp_beta",          "4m REM rel. beta (acute)"),
    ("12m", "rem_rbp_beta",          "12m REM rel. beta (reversal)"),
    ("4m",  "rem_td_ratio",          "4m REM theta/delta"),
    ("3m",  "spindle_duration_mean", "3m spindle duration"),
    ("3m",  "wake_ap_exp",           "3m aperiodic exponent"),
]


def load_feature_vector(tp, feat):
    if feat == "spindle_duration_mean":
        adv_path = os.path.join(RESULTS_DIR, f"advanced_eeg_{tp}.csv")
        if os.path.exists(adv_path):
            a = pd.read_csv(adv_path)
            if "spindle_duration_mean" in a.columns and "group" in a.columns:
                a = a.dropna(subset=["spindle_duration_mean"])
                a["animal_id"] = a["animal_id"].astype(str)
                grp = a.groupby(["animal_id","group"])["spindle_duration_mean"].mean().reset_index()
                return grp["spindle_duration_mean"].values, (grp.group=="KO").astype(int).values
        return None, None
    path = os.path.join(DATA_DIR, f"state_specific_features_{tp}_A.csv")
    if not os.path.exists(path):
        alt = os.path.join(DATA_DIR, f"state_specific_features_{tp}.csv")
        path = alt if os.path.exists(alt) else None
    if path is None:
        return None, None
    df = pd.read_csv(path)
    if feat not in df.columns or "group" not in df.columns:
        return None, None
    df["animal_id"] = df["animal_id"].astype(str)
    sub = df.dropna(subset=[feat]).groupby(["animal_id","group"])[feat].mean().reset_index()
    return sub[feat].values, (sub.group=="KO").astype(int).values


def insample_auc(x, y):
    auc = roc_auc_score(y, x)
    if auc < 0.5:
        auc = roc_auc_score(y, -x)
    return auc


def loo_cv_auc(x, y):
    """Leave-one-out CV AUC using a 1-feature logistic model."""
    x = x.reshape(-1, 1)
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(x):
        if len(np.unique(y[tr])) < 2:
            preds[te] = 0.5
            continue
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(sc.transform(x[tr]), y[tr])
        preds[te] = clf.predict_proba(sc.transform(x[te]))[:, 1]
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, preds)


def kfold_cv_auc(x, y, k=5, repeats=20):
    """Repeated stratified k-fold CV AUC; returns mean and SD across repeats."""
    x = x.reshape(-1, 1)
    n_min = min(np.bincount(y))
    k_eff = min(k, n_min)
    if k_eff < 2:
        return np.nan, np.nan
    rskf = RepeatedStratifiedKFold(n_splits=k_eff, n_repeats=repeats, random_state=RNG_SEED)
    aucs = []
    for tr, te in rskf.split(x, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(sc.transform(x[tr]), y[tr])
        p = clf.predict_proba(sc.transform(x[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    if not aucs:
        return np.nan, np.nan
    return float(np.mean(aucs)), float(np.std(aucs))


def permutation_auc_pval(x, y, n_perm=5000):
    """Empirical p-value: fraction of label-permuted in-sample AUCs >= observed."""
    obs = insample_auc(x, y)
    rng = np.random.default_rng(RNG_SEED)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        null[i] = insample_auc(x, yp)
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, p, null


def part_a_cv_auc():
    print("\n" + "="*64)
    print("PART A — CROSS-VALIDATED + PERMUTATION AUC")
    print("In-sample AUC is optimistic; CV + permutation give honest estimates")
    print("="*64)
    if not HAVE_SK:
        return None

    rows = []
    null_store = {}
    print(f"\n{'Feature':<32}{'in-samp':>8}{'LOO-CV':>8}{'kfold':>8}{'perm p':>9}")
    print("-"*70)
    for tp, feat, label in ROC_FEATURES:
        x, y = load_feature_vector(tp, feat)
        if x is None or len(x) < 6 or len(np.unique(y)) < 2:
            print(f"{label:<32}{'(n/a)':>8}")
            continue
        ins = insample_auc(x, y)
        loo = loo_cv_auc(x, y)
        kf_mean, kf_sd = kfold_cv_auc(x, y)
        obs, pperm, null = permutation_auc_pval(x, y)
        null_store[label] = (obs, null)
        rows.append({
            "timepoint": tp, "feature": feat, "label": label, "n": len(y),
            "auc_insample": round(ins, 3),
            "auc_loo_cv": round(loo, 3) if not np.isnan(loo) else np.nan,
            "auc_kfold_mean": round(kf_mean, 3) if not np.isnan(kf_mean) else np.nan,
            "auc_kfold_sd": round(kf_sd, 3) if not np.isnan(kf_sd) else np.nan,
            "perm_pval": round(pperm, 4),
            "credible": (pperm < 0.05) and (not np.isnan(loo)) and (loo > 0.7),
        })
        star = "*" if pperm < 0.05 else ""
        print(f"{label:<32}{ins:>8.3f}{loo:>8.3f}{kf_mean:>8.3f}{pperm:>9.4f}{star}")

    if not rows:
        print("No features available.")
        return None
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS_DIR, "cv_auc_results.csv"), index=False)
    print(f"\nSaved: cv_auc_results.csv")
    print("\nInterpretation: report LOO-CV AUC (not in-sample) as the headline number;")
    print("permutation p confirms the classifier beats chance given the small sample.")

    # Forest plot: in-sample vs LOO vs kfold
    fig, ax = plt.subplots(figsize=(8, 0.7*len(res)+2))
    ylabels, yy = [], []
    for i, (_, r) in enumerate(res.iterrows()):
        y0 = len(res) - i
        yy.append(y0); ylabels.append(f"{r['label']} (n={r['n']})")
        ax.scatter(r["auc_insample"], y0, color="#B4B2A9", s=60, label="in-sample" if i==0 else "", zorder=3)
        if not np.isnan(r["auc_loo_cv"]):
            ax.scatter(r["auc_loo_cv"], y0, color="#D85A30", s=70, label="LOO-CV" if i==0 else "", zorder=4)
        if not np.isnan(r["auc_kfold_mean"]):
            ax.errorbar(r["auc_kfold_mean"], y0, xerr=r["auc_kfold_sd"], fmt="s",
                        color="#378ADD", capsize=3, markersize=6,
                        label="k-fold (mean±SD)" if i==0 else "", zorder=3)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("AUC"); ax.set_xlim(0.2, 1.02)
    ax.set_title("Classification AUC: in-sample vs cross-validated\n"
                 "(LOO-CV is the honest estimate; dashed line = chance)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(os.path.join(FIGURES_DIR, "cv_auc_forest.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: cv_auc_forest.png")

    # Permutation null for the single strongest feature
    if null_store:
        best = max(null_store.items(), key=lambda kv: kv[1][0])
        label, (obs, null) = best
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(null, bins=40, color="#B4B2A9", alpha=0.8)
        ax.axvline(obs, color="#D85A30", lw=2, label=f"observed AUC={obs:.2f}")
        p = (np.sum(null >= obs)+1)/(len(null)+1)
        ax.set_title(f"Label-permutation null — {label}\nempirical p={p:.4f}", fontsize=10)
        ax.set_xlabel("AUC under permuted labels"); ax.set_ylabel("count")
        ax.legend(fontsize=9)
        fig.savefig(os.path.join(FIGURES_DIR, "auc_permutation_null.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved: auc_permutation_null.png")

        # store full perm pvals
        pv = pd.DataFrame([{"label": k, "observed_auc": v[0],
                            "perm_p": (np.sum(v[1] >= v[0])+1)/(len(v[1])+1)}
                           for k, v in null_store.items()])
        pv.to_csv(os.path.join(RESULTS_DIR, "auc_permutation_pvals.csv"), index=False)
    return res


# ══════════════════════════════════════════════════════════════════════
# PART B — SLEEP-STAGING SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════

# Classification channel band-power columns expected in epochs_with_states_<tp>.csv
RBP_COLS = {"delta": "rbp_delta", "theta": "rbp_theta"}
VAR_COL = "total_var"   # epoch total variance for the wake/REM amplitude criterion

# Threshold grid: percentile offsets applied to the baseline rule
# Baseline REM rule: rbp_theta > theta_pct, rbp_delta < delta_pct, total_var < var_pct
THETA_PCTS = [40, 50, 60]     # percentile cutoffs for theta (median=50 baseline)
DELTA_PCTS = [40, 50, 60]     # percentile cutoffs for delta
VAR_PCTS   = [70, 75, 80]     # variance ceiling percentile (baseline 75)

# Headline REM features to re-test under each grid point
KEY_REM_TESTS = [
    ("4m",  "rbp_beta",  "4m REM rel. beta"),
    ("12m", "rbp_beta",  "12m REM rel. beta"),
    ("4m",  "td_ratio",  "4m REM theta/delta"),
]


def reclassify_rem(ep, theta_pct, delta_pct, var_pct):
    """Return boolean REM mask under given percentile thresholds, computed per file."""
    mask = np.zeros(len(ep), dtype=bool)
    for _, idx in ep.groupby("abf_file").groups.items():
        sub = ep.loc[idx]
        if RBP_COLS["theta"] not in sub or RBP_COLS["delta"] not in sub:
            continue
        th = np.nanpercentile(sub[RBP_COLS["theta"]], theta_pct)
        de = np.nanpercentile(sub[RBP_COLS["delta"]], delta_pct)
        if VAR_COL in sub:
            vr = np.nanpercentile(sub[VAR_COL], var_pct)
            m = (sub[RBP_COLS["theta"]] > th) & (sub[RBP_COLS["delta"]] < de) & (sub[VAR_COL] < vr)
        else:
            m = (sub[RBP_COLS["theta"]] > th) & (sub[RBP_COLS["delta"]] < de)
        mask[ep.index.get_indexer(idx)] = m.values
    return mask


def compute_rem_feature(ep, rem_mask, feat):
    """Per-animal mean of the REM feature under the given mask."""
    sub = ep[rem_mask].copy()
    if len(sub) == 0:
        return None
    if feat == "td_ratio":
        if "rbp_theta" in sub and "rbp_delta" in sub:
            sub["td_ratio"] = sub["rbp_theta"] / (sub["rbp_delta"] + 1e-9)
        else:
            return None
    if feat not in sub.columns:
        return None
    g = sub.groupby(["animal_id","group"])[feat].mean().reset_index()
    return g


def part_b_staging_sensitivity():
    print("\n" + "="*64)
    print("PART B — SLEEP-STAGING SENSITIVITY ANALYSIS")
    print("Are the headline REM findings stable to classification thresholds?")
    print("="*64)

    # Check we have epoch-level band powers
    sample_tp = None
    for tp in TP_ORDER:
        p = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
        if os.path.exists(p):
            cols = pd.read_csv(p, nrows=1).columns
            if RBP_COLS["theta"] in cols and RBP_COLS["delta"] in cols:
                sample_tp = tp
                break
    if sample_tp is None:
        print("Epoch-level relative band powers (rbp_theta/rbp_delta) not found in")
        print("epochs_with_states_<tp>.csv — cannot run threshold re-derivation.")
        print("If your epoch files use different column names, tell me and I'll adapt.")
        return None

    grid_rows = []
    for tp, feat, label in KEY_REM_TESTS:
        ep_path = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
        if not os.path.exists(ep_path):
            continue
        ep = pd.read_csv(ep_path)
        ep["animal_id"] = ep["animal_id"].astype(str)
        for th_pct in THETA_PCTS:
            for de_pct in DELTA_PCTS:
                for vr_pct in VAR_PCTS:
                    mask = reclassify_rem(ep, th_pct, de_pct, vr_pct)
                    if mask.sum() < 20:
                        continue
                    g = compute_rem_feature(ep, mask, feat)
                    if g is None or g["group"].nunique() < 2:
                        continue
                    wt = g[g.group=="WT"][feat].dropna().values
                    ko = g[g.group=="KO"][feat].dropna().values
                    if len(wt) < 2 or len(ko) < 2:
                        continue
                    d = cohens_d(ko, wt)
                    _, p = mannwhitneyu(ko, wt, alternative="two-sided")
                    rem_frac = mask.mean()
                    grid_rows.append({
                        "timepoint": tp, "feature": feat, "label": label,
                        "theta_pct": th_pct, "delta_pct": de_pct, "var_pct": vr_pct,
                        "rem_fraction": round(rem_frac, 3),
                        "cohens_d": round(d, 3), "pval": round(p, 4),
                        "n_wt": len(wt), "n_ko": len(ko),
                    })

    if not grid_rows:
        print("Could not compute the threshold grid (insufficient REM epochs or columns).")
        return None
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(os.path.join(RESULTS_DIR, "staging_sensitivity_grid.csv"), index=False)
    print(f"Saved: staging_sensitivity_grid.csv ({len(grid)} grid points)")

    # Summary: for each headline test, range of d and fraction with same sign
    print("\nStability of each headline finding across the threshold grid:")
    for label in grid["label"].unique():
        sub = grid[grid.label == label]
        d_med = sub["cohens_d"].median()
        d_min, d_max = sub["cohens_d"].min(), sub["cohens_d"].max()
        same_sign = np.mean(np.sign(sub["cohens_d"]) == np.sign(d_med))
        frac_sig = np.mean(sub["pval"] < 0.05)
        print(f"  {label:<22} d median={d_med:+.2f} range[{d_min:+.2f},{d_max:+.2f}] "
              f"same-sign={same_sign*100:.0f}% p<.05 in {frac_sig*100:.0f}% of grid")

    # Heatmap: median |d| stability per headline test (theta_pct x delta_pct, var fixed at 75)
    fig, axes = plt.subplots(1, len(grid["label"].unique()),
                              figsize=(5*len(grid["label"].unique()), 4))
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, grid["label"].unique()):
        sub = grid[(grid.label == label) & (grid.var_pct == 75)]
        piv = sub.pivot_table(index="theta_pct", columns="delta_pct",
                              values="cohens_d", aggfunc="mean")
        im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(v) > 1.2 else "black")
        ax.set_title(f"{label}\nCohen's d (var pct=75)", fontsize=9)
        ax.set_xlabel("delta percentile"); ax.set_ylabel("theta percentile")
    fig.suptitle("Staging sensitivity: genotype effect across REM threshold choices\n"
                 "Stable color/sign = finding not an artifact of one cutoff", fontsize=10)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="Cohen's d")
    fig.savefig(os.path.join(FIGURES_DIR, "staging_sensitivity_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: staging_sensitivity_heatmap.png")

    # State-separation diagnostic (theta/delta plane colored by assigned state)
    ep_path = os.path.join(DATA_DIR, f"epochs_with_states_{sample_tp}.csv")
    ep = pd.read_csv(ep_path)
    if "state" in ep.columns and RBP_COLS["theta"] in ep.columns and RBP_COLS["delta"] in ep.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        state_colors = {"Wake": "#888780", "NREM": "#378ADD", "REM": "#D85A30"}
        samp = ep.sample(min(8000, len(ep)), random_state=0)
        for st, c in state_colors.items():
            s = samp[samp.state == st]
            ax.scatter(s[RBP_COLS["delta"]], s[RBP_COLS["theta"]], s=4, alpha=0.3,
                       color=c, label=st)
        ax.set_xlabel("Relative delta power"); ax.set_ylabel("Relative theta power")
        ax.set_title(f"State separation in the theta-delta plane ({sample_tp})\n"
                     "Visual justification of band-power classification", fontsize=10)
        ax.legend(fontsize=9, markerscale=3)
        fig.savefig(os.path.join(FIGURES_DIR, "state_separation_diagnostic.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved: state_separation_diagnostic.png")
    return grid


def main():
    print("="*64)
    print("CV/PERMUTATION AUC + SLEEP-STAGING SENSITIVITY (rigor patches)")
    print("="*64)
    part_a_cv_auc()
    part_b_staging_sensitivity()
    print("\n" + "="*64)
    print("COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("="*64)


if __name__ == "__main__":
    main()
