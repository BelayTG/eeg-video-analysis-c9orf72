"""
15_mixed_effects_and_transitions.py
=====================================
Two rigorous additions a top-tier reviewer will expect:

PART 1 — LINEAR MIXED-EFFECTS MODELS (genotype × time interaction)
  Tests whether KO and WT TRAJECTORIES diverge over time, not just
  whether they differ at individual timepoints. Random intercept per
  animal accounts for repeated measures (same animals across timepoints).

  Model:  feature ~ genotype * time_months + (1 | animal_id)
  Key test: the genotype:time interaction term.
    - Significant interaction = trajectories diverge (the biphasic claim)
    - For beta sign-flip: interaction captures the crossover directly

  Features tested (the prespecified primaries):
    rem_td_ratio, rem_rbp_theta, rem_rbp_beta, wake_rbp_beta,
    nrem_rbp_theta, spindle_duration_mean

PART 2 — SLEEP STATE-TRANSITION DYNAMICS
  Beyond state percentages: how often does the brain switch between
  Wake / NREM / REM? Fragmented transitions are a neurodegeneration
  marker. Computes per-animal transition probability matrices and
  tests WT vs KO at each timepoint.

  Metrics:
    - Transition rate (switches per hour)
    - P(Wake→NREM), P(NREM→REM), P(REM→Wake), etc. (3x3 matrix)
    - State stability (mean dwell time per state)
    - Fragmentation index (number of distinct bouts / total time)

Outputs:
  results/mixed_effects_results.csv
  results/state_transition_matrices.csv
  results/state_transition_stats.csv
  figures/mixed_effects_trajectories.png
  figures/state_transition_heatmap.png

Run:
    python src/15_mixed_effects_and_transitions.py
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
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAVE_SM = True
except ImportError:
    HAVE_SM = False
    print("WARNING: statsmodels not installed. Run: pip install statsmodels --break-system-packages")

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")

COLORS   = {"WT": "#378ADD", "KO": "#D85A30"}
TP_ORDER = ["3m", "4m", "6m", "7m", "9m", "12m"]
TP_MONTHS = {"3m": 3, "4m": 4, "6m": 6, "7m": 7, "9m": 9, "12m": 12}
EPOCH_S  = 4.0


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


# ══════════════════════════════════════════════════════════════════════
# PART 1 — MIXED-EFFECTS MODELS
# ══════════════════════════════════════════════════════════════════════

MIXED_FEATURES = [
    ("rem_td_ratio",         "REM Theta/Delta Ratio"),
    ("rem_rbp_theta",        "REM Relative Theta"),
    ("rem_rbp_beta",         "REM Relative Beta"),
    ("wake_rbp_beta",        "Wake Relative Beta"),
    ("nrem_rbp_theta",       "NREM Relative Theta"),
    ("spindle_duration_mean","Sleep Spindle Duration"),
]


def load_longitudinal_data(scenario="A"):
    """
    Build long-format dataframe: one row per animal × timepoint × feature.
    Combines state_specific_features and advanced_eeg outputs.
    """
    rows = []
    for tp in TP_ORDER:
        # State-specific features (band power per state)
        sf_path = os.path.join(DATA_DIR, f"state_specific_features_{tp}_{scenario}.csv")
        if os.path.exists(sf_path):
            sf = pd.read_csv(sf_path)
            sf["animal_id"] = sf["animal_id"].astype(str)
            sf["timepoint"] = tp
            sf["time_months"] = TP_MONTHS[tp]
            rows.append(sf)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)

    # Merge spindle duration if available
    spindle_rows = []
    for tp in TP_ORDER:
        adv_path = os.path.join(RESULTS_DIR, f"advanced_eeg_{tp}.csv")
        if os.path.exists(adv_path):
            adv = pd.read_csv(adv_path)
            adv["animal_id"] = adv["animal_id"].astype(str)
            if "spindle_duration_mean" in adv.columns:
                sub = adv.groupby(["animal_id","group"])["spindle_duration_mean"].mean().reset_index()
                sub["timepoint"] = tp
                spindle_rows.append(sub)
    if spindle_rows:
        spindle_df = pd.concat(spindle_rows, ignore_index=True)
        df = df.merge(spindle_df[["animal_id","timepoint","spindle_duration_mean"]],
                      on=["animal_id","timepoint"], how="left",
                      suffixes=("","_adv"))

    return df


def run_mixed_effects(df, feature, label):
    """
    Fit: feature ~ genotype * time_months + (1|animal_id)
    Returns dict with interaction p-value and coefficients.

    Also runs a sensitivity check: OLS with cluster-robust (by animal)
    standard errors. When the mixed model's random-effect variance is
    near zero (boundary), the cluster-robust OLS gives a more honest
    interaction p-value and is reported alongside.
    """
    if not HAVE_SM:
        return None

    sub = df[["animal_id","group","time_months",feature]].dropna().copy()
    if len(sub) < 12 or sub["group"].nunique() < 2:
        return None

    # Need enough timepoints
    if sub["time_months"].nunique() < 3:
        return None

    sub["genotype"] = (sub["group"] == "KO").astype(int)
    sub = sub.rename(columns={feature: "y"})

    result_dict = {
        "feature": feature,
        "label": label,
        "n_obs": len(sub),
        "n_animals": sub["animal_id"].nunique(),
    }

    # ── Primary: mixed model with random intercept ──────────────────
    try:
        model = smf.mixedlm("y ~ genotype * time_months", sub,
                            groups=sub["animal_id"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(reml=True, method="lbfgs")

        params  = res.params
        pvalues = res.pvalues
        interaction_key = next((k for k in params.index
                                if "genotype:time" in k), None)

        if interaction_key:
            # Random effect variance (group variance)
            re_var = float(res.cov_re.iloc[0, 0]) if res.cov_re.shape[0] > 0 else 0.0
            result_dict.update({
                "genotype_coef": float(params.get("genotype", np.nan)),
                "genotype_p": float(pvalues.get("genotype", np.nan)),
                "time_coef": float(params.get("time_months", np.nan)),
                "time_p": float(pvalues.get("time_months", np.nan)),
                "interaction_coef": float(params[interaction_key]),
                "interaction_p": float(pvalues[interaction_key]),
                "converged": bool(res.converged),
                "re_var": re_var,
                "re_boundary": re_var < 1e-6,   # variance collapsed to zero
            })
    except Exception as e:
        result_dict["mixed_error"] = str(e)

    # ── Sensitivity: OLS with cluster-robust SE (by animal) ─────────
    try:
        ols = smf.ols("y ~ genotype * time_months", sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["animal_id"]})
        ikey = next((k for k in ols.params.index if "genotype:time" in k), None)
        if ikey:
            result_dict["robust_interaction_coef"] = float(ols.params[ikey])
            result_dict["robust_interaction_p"] = float(ols.pvalues[ikey])
    except Exception as e:
        result_dict["robust_error"] = str(e)

    # Need at least one valid interaction estimate
    if "interaction_p" not in result_dict and "robust_interaction_p" not in result_dict:
        return None

    return result_dict


def part1_mixed_effects():
    print("\n" + "="*64)
    print("PART 1 — LINEAR MIXED-EFFECTS MODELS")
    print("Model: feature ~ genotype * time + (1|animal)")
    print("Key test: genotype × time interaction (trajectory divergence)")
    print("="*64)

    if not HAVE_SM:
        print("\nstatsmodels required. Install with:")
        print("  pip install statsmodels --break-system-packages")
        return

    df = load_longitudinal_data(scenario="A")
    if df.empty:
        print("No longitudinal data found")
        return

    print(f"\nLoaded {len(df)} observations across {df['animal_id'].nunique()} animals")
    print(f"Timepoints: {sorted(df['time_months'].unique())}")

    results = []
    print(f"\n{'Feature':<26} {'Mixed p':>10} {'Robust p':>10} {'RE var':>9} {'Note'}")
    print("-" * 72)
    for feature, label in MIXED_FEATURES:
        if feature not in df.columns:
            print(f"{label:<26} {'(not in data)':>10}")
            continue
        res = run_mixed_effects(df, feature, label)
        if res is None:
            print(f"{label:<26} {'(insufficient)':>10}")
            continue
        results.append(res)

        mixed_p  = res.get("interaction_p", np.nan)
        robust_p = res.get("robust_interaction_p", np.nan)
        re_var   = res.get("re_var", np.nan)
        # Use robust p when random effect collapsed to boundary
        report_p = robust_p if res.get("re_boundary", False) and not np.isnan(robust_p) else mixed_p
        res["report_p"] = report_p

        sig = ("***" if report_p < 0.001 else "**" if report_p < 0.01 else
               "*" if report_p < 0.05 else "ns")
        note = ""
        if res.get("re_boundary", False):
            note = " [RE→0, using robust p]"
        elif not res.get("converged", True):
            note = " [!conv]"
        print(f"{label:<26} {mixed_p:>10.5f} {robust_p:>10.5f} "
              f"{re_var:>9.2e} {sig}{note}")

    if not results:
        print("\nNo models could be fit")
        return

    res_df = pd.DataFrame(results)

    # FDR on the reported interaction p-values (robust where RE collapsed)
    if len(res_df) > 1:
        res_df["report_p_fdr"] = fdr_bh(res_df["report_p"].values).round(5)
        res_df["report_fdr_sig"] = res_df["report_p_fdr"] < 0.05
        # Keep legacy columns for compatibility
        res_df["interaction_p_fdr"] = res_df["report_p_fdr"]
        res_df["interaction_fdr_sig"] = res_df["report_fdr_sig"]

    res_df.to_csv(os.path.join(RESULTS_DIR, "mixed_effects_results.csv"), index=False)
    print(f"\nSaved: mixed_effects_results.csv")

    # Summary
    print("\n--- INTERPRETATION (reported p = robust where random effect collapsed) ---")
    for _, r in res_df.iterrows():
        report_p = r.get("report_p", np.nan)
        if r.get("report_fdr_sig", False) or report_p < 0.05:
            coef = r.get("interaction_coef", r.get("robust_interaction_coef", np.nan))
            fdr_note = f" (FDR q={r.get('report_p_fdr', np.nan):.4f})" if "report_p_fdr" in r else ""
            method = "robust OLS" if r.get("re_boundary", False) else "mixed model"
            print(f"  {r['label']}: trajectories diverge over time, "
                  f"p={report_p:.4f}{fdr_note} [{method}]")
            print(f"    → KO slope differs from WT by {coef:+.4f}/month")

    # Figure: trajectories with fitted lines
    plot_mixed_effects_trajectories(df, res_df)


def plot_mixed_effects_trajectories(df, res_df):
    feats_to_plot = [(r["feature"], r["label"], r.get("report_p", np.nan),
                       r.get("report_p_fdr", np.nan))
                      for _, r in res_df.iterrows()]
    n = len(feats_to_plot)
    if n == 0:
        return
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.5, nrows*3.8))
    axes = np.atleast_1d(axes).flatten()

    for ax, (feat, label, p_int, p_fdr) in zip(axes, feats_to_plot):
        for group, color in COLORS.items():
            sub = df[df.group==group]
            xs, means, sems = [], [], []
            for tp in TP_ORDER:
                vals = sub[sub.timepoint==tp][feat].dropna()
                if len(vals)==0: continue
                xs.append(TP_MONTHS[tp]); means.append(vals.mean()); sems.append(vals.sem())
            if xs:
                ax.errorbar(xs, means, yerr=sems, fmt="-o", color=color,
                            lw=2, markersize=6, label=group, capsize=3)
        sig = ("***" if p_int<0.001 else "**" if p_int<0.01 else
               "*" if p_int<0.05 else "ns")
        fdr_str = f", FDR={p_fdr:.3f}" if not np.isnan(p_fdr) else ""
        ax.set_title(f"{label}\ngenotype×time p={p_int:.4f} {sig}{fdr_str}",
                     fontsize=9)
        ax.set_xlabel("Age (months)", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Mixed-Effects Trajectories — genotype × time interaction\n"
                 "Significant interaction = KO and WT diverge over time",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "mixed_effects_trajectories.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: mixed_effects_trajectories.png")


# ══════════════════════════════════════════════════════════════════════
# PART 2 — STATE-TRANSITION DYNAMICS
# ══════════════════════════════════════════════════════════════════════

STATES = ["Wake", "NREM", "REM"]


def compute_transitions(state_sequence):
    """
    Compute transition counts and probabilities from a state sequence.
    Returns 3x3 matrix of P(from→to), transition rate, dwell times.
    """
    # Remove consecutive duplicates to find transitions
    states = [s for s in state_sequence if s in STATES]
    if len(states) < 10:
        return None

    # Transition count matrix
    trans_counts = {f: {t: 0 for t in STATES} for f in STATES}
    n_transitions = 0
    for i in range(len(states)-1):
        a, b = states[i], states[i+1]
        trans_counts[a][b] += 1
        if a != b:
            n_transitions += 1

    # Transition probabilities (row-normalized)
    trans_prob = {}
    for f in STATES:
        total = sum(trans_counts[f].values())
        trans_prob[f] = {t: (trans_counts[f][t]/total if total>0 else 0)
                         for t in STATES}

    # Dwell times (mean consecutive run length per state, in seconds)
    dwell = {s: [] for s in STATES}
    current = states[0]; run = 1
    for s in states[1:]:
        if s == current:
            run += 1
        else:
            dwell[current].append(run * EPOCH_S)
            current = s; run = 1
    dwell[current].append(run * EPOCH_S)
    mean_dwell = {s: (np.mean(dwell[s]) if dwell[s] else 0) for s in STATES}

    # Fragmentation: transitions per hour
    total_hours = len(states) * EPOCH_S / 3600
    trans_rate = n_transitions / total_hours if total_hours > 0 else 0

    # Number of distinct bouts
    n_bouts = sum(len(dwell[s]) for s in STATES)

    return {
        "trans_prob": trans_prob,
        "trans_rate_per_hour": trans_rate,
        "mean_dwell": mean_dwell,
        "n_bouts": n_bouts,
        "n_transitions": n_transitions,
        "total_hours": total_hours,
    }


def part2_state_transitions():
    print("\n" + "="*64)
    print("PART 2 — SLEEP STATE-TRANSITION DYNAMICS")
    print("Transition probabilities, rates, dwell times: WT vs KO")
    print("="*64)

    all_rows = []
    matrix_rows = []

    for tp in TP_ORDER:
        ep_path = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
        if not os.path.exists(ep_path):
            continue
        ep = pd.read_csv(ep_path)
        ep["animal_id"] = ep["animal_id"].astype(str)

        for (animal_id, group), adf in ep.groupby(["animal_id","group"]):
            # Process per file then average, or concatenate per animal
            adf = adf.sort_values(["abf_file","epoch_idx"])
            result = compute_transitions(adf["state"].tolist())
            if result is None:
                continue

            row = {
                "animal_id": animal_id,
                "group": group,
                "timepoint": tp,
                "trans_rate_per_hour": result["trans_rate_per_hour"],
                "n_bouts": result["n_bouts"],
                "dwell_wake": result["mean_dwell"]["Wake"],
                "dwell_nrem": result["mean_dwell"]["NREM"],
                "dwell_rem":  result["mean_dwell"]["REM"],
            }
            # Add transition probabilities
            for f in STATES:
                for t in STATES:
                    row[f"p_{f.lower()}_{t.lower()}"] = result["trans_prob"][f][t]
            all_rows.append(row)

            # Matrix storage
            for f in STATES:
                for t in STATES:
                    matrix_rows.append({
                        "animal_id": animal_id, "group": group,
                        "timepoint": tp, "from_state": f, "to_state": t,
                        "probability": result["trans_prob"][f][t],
                    })

    if not all_rows:
        print("No epoch data available")
        return

    trans_df = pd.DataFrame(all_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    trans_df.to_csv(os.path.join(RESULTS_DIR, "state_transition_features.csv"), index=False)
    matrix_df.to_csv(os.path.join(RESULTS_DIR, "state_transition_matrices.csv"), index=False)
    print(f"\nSaved: state_transition_features.csv ({len(trans_df)} rows)")

    # Statistics: WT vs KO per timepoint
    print("\nWT vs KO transition metrics (p<0.10 shown):")
    test_features = ["trans_rate_per_hour", "n_bouts",
                     "dwell_wake", "dwell_nrem", "dwell_rem",
                     "p_wake_nrem", "p_nrem_rem", "p_rem_wake",
                     "p_nrem_wake", "p_rem_nrem", "p_wake_rem"]
    stat_rows = []
    for tp in TP_ORDER:
        sub = trans_df[trans_df.timepoint==tp]
        for feat in test_features:
            if feat not in sub.columns:
                continue
            wt = sub[sub.group=="WT"][feat].dropna().values
            ko = sub[sub.group=="KO"][feat].dropna().values
            if len(wt)<2 or len(ko)<2:
                continue
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            d = (np.mean(ko)-np.mean(wt))/np.sqrt((np.std(wt)**2+np.std(ko)**2)/2+1e-10)
            stat_rows.append({
                "timepoint": tp, "feature": feat,
                "wt_mean": np.mean(wt), "ko_mean": np.mean(ko),
                "cohens_d": d, "pval": p, "n_wt": len(wt), "n_ko": len(ko),
            })
            if p < 0.10:
                print(f"  [{tp}] {feat:<22}: WT={np.mean(wt):.4f} KO={np.mean(ko):.4f} "
                      f"d={d:.3f} p={p:.4f}")

    stat_df = pd.DataFrame(stat_rows)
    if len(stat_df) > 1:
        stat_df["pval_fdr"] = fdr_bh(stat_df["pval"].values).round(5)
        stat_df["fdr_sig"] = stat_df["pval_fdr"] < 0.05
    stat_df.to_csv(os.path.join(RESULTS_DIR, "state_transition_stats.csv"), index=False)

    fdr_sig = stat_df[stat_df.get("fdr_sig", pd.Series(False))==True] \
        if "fdr_sig" in stat_df.columns else pd.DataFrame()
    print(f"\nFDR significant: {len(fdr_sig)}")
    for _, r in fdr_sig.iterrows():
        print(f"  *** [{r.timepoint}] {r.feature}: d={r.cohens_d:.3f} "
              f"p={r.pval:.5f} FDR={r.pval_fdr:.5f}")

    # Figure: transition matrix heatmaps WT vs KO at 4m and 12m
    plot_transition_heatmaps(matrix_df)


def plot_transition_heatmaps(matrix_df):
    # Show 3m, 4m, 12m for both groups
    tps_to_show = [t for t in ["3m","4m","12m"] if t in matrix_df["timepoint"].unique()]
    if not tps_to_show:
        return

    fig, axes = plt.subplots(2, len(tps_to_show),
                              figsize=(len(tps_to_show)*3.2, 6.4))
    if len(tps_to_show) == 1:
        axes = axes.reshape(2, 1)

    for col, tp in enumerate(tps_to_show):
        for row, group in enumerate(["WT", "KO"]):
            ax = axes[row, col]
            sub = matrix_df[(matrix_df.timepoint==tp) & (matrix_df.group==group)]
            # Average matrix
            mat = np.zeros((3, 3))
            for i, f in enumerate(STATES):
                for j, t in enumerate(STATES):
                    vals = sub[(sub.from_state==f) & (sub.to_state==t)]["probability"]
                    mat[i, j] = vals.mean() if len(vals) > 0 else 0

            im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(3)); ax.set_xticklabels(STATES, fontsize=8)
            ax.set_yticks(range(3)); ax.set_yticklabels(STATES, fontsize=8)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if mat[i,j]>0.5 else "black")
            ax.set_title(f"{group} — {tp}", fontsize=9)
            if col == 0:
                ax.set_ylabel("From state", fontsize=8)
            if row == 1:
                ax.set_xlabel("To state", fontsize=8)

    fig.suptitle("Sleep State-Transition Probabilities — WT vs KO\n"
                 "P(row state → column state)", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "state_transition_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: state_transition_heatmap.png")


# ══════════════════════════════════════════════════════════════════════

def main():
    print("="*64)
    print("MIXED-EFFECTS MODELS + STATE-TRANSITION DYNAMICS")
    print("="*64)

    part1_mixed_effects()
    part2_state_transitions()

    print("\n" + "="*64)
    print("COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("="*64)


if __name__ == "__main__":
    main()
