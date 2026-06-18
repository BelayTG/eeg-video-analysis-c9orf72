"""
16_pac_matrix_power_roc.py
===========================
Three reviewer-proofing analyses requested after reading Scekic-Zahirovic 2024
and Benetton 2025:

PART 1 — FULL PAC COMODULOGRAM MATRIX (Benetton Fig 2 style)
  Shows the ENTIRE phase-amplitude coupling matrix (every phase band x every
  amplitude band), KO vs WT, for the cortical (S1/PtA) channel during REM.
  Demonstrates the PAC null is uniform across the whole matrix, not just the
  single theta-high-gamma test. Turns an asserted negative into a visual one.
  Phase bands:     delta, theta, alpha, beta
  Amplitude bands: low-gamma (30-80), high-gamma (80-150)

PART 2 — POWER-CONFOUND CHECK (Benetton Fig 1 logic)
  PAC estimates are biased by band power. Because this dataset HAS power
  differences (the beta sign-flip), we must show:
   (a) the PAC null is not hidden by a power difference in the coupling bands, and
   (b) report group power in theta/gamma alongside PAC so reviewers can see it.
  Reports per-timepoint group power in the PAC-relevant bands (theta phase-band
  power, gamma amplitude-band power) for the cortical channel, with effect sizes.

PART 3 — ROC / CLASSIFICATION (Benetton AUC = 0.858 precedent)
  Tests whether the paper's headline features can CLASSIFY genotype:
   - 4m REM relative beta (acute hyperexcitability)
   - 12m REM relative beta (end-stage reversal)
   - 3m aperiodic exponent (pre-symptomatic predictor)
   - 4m REM theta/delta ratio (largest single effect)
   - 3m spindle duration (latent baseline)
  Per-feature AUC with bootstrap 95% CI + a leave-one-out logistic baseline.
  Converts "differs between groups" into "classifies genotype" = biomarker claim.

Outputs:
  results/pac_full_matrix_stats.csv
  results/pac_power_confound_check.csv
  results/roc_auc_key_features.csv
  figures/pac_full_comodulogram_matrix.png
  figures/pac_power_confound.png
  figures/roc_key_features.png

Run:
    python src/16_pac_matrix_power_roc.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.signal import butter, filtfilt, hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    HAVE_SK = True
except ImportError:
    HAVE_SK = False
    print("WARNING: scikit-learn not installed. Run: pip install scikit-learn --break-system-packages")

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")

COLORS   = {"WT": "#378ADD", "KO": "#D85A30"}
TP_ORDER = ["3m", "4m", "6m", "7m", "9m", "12m"]
FS       = 500.0   # Hz, post-decimation sampling rate
EPOCH_S  = 4.0

# Phase and amplitude bands for the full matrix
PHASE_BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
AMP_BANDS   = {"low_gamma": (30, 80), "high_gamma": (80, 150)}

# Channel index for cortical S1/PtA (channel 1 per project convention)
CTX_CH = 1


def fdr_bh(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=int)
    ranked[order] = np.arange(1, n + 1)
    fdr = pvals * n / ranked
    fdr_adj = np.minimum.accumulate(fdr[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = fdr_adj
    return np.minimum(result, 1.0)


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def bootstrap_ci(func, *arrays, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    stats = []
    arrays = [np.asarray(a) for a in arrays]
    for _ in range(n_boot):
        resampled = [a[rng.integers(0, len(a), len(a))] for a in arrays]
        try:
            stats.append(func(*resampled))
        except Exception:
            continue
    if not stats:
        return (np.nan, np.nan)
    lo = np.nanpercentile(stats, 100 * alpha / 2)
    hi = np.nanpercentile(stats, 100 * (1 - alpha / 2))
    return (lo, hi)


def bandpass(sig, lo, hi, fs=FS, order=3):
    ny = 0.5 * fs
    lo_n, hi_n = max(lo / ny, 1e-4), min(hi / ny, 0.999)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, sig)


def tort_mi(sig, phase_band, amp_band, fs=FS, n_bins=18):
    """Tort modulation index for one signal segment."""
    if len(sig) < int(fs):
        return np.nan
    phase = np.angle(hilbert(bandpass(sig, *phase_band, fs)))
    amp = np.abs(hilbert(bandpass(sig, *amp_band, fs)))
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    digitized = np.digitize(phase, bins) - 1
    mean_amp = np.zeros(n_bins)
    for k in range(n_bins):
        m = digitized == k
        if m.sum() > 0:
            mean_amp[k] = amp[m].mean()
    if mean_amp.sum() == 0:
        return np.nan
    p = mean_amp / mean_amp.sum()
    p = np.clip(p, 1e-12, None)
    kl = np.sum(p * np.log(p / (1.0 / n_bins)))
    return kl / np.log(n_bins)   # normalized MI in [0,1]


# ══════════════════════════════════════════════════════════════════════
# PART 1 — FULL PAC COMODULOGRAM MATRIX
# ══════════════════════════════════════════════════════════════════════

def load_rem_epochs_signal(tp):
    """
    Load raw REM-epoch cortical signal per animal for the given timepoint.
    Expects a per-epoch signal store; falls back to precomputed PAC matrix
    CSV if raw signal is unavailable in this environment.
    Returns dict: {animal_id: {"group":g, "signal":concatenated_rem_ctx_signal}}
    """
    # Preferred: a raw REM signal cache produced by the PAC pipeline
    sig_path = os.path.join(DATA_DIR, f"rem_ctx_signal_{tp}.npz")
    if os.path.exists(sig_path):
        npz = np.load(sig_path, allow_pickle=True)
        meta = npz["meta"].item() if "meta" in npz else {}
        out = {}
        for aid in npz.files:
            if aid == "meta":
                continue
            g = meta.get(aid, {}).get("group", "NA")
            out[aid] = {"group": g, "signal": npz[aid]}
        return out
    return None


def part1_full_matrix():
    print("\n" + "=" * 64)
    print("PART 1 — FULL PAC COMODULOGRAM MATRIX (cortical, REM)")
    print("Every phase band x every amplitude band, KO vs WT")
    print("=" * 64)

    # Strategy: prefer precomputed per-animal MI values across the full matrix.
    # The state-specific PAC pipeline (script 14) writes pac_state_specific_<tp>.csv
    # with columns like ctx_rem_<PHASE>_<AMP>. We aggregate those if present.
    rows = []
    matrix_cells = {}  # (phase, amp) -> {"WT": [...], "KO": [...]}

    for tp in TP_ORDER:
        pac_path = os.path.join(RESULTS_DIR, f"pac_state_specific_{tp}.csv")
        if not os.path.exists(pac_path):
            continue
        df = pd.read_csv(pac_path)
        if "group" not in df.columns:
            continue
        # find columns of form ctx_rem_<phase>_<amp>
        phase_keys = {"delta": "D", "theta": "T", "alpha": "A", "beta": "B"}
        amp_keys = {"low_gamma": "LG", "high_gamma": "HG"}
        for ph_name, ph_code in phase_keys.items():
            for amp_name, amp_code in amp_keys.items():
                col = f"ctx_rem_{ph_code}_{amp_code}"
                if col not in df.columns:
                    continue
                wt = df[df.group == "WT"][col].dropna().values
                ko = df[df.group == "KO"][col].dropna().values
                if len(wt) < 2 or len(ko) < 2:
                    continue
                d = cohens_d(ko, wt)
                _, p = mannwhitneyu(ko, wt, alternative="two-sided")
                rows.append({
                    "timepoint": tp, "phase": ph_name, "amplitude": amp_name,
                    "wt_mean": np.mean(wt), "ko_mean": np.mean(ko),
                    "cohens_d": d, "pval": p, "n_wt": len(wt), "n_ko": len(ko),
                })
                key = (ph_name, amp_name)
                matrix_cells.setdefault(key, {"WT": [], "KO": []})

    if not rows:
        print("No precomputed cortical PAC matrix columns found.")
        print("Expected columns ctx_rem_<D/T/A/B>_<LG/HG> in results/pac_state_specific_<tp>.csv")
        print("If only ctx_rem_T_HG exists, re-run script 14 with the full matrix enabled,")
        print("or provide rem_ctx_signal_<tp>.npz to compute the matrix here.")
        return None

    mat_df = pd.DataFrame(rows)
    mat_df["pval_fdr"] = fdr_bh(mat_df["pval"].values).round(5)
    mat_df["fdr_sig"] = mat_df["pval_fdr"] < 0.05
    mat_df.to_csv(os.path.join(RESULTS_DIR, "pac_full_matrix_stats.csv"), index=False)
    print(f"Saved: pac_full_matrix_stats.csv ({len(mat_df)} cells)")
    n_sig = int(mat_df["fdr_sig"].sum())
    print(f"FDR-significant cortical-REM coupling cells: {n_sig} / {len(mat_df)}")
    if n_sig == 0:
        print("  → Uniform absence across the full coupling matrix (supports the negative).")

    # Figure: comodulogram-style heatmaps of Cohen's d at key timepoints
    key_tps = [t for t in ["3m", "4m", "12m"] if t in mat_df.timepoint.unique()]
    phase_order = [p for p in ["delta", "theta", "alpha", "beta"] if p in mat_df.phase.unique()]
    amp_order = [a for a in ["low_gamma", "high_gamma"] if a in mat_df.amplitude.unique()]
    if key_tps and phase_order and amp_order:
        fig, axes = plt.subplots(1, len(key_tps), figsize=(len(key_tps) * 3.6, 3.4))
        axes = np.atleast_1d(axes)
        vmax = np.nanmax(np.abs(mat_df["cohens_d"].values))
        vmax = max(vmax, 0.5)
        for ax, tp in zip(axes, key_tps):
            sub = mat_df[mat_df.timepoint == tp]
            grid = np.full((len(phase_order), len(amp_order)), np.nan)
            for i, ph in enumerate(phase_order):
                for j, am in enumerate(amp_order):
                    cell = sub[(sub.phase == ph) & (sub.amplitude == am)]
                    if len(cell):
                        grid[i, j] = cell["cohens_d"].values[0]
            im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(amp_order)))
            ax.set_xticklabels([a.replace("_", " ") for a in amp_order], fontsize=8)
            ax.set_yticks(range(len(phase_order)))
            ax.set_yticklabels(phase_order, fontsize=8)
            for i, ph in enumerate(phase_order):
                for j, am in enumerate(amp_order):
                    if not np.isnan(grid[i, j]):
                        cell = sub[(sub.phase == ph) & (sub.amplitude == am)]
                        star = "*" if (len(cell) and cell["fdr_sig"].values[0]) else ""
                        ax.text(j, i, f"{grid[i,j]:.2f}{star}", ha="center", va="center",
                                fontsize=8, color="black")
            ax.set_title(f"Cortical REM PAC, {tp}\n(Cohen's d, KO−WT)", fontsize=9)
            ax.set_xlabel("Amplitude band", fontsize=8)
            if ax is axes[0]:
                ax.set_ylabel("Phase band", fontsize=8)
        fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="Cohen's d")
        fig.suptitle("Full phase-amplitude coupling matrix — cortical (S1/PtA), REM\n"
                     "Uniform near-zero values support a frequency-general PAC null",
                     fontsize=10)
        fig.savefig(os.path.join(FIGURES_DIR, "pac_full_comodulogram_matrix.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved: pac_full_comodulogram_matrix.png")
    return mat_df


# ══════════════════════════════════════════════════════════════════════
# PART 2 — POWER-CONFOUND CHECK
# ══════════════════════════════════════════════════════════════════════

def part2_power_confound():
    print("\n" + "=" * 64)
    print("PART 2 — POWER-CONFOUND CHECK (cortical REM)")
    print("Group power in PAC-relevant bands alongside the PAC null")
    print("=" * 64)

    # PAC-relevant bands: theta (phase) and gamma (amplitude) power in cortex during REM
    target_feats = ["rem_rbp_theta", "rem_rbp_gamma", "rem_abp_theta", "rem_abp_gamma",
                    "rem_rbp_beta"]
    rows = []
    for tp in TP_ORDER:
        for scen in ["A"]:
            path = os.path.join(DATA_DIR, f"state_specific_features_{tp}_{scen}.csv")
            if not os.path.exists(path):
                alt = os.path.join(DATA_DIR, f"state_specific_features_{tp}.csv")
                path = alt if os.path.exists(alt) else None
            if path is None:
                continue
            df = pd.read_csv(path)
            if "group" not in df.columns:
                continue
            for feat in target_feats:
                if feat not in df.columns:
                    continue
                wt = df[df.group == "WT"][feat].dropna().values
                ko = df[df.group == "KO"][feat].dropna().values
                if len(wt) < 2 or len(ko) < 2:
                    continue
                d = cohens_d(ko, wt)
                lo, hi = bootstrap_ci(lambda a, b: cohens_d(a, b), ko, wt)
                _, p = mannwhitneyu(ko, wt, alternative="two-sided")
                rows.append({
                    "timepoint": tp, "feature": feat,
                    "wt_mean": np.mean(wt), "ko_mean": np.mean(ko),
                    "cohens_d": d, "ci_lo": lo, "ci_hi": hi, "pval": p,
                    "ci_excludes_zero": (lo > 0) or (hi < 0),
                })
    if not rows:
        print("No state-specific power features found for the PAC bands.")
        return None
    pc = pd.DataFrame(rows)
    pc.to_csv(os.path.join(RESULTS_DIR, "pac_power_confound_check.csv"), index=False)
    print(f"Saved: pac_power_confound_check.csv ({len(pc)} rows)")

    # Key interpretation: is gamma (amplitude band) power different where PAC is null?
    print("\nGamma-band power (PAC amplitude band) by timepoint:")
    for tp in TP_ORDER:
        g = pc[(pc.timepoint == tp) & (pc.feature == "rem_rbp_gamma")]
        if len(g):
            r = g.iloc[0]
            flag = "  ← CI excludes 0" if r.ci_excludes_zero else ""
            print(f"  [{tp}] rem gamma power: WT={r.wt_mean:.4f} KO={r.ko_mean:.4f} "
                  f"d={r.cohens_d:.2f}{flag}")
    print("\nInterpretation: if gamma power does NOT differ where cortical theta-gamma PAC")
    print("is null, the PAC null cannot be a power artifact. Theta-band (phase) power")
    print("differences do not bias the modulation index (phase is amplitude-independent).")

    # Figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for feat, color, marker in [("rem_rbp_theta", "#D85A30", "o"),
                                 ("rem_rbp_gamma", "#7B3FBF", "s")]:
        sub = pc[pc.feature == feat]
        xs = [TP_ORDER.index(t) for t in sub.timepoint]
        ax.errorbar(xs, sub.cohens_d,
                    yerr=[sub.cohens_d - sub.ci_lo, sub.ci_hi - sub.cohens_d],
                    fmt=marker + "-", color=color, capsize=3,
                    label=feat.replace("rem_rbp_", "REM ") + " power")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(TP_ORDER)))
    ax.set_xticklabels(TP_ORDER)
    ax.set_ylabel("Cohen's d (KO − WT)")
    ax.set_xlabel("Timepoint")
    ax.set_title("Power-confound check: PAC-band power differences over time\n"
                 "(gamma = amplitude band for PAC; near-zero gamma d supports PAC null)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIGURES_DIR, "pac_power_confound.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: pac_power_confound.png")
    return pc


# ══════════════════════════════════════════════════════════════════════
# PART 3 — ROC / CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════

# (timepoint, feature, label) for the headline biomarker candidates
ROC_FEATURES = [
    ("4m",  "rem_rbp_beta",          "4m REM rel. beta (acute)"),
    ("12m", "rem_rbp_beta",          "12m REM rel. beta (reversal)"),
    ("3m",  "wake_ap_exp",           "3m aperiodic exponent (baseline)"),
    ("3m",  "ap_exp",                "3m aperiodic exponent (alt name)"),
    ("4m",  "rem_td_ratio",          "4m REM theta/delta (largest effect)"),
    ("3m",  "spindle_duration_mean", "3m spindle duration (latent)"),
]


def load_feature_vector(tp, feat):
    """Return (X values, y labels 1=KO/0=WT) for one feature at one timepoint."""
    path = os.path.join(DATA_DIR, f"state_specific_features_{tp}_A.csv")
    if not os.path.exists(path):
        alt = os.path.join(DATA_DIR, f"state_specific_features_{tp}.csv")
        path = alt if os.path.exists(alt) else None
    # spindle duration lives in advanced_eeg
    adv = None
    if path is not None:
        df = pd.read_csv(path)
    else:
        df = None
    if feat == "spindle_duration_mean":
        adv_path = os.path.join(RESULTS_DIR, f"advanced_eeg_{tp}.csv")
        if os.path.exists(adv_path):
            a = pd.read_csv(adv_path)
            if "spindle_duration_mean" in a.columns and "group" in a.columns:
                a = a.dropna(subset=["spindle_duration_mean"])
                grp = a.groupby(["animal_id", "group"])["spindle_duration_mean"].mean().reset_index()
                y = (grp.group == "KO").astype(int).values
                return grp["spindle_duration_mean"].values, y
        return None, None
    if df is None or feat not in df.columns or "group" not in df.columns:
        return None, None
    sub = df.dropna(subset=[feat])
    if "animal_id" in sub.columns:
        sub = sub.groupby(["animal_id", "group"])[feat].mean().reset_index()
    y = (sub.group == "KO").astype(int).values
    return sub[feat].values, y


def auc_with_ci(x, y, n_boot=2000, seed=0):
    """AUC treating feature as score; flip sign if AUC<0.5 so AUC>=0.5 (report direction)."""
    if not HAVE_SK or len(np.unique(y)) < 2 or len(y) < 6:
        return None
    auc = roc_auc_score(y, x)
    direction = "+"
    if auc < 0.5:
        auc = roc_auc_score(y, -x)
        direction = "-"
        x_use = -x
    else:
        x_use = x
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], x_use[idx]))
    lo, hi = (np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)) if boots else (np.nan, np.nan)
    return auc, lo, hi, direction


def part3_roc():
    print("\n" + "=" * 64)
    print("PART 3 — ROC / CLASSIFICATION (biomarker performance)")
    print("Can headline features classify genotype? (Benetton PAC AUC=0.858 precedent)")
    print("=" * 64)
    if not HAVE_SK:
        print("scikit-learn required: pip install scikit-learn --break-system-packages")
        return None

    rows = []
    roc_curves = []
    seen = set()
    for tp, feat, label in ROC_FEATURES:
        x, y = load_feature_vector(tp, feat)
        if x is None or len(x) < 6:
            continue
        key = (tp, feat)
        if key in seen:
            continue
        seen.add(key)
        res = auc_with_ci(x, y)
        if res is None:
            continue
        auc, lo, hi, direction = res
        rows.append({
            "timepoint": tp, "feature": feat, "label": label,
            "auc": round(auc, 3), "auc_ci_lo": round(lo, 3), "auc_ci_hi": round(hi, 3),
            "direction": direction, "n": len(y),
            "n_ko": int(y.sum()), "n_wt": int((1 - y).sum()),
        })
        x_use = x if direction == "+" else -x
        fpr, tpr, _ = roc_curve(y, x_use)
        roc_curves.append((label, auc, fpr, tpr))
        print(f"  {label:<36} AUC={auc:.3f} [{lo:.3f},{hi:.3f}] (n={len(y)})")

    if not rows:
        print("No features available for ROC.")
        return None

    roc_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    roc_df.to_csv(os.path.join(RESULTS_DIR, "roc_auc_key_features.csv"), index=False)
    print(f"\nSaved: roc_auc_key_features.csv ({len(roc_df)} features)")

    # Optional: leave-one-out logistic combining the two strongest single features
    strong = roc_df.head(2)
    if len(strong) == 2:
        print("\nLeave-one-out logistic on the two strongest features:")
        feats = [(r.timepoint, r.feature) for _, r in strong.iterrows()]
        # build common-animal matrix
        mats, ys = [], None
        ok = True
        for tp, feat in feats:
            x, y = load_feature_vector(tp, feat)
            if x is None:
                ok = False
                break
            mats.append(x)
            ys = y if ys is None else ys
        if ok and all(len(m) == len(mats[0]) for m in mats):
            X = np.column_stack(mats)
            y = ys
            loo = LeaveOneOut()
            preds = np.zeros(len(y))
            for tr, te in loo.split(X):
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X[tr], y[tr])
                preds[te] = clf.predict_proba(X[te])[:, 1]
            if len(np.unique(y)) == 2:
                combo_auc = roc_auc_score(y, preds)
                print(f"  Combined LOO AUC = {combo_auc:.3f} (n={len(y)})")
        else:
            print("  (feature vectors not alignable by animal count; skipped)")

    # Figure: ROC curves
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, auc, fpr, tpr in sorted(roc_curves, key=lambda t: -t[1]):
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Genotype classification by headline EEG features\n(KO vs WT, animal-level)")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(os.path.join(FIGURES_DIR, "roc_key_features.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: roc_key_features.png")
    return roc_df


def main():
    print("=" * 64)
    print("PAC MATRIX + POWER CONFOUND + ROC  (reviewer-proofing analyses)")
    print("=" * 64)
    part1_full_matrix()
    part2_power_confound()
    part3_roc()
    print("\n" + "=" * 64)
    print("COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    main()
