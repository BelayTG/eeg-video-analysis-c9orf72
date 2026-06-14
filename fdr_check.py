import pandas as pd
import numpy as np
import os

def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction without statsmodels."""
    pvals = np.array(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=int)
    ranked[order] = np.arange(1, n+1)
    fdr = pvals * n / ranked
    # Enforce monotonicity
    fdr_adj = np.minimum.accumulate(fdr[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = fdr_adj
    return np.minimum(result, 1.0)

results_dir = "results"

print("=" * 70)
print("FDR CORRECTION STATUS — ALL RESULTS")
print("=" * 70)

# ── Script 05 statistics (band power) ─────────────────────────────────────
print("\n=== SCRIPT 05: Band Power Statistics ===")
for tp in ["3m","4m","6m","7m","9m","12m"]:
    for sc in ["A","B"]:
        path = os.path.join(results_dir, f"statistics_{tp}_{sc}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        has_fdr = "pval_fdr" in df.columns
        sig_raw = (df.pval < 0.05).sum()
        sig_fdr = (df.pval_fdr < 0.05).sum() if has_fdr else "N/A"
        print(f"  {tp} Sc{sc}: {sig_raw} sig (raw) | {sig_fdr} sig (FDR) | "
              f"FDR column: {has_fdr}")
        if has_fdr and sig_fdr != "N/A" and sig_fdr > 0:
            fdr_sig = df[df.pval_fdr < 0.05][["feature","wt_mean","ko_mean",
                                               "cohens_d","pval","pval_fdr","sig"]]
            for _, r in fdr_sig.iterrows():
                print(f"    *** {r.feature:<35} d={r.cohens_d:.3f} "
                      f"p={r.pval:.5f} FDR_p={r.pval_fdr:.5f}")

# ── Script 06 longitudinal effect sizes ───────────────────────────────────
print("\n=== SCRIPT 06: Longitudinal Effect Sizes ===")
for sc in ["scenA","scenB"]:
    path = os.path.join(results_dir, f"longitudinal_effect_sizes_{sc}.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    has_fdr = "pval_fdr" in df.columns
    sig_raw = (df.pval < 0.05).sum()
    sig_fdr = (df.pval_fdr < 0.05).sum() if has_fdr else "N/A"
    print(f"  {sc}: {sig_raw} sig (raw) | {sig_fdr} sig (FDR) | "
          f"FDR column: {has_fdr}")
    if has_fdr and isinstance(sig_fdr, int) and sig_fdr > 0:
        fdr_sig = df[df.pval_fdr < 0.05].sort_values("pval_fdr")
        for _, r in fdr_sig.iterrows():
            print(f"    *** [{r.timepoint}] {r.feature:<35} "
                  f"d={r.cohens_d:.3f} p={r.pval:.5f} FDR_p={r.pval_fdr:.5f}")

# ── Script 08 advanced EEG ────────────────────────────────────────────────
print("\n=== SCRIPT 08: Advanced EEG Statistics ===")
for tp in ["3m","4m","6m","7m","9m","12m"]:
    path = os.path.join(results_dir, f"advanced_eeg_stats_{tp}.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    has_fdr = "pval_fdr" in df.columns
    sig_raw = (df.pval < 0.05).sum()
    sig_fdr = (df.pval_fdr < 0.05).sum() if has_fdr else "N/A"
    print(f"  {tp}: {sig_raw} sig (raw) | {sig_fdr} sig (FDR) | "
          f"FDR column: {has_fdr}")
    if has_fdr and isinstance(sig_fdr, int) and sig_fdr > 0:
        fdr_sig = df[df.pval_fdr < 0.05][["feature","wt_mean","ko_mean",
                                           "cohens_d","pval","pval_fdr"]]
        for _, r in fdr_sig.iterrows():
            print(f"    *** {r.feature:<35} d={r.cohens_d:.3f} "
                  f"p={r.pval:.5f} FDR_p={r.pval_fdr:.5f}")

# ── Apply FDR to any files missing it ─────────────────────────────────────
print("\n=== APPLYING FDR WHERE MISSING ===")
for tp in ["3m","4m","6m","7m","9m","12m"]:
    for sc in ["A","B"]:
        path = os.path.join(results_dir, f"statistics_{tp}_{sc}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "pval_fdr" not in df.columns:
            df["pval_fdr"] = fdr_bh(df["pval"].values)
            df["fdr_sig"] = df["pval_fdr"] < 0.05
            df.to_csv(path, index=False)
            print(f"  Added FDR to statistics_{tp}_{sc}.csv")

print("\nDONE")
