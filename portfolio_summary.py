import pandas as pd
import numpy as np

# Load stats
stats = pd.read_csv("results/portfolio_integration_stats.csv")
pred  = pd.read_csv("results/predictive_3m_to_outcome.csv")
ri    = pd.read_csv("results/recovery_index.csv")

print("=" * 70)
print("FULL PORTFOLIO INTEGRATION RESULTS")
print("=" * 70)

# Add FDR
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

if "pval_fdr" not in stats.columns:
    stats["pval_fdr"] = fdr_bh(stats["pval"].values)
    stats["fdr_sig"]  = stats["pval_fdr"] < 0.05

TP_ORDER = ["3m","4m","6m","7m","9m","12m"]

print("\n=== STATISTICS BY FEATURE (sorted by effect size) ===")
feats = stats["feature"].unique()
for feat in feats:
    sub = stats[stats.feature==feat].copy()
    sub["tp_order"] = sub["timepoint"].map({t:i for i,t in enumerate(TP_ORDER)})
    sub = sub.sort_values("tp_order")
    print(f"\n  {feat}")
    print(f"  {'TP':<5} {'WT':>10} {'KO':>10} {'d':>7} {'95%CI':>16} {'p_raw':>9} {'p_FDR':>9} {'note'}")
    for _, r in sub.iterrows():
        ci_flag = " CI✓" if r.ci_excludes_zero else ""
        fdr_flag = " FDR*" if r.get("fdr_sig", False) else ""
        print(f"  {r.timepoint:<5} {r.wt_mean:>10.5f} {r.ko_mean:>10.5f} "
              f"{r.cohens_d:>7.3f} [{r.d_ci_lo:>6.3f},{r.d_ci_hi:>6.3f}] "
              f"{r.pval:>9.5f} {r.pval_fdr:>9.5f}{ci_flag}{fdr_flag}")

print("\n" + "="*70)
print("SUMMARY: FINDINGS WITH CI EXCLUDING ZERO")
print("="*70)
ci_sig = stats[stats.ci_excludes_zero].sort_values("pval")
for _, r in ci_sig.iterrows():
    fdr = " *** FDR*" if r.get("fdr_sig", False) else ""
    print(f"  [{r.timepoint}] {r.feature:<25} WT={r.wt_mean:.4f} KO={r.ko_mean:.4f} "
          f"d={r.cohens_d:.3f} [{r.d_ci_lo:.3f},{r.d_ci_hi:.3f}] "
          f"p={r.pval:.5f}{fdr}")

print("\n" + "="*70)
print("FDR SIGNIFICANT FINDINGS")
print("="*70)
fdr_sig = stats[stats.get("fdr_sig", False) == True] if "fdr_sig" in stats else pd.DataFrame()
if len(fdr_sig) > 0:
    for _, r in fdr_sig.iterrows():
        print(f"  *** [{r.timepoint}] {r.feature}: d={r.cohens_d:.3f} p={r.pval:.5f} FDR={r.pval_fdr:.5f}")
else:
    print("  None at FDR q<0.05")

print("\n" + "="*70)
print("PREDICTIVE ANALYSIS: 3m → 9m/12m outcomes")
print("="*70)
sig_pred = pred[pred.pval < 0.10].sort_values("pval")
for _, r in sig_pred.iterrows():
    fdr = " FDR*" if r.get("pval_fdr", 1) < 0.05 else ""
    print(f"  3m {r.predictor:<22} → [{r.tp_outcome}] {r.outcome:<25} "
          f"ρ={r.rho:.3f} p={r.pval:.5f}{fdr}")

print("\n" + "="*70)
print("RECOVERY INDEX: notable findings")
print("="*70)
for feat in ri["feature"].unique():
    for tp in ["9m","12m"]:
        sub = ri[(ri.feature==feat) & (ri.timepoint==tp)]
        wt = sub[sub.group=="WT"]["recovery_index"].dropna()
        ko = sub[sub.group=="KO"]["recovery_index"].dropna()
        if len(wt) >= 2 and len(ko) >= 2:
            from scipy.stats import mannwhitneyu
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            if p < 0.20:
                print(f"  [{tp}] {feat:<25} WT_ri={wt.mean():.3f} KO_ri={ko.mean():.3f} p={p:.5f}")
