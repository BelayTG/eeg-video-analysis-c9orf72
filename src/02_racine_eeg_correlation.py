"""
02_racine_eeg_correlation.py
=============================
Correlates manual Racine behavioral seizure scores (4m KA)
with EEG features from the same animals.

Input:
  - data/racine_scores.csv  (you fill this in)
  - EEG features from existing 9m portfolio

Racine CSV format:
  animal_id, group, latency_first_seizure_min,
  n_stage4, duration_stage4_s,
  n_stage5, duration_stage5_s,
  total_stage45_duration_s, max_racine_stage,
  mean_racine_score, total_seizure_duration_min

Run:
    python src/02_racine_eeg_correlation.py
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
SRC_9M      = r"C:\Users\belay\eeg-network-vulnerability-c9orf72-9m"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")
for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

COLORS = {"WT": "#378ADD", "KO": "#D85A30"}

# ── Create Racine template if not exists ──────────────────────────────────
racine_path = os.path.join(DATA_DIR, "racine_scores.csv")
if not os.path.exists(racine_path):
    print("Creating Racine score template...")

    # Get animal IDs from existing EEG data
    means_path = os.path.join(SRC_9M, "data", "animal_means_9m.csv")
    if os.path.exists(means_path):
        means = pd.read_csv(means_path)
        means["animal_id"] = means["animal_id"].astype(str)
        animals = means[["animal_id","group"]].drop_duplicates()
    else:
        # Create manually
        animals = pd.DataFrame({
            "animal_id": ["165721","165722","165941","165942",
                          "165961","165962","166401","166402",
                          "166451","166452","166531","166532",
                          "166541","166542","166571","166572",
                          "166581","166582",
                          "165701","165702","165711","165712",
                          "165931","165932","165951","165952",
                          "165971","165972","165981","165982",
                          "166001","166002","166381","166382",
                          "166441","166442"],
            "group": (["WT"]*18 + ["KO"]*18)
        })

    template = animals.copy()
    template["latency_first_seizure_min"] = np.nan
    template["n_stage4"]                  = np.nan
    template["duration_stage4_s"]         = np.nan
    template["n_stage5"]                  = np.nan
    template["duration_stage5_s"]         = np.nan
    template["total_stage45_duration_s"]  = np.nan
    template["max_racine_stage"]          = np.nan
    template["mean_racine_score"]         = np.nan
    template["total_seizure_duration_min"]= np.nan
    template["notes"]                     = ""

    template.to_csv(racine_path, index=False)
    print(f"Template saved: {racine_path}")
    print(f"Fill in the Racine scores and rerun this script.")
    print(f"\nAnimal IDs to fill ({len(template)}):")
    print(template[["animal_id","group"]].to_string(index=False))
    import sys; sys.exit(0)

# ── Load Racine scores ─────────────────────────────────────────────────────
racine = pd.read_csv(racine_path)
racine["animal_id"] = racine["animal_id"].astype(str)
racine_filled = racine.dropna(subset=["mean_racine_score"])
print(f"Loaded Racine scores: {len(racine_filled)}/{len(racine)} animals filled")

if len(racine_filled) < 4:
    print("Not enough data — fill in racine_scores.csv and rerun")
    import sys; sys.exit(0)

# ── Load EEG features ──────────────────────────────────────────────────────
# Use 3m baseline EEG and 4m KA EEG features from 9m portfolio
means_path = os.path.join(SRC_9M, "data", "animal_means_9m.csv")
means = pd.read_csv(means_path)
means["animal_id"] = means["animal_id"].astype(str)

# Also load complexity
comp_path = os.path.join(SRC_9M, "data", "complexity_longitudinal.csv")
if os.path.exists(comp_path):
    comp = pd.read_csv(comp_path)
    comp["animal_id"] = comp["animal_id"].astype(str)
    means = means.merge(
        comp[["animal_id","group","timepoint","lzc","peen","hjorth_mob","hjorth_comp"]],
        on=["animal_id","group","timepoint"],how="left",suffixes=("","_c"))

# 3m baseline EEG
bl3 = means[means["timepoint"]=="3m"].copy()
# 4m KA EEG
ka4 = means[means["timepoint"]=="4m"].copy()

# ── Merge with Racine ──────────────────────────────────────────────────────
eeg_cols_3m = ["aperiodic_exp","bp_gamma","bp_theta","bp_beta",
                "lzc","peen","hjorth_mob","wpli_theta","wpli_alpha"]
eeg_cols_4m = ["lzc","hjorth_comp","bp_beta","bp_theta","aperiodic_exp"]

eeg_cols_3m = [c for c in eeg_cols_3m if c in bl3.columns]
eeg_cols_4m = [c for c in eeg_cols_4m if c in ka4.columns]

m3 = racine_filled.merge(
    bl3[["animal_id","group"]+eeg_cols_3m],
    on=["animal_id","group"],how="inner")
m4 = racine_filled.merge(
    ka4[["animal_id","group"]+eeg_cols_4m],
    on=["animal_id","group"],how="inner")

print(f"\nMerged 3m EEG + Racine: n={len(m3)}")
print(f"Merged 4m EEG + Racine: n={len(m4)}")

# ── Correlations ───────────────────────────────────────────────────────────
racine_outcomes = ["total_stage45_duration_s","n_stage4","n_stage5",
                   "latency_first_seizure_min","mean_racine_score"]
racine_outcomes = [c for c in racine_outcomes if c in racine_filled.columns]

print("\nEEG-Racine correlations (p<0.10):")
print(f"{'EEG feature':<25} {'Racine outcome':<30} {'r':<8} {'p':<10} sig  n")
print("-"*78)

corr_rows = []
for tp_label, m, eeg_cols in [("3m",m3,eeg_cols_3m),("4m",m4,eeg_cols_4m)]:
    for eeg_col in eeg_cols:
        for rac_col in racine_outcomes:
            if rac_col not in m.columns: continue
            xy = m[[eeg_col,rac_col]].dropna()
            if len(xy)<4: continue
            r,p = spearmanr(xy[eeg_col],xy[rac_col])
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            corr_rows.append({"tp":tp_label,"eeg":eeg_col,"racine":rac_col,
                               "r":round(r,3),"p":round(p,4),"sig":sig,"n":len(xy)})
            if p<0.10:
                print(f"  [{tp_label}] {eeg_col:<23} {rac_col:<28} "
                      f"{r:<8.3f} {p:<10.4f} {sig}  n={len(xy)}")

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(os.path.join(RESULTS_DIR,"eeg_racine_correlations.csv"),index=False)
print(f"\nSaved: eeg_racine_correlations.csv")
print(f"Significant (p<0.05): {(corr_df['p']<0.05).sum()}")

# ── Group comparison ───────────────────────────────────────────────────────
print("\nRacine score group comparison (WT vs KO):")
for col in racine_outcomes:
    wt = racine_filled[racine_filled["group"]=="WT"][col].dropna()
    ko = racine_filled[racine_filled["group"]=="KO"][col].dropna()
    if len(wt)<2 or len(ko)<2: continue
    _,p = mannwhitneyu(wt,ko,alternative="two-sided")
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    print(f"  {col}: WT={wt.mean():.2f} KO={ko.mean():.2f} p={p:.4f} {sig}")

# ── Figure ─────────────────────────────────────────────────────────────────
sig_corrs = corr_df[corr_df["p"]<0.10].head(6)
if len(sig_corrs)>0:
    n_plots = len(sig_corrs)
    fig,axes=plt.subplots(1,n_plots,figsize=(n_plots*4,5))
    if n_plots==1: axes=[axes]
    for ax,(_,row) in zip(axes,sig_corrs.iterrows()):
        tp    = row["tp"]
        m_use = m3 if tp=="3m" else m4
        xy    = m_use[[row["eeg"],row["racine"],"group"]].dropna()
        for g,color,marker in [("WT",COLORS["WT"],"o"),("KO",COLORS["KO"],"s")]:
            mm = xy["group"]==g
            ax.scatter(xy[mm][row["eeg"]],xy[mm][row["racine"]],
                       color=color,marker=marker,s=65,alpha=0.8,label=g,zorder=3)
        mf,b = np.polyfit(xy[row["eeg"]],xy[row["racine"]],1)
        xl = np.linspace(xy[row["eeg"]].min(),xy[row["eeg"]].max(),100)
        ax.plot(xl,mf*xl+b,"k--",lw=1.5,alpha=0.6)
        ax.set_xlabel(f"{row['eeg']} ({tp})",fontsize=9)
        ax.set_ylabel(row["racine"].replace("_"," "),fontsize=9)
        ax.set_title(f"r={row['r']:.3f} {row['sig']}\np={row['p']:.4f}",fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("EEG features predict seizure severity (Racine scores)\n"
                 "C9orf72-KO vs WT",fontsize=11,y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR,"fig3_eeg_racine_correlation.png"),
                dpi=300,bbox_inches="tight")
    plt.close()
    print("Saved: fig3_eeg_racine_correlation.png")

print("\nRACINE CORRELATION ANALYSIS COMPLETE")
