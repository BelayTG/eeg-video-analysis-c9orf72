"""
01_sleep_state_classification.py
=================================
Classifies each 4-second EEG epoch as Wake, NREM, or REM
using delta/theta power ratio thresholds.

Method:
  Wake:  Low delta, variable theta, high broadband activity
  NREM:  High delta (0.5-4 Hz), low theta, low EMG-like activity
  REM:   Low delta, high theta (4-8 Hz), low broadband

Thresholds are computed per-animal (z-score normalized)
to account for individual differences in signal amplitude.

Both scenarios:
  n=36: session-level statistics
  n=18: animal-level (averaged across sessions)

Run:
    python src/01_sleep_state_classification.py
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import pyabf
from scipy.signal import welch, butter, filtfilt, decimate
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
PORT_DIR   = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR   = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")
for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

COLORS = {"WT": "#378ADD", "KO": "#D85A30"}
EPOCH_S = 4.0
FS_DS   = 500
MIN_EPOCHS_PER_STATE = 10

# ── Load inventory ─────────────────────────────────────────────────────────
inv_path = os.path.join(DATA_DIR, "file_inventory_3m.csv")
if not os.path.exists(inv_path):
    print("ERROR: Run 00_setup_and_inventory.py first")
    import sys; sys.exit(1)

inventory = pd.read_csv(inv_path)
print(f"Loaded inventory: {len(inventory)} files")
print(f"Animals: WT={inventory[inventory.group=='WT'].animal_id.nunique()} "
      f"KO={inventory[inventory.group=='KO'].animal_id.nunique()}")

# ── EEG helpers ────────────────────────────────────────────────────────────
def load_signal(abf_path, channel=0, target_fs=FS_DS):
    abf = pyabf.ABF(abf_path)
    fs  = float(abf.dataRate)
    abf.setSweep(0, channel=channel)
    sig = abf.sweepY.copy().astype(np.float64)
    del abf; gc.collect()
    if len(sig) < int(fs * 10):
        return None, None
    factor = int(fs / target_fs)
    if factor > 1:
        sig = decimate(sig - sig.mean(), factor, zero_phase=True)
    return sig, float(target_fs)

def band_power(psd, freqs, lo, hi):
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any(): return 0.0
    total = np.trapz(psd, freqs) + 1e-12
    return float(np.trapz(psd[m], freqs[m]) / total)

def compute_epoch_features(sig, fs, epoch_s=EPOCH_S):
    epoch_n = int(epoch_s * fs)
    n_epochs = len(sig) // epoch_n
    if n_epochs == 0: return pd.DataFrame()
    rows = []
    for i in range(n_epochs):
        ep = sig[i*epoch_n:(i+1)*epoch_n]
        if np.abs(ep).max() > 500: continue   # artifact
        if np.std(ep) < 0.001:    continue   # flat
        f, psd = welch(ep, fs=fs, nperseg=min(int(fs*2), epoch_n))
        rows.append({
            "epoch_idx": i,
            "onset_s":   i * epoch_s,
            "bp_delta":  band_power(psd, f, 0.5, 4),
            "bp_theta":  band_power(psd, f, 4, 8),
            "bp_alpha":  band_power(psd, f, 8, 13),
            "bp_beta":   band_power(psd, f, 13, 30),
            "bp_gamma":  band_power(psd, f, 30, 80),
            "total_var": float(np.var(ep)),
        })
    return pd.DataFrame(rows)

def classify_sleep_states(epoch_df):
    if len(epoch_df) < 10:
        epoch_df["state"] = "unknown"
        return epoch_df

    df = epoch_df.copy()

    # Normalize within-animal (z-score)
    for col in ["bp_delta","bp_theta","bp_alpha","bp_beta","bp_gamma","total_var"]:
        m = df[col].mean(); s = df[col].std() + 1e-8
        df[f"z_{col}"] = (df[col] - m) / s

    # Delta/theta ratio
    df["dt_ratio"] = df["bp_delta"] / (df["bp_theta"] + 1e-8)
    dt_med = df["dt_ratio"].median()

    # Variance for wake detection
    var_75 = df["total_var"].quantile(0.75)
    var_25 = df["total_var"].quantile(0.25)

    states = []
    for _, row in df.iterrows():
        # Wake: high variance OR high beta/gamma
        if (row["total_var"] > var_75 or
            row["z_bp_beta"] > 1.0 or
            row["z_bp_gamma"] > 1.0):
            states.append("Wake")
        # REM: low delta, high theta, low variance
        elif (row["dt_ratio"] < dt_med * 0.5 and
              row["z_bp_theta"] > 0 and
              row["total_var"] < var_75):
            states.append("REM")
        # NREM: high delta, low theta
        elif row["dt_ratio"] > dt_med:
            states.append("NREM")
        # Default to Wake for ambiguous
        else:
            states.append("Wake")

    df["state"] = states
    return df

# ── Main processing ────────────────────────────────────────────────────────
print("\nProcessing EEG files for sleep state classification...")
print("="*60)

all_epoch_rows = []
session_summary = []

for _, file_row in inventory.iterrows():
    abf_path   = file_row["abf_path"]
    animal_id  = str(file_row["animal_id"])
    session_id = str(file_row["session_id"])
    group      = file_row["group"]
    abf_file   = file_row["abf_file"]

    if not os.path.exists(abf_path):
        print(f"  SKIP (not found): {abf_path}")
        continue

    try:
        sig, fs = load_signal(abf_path, channel=0)
        if sig is None:
            continue

        ep_df = compute_epoch_features(sig, fs)
        if len(ep_df) < MIN_EPOCHS_PER_STATE:
            continue

        ep_df = classify_sleep_states(ep_df)
        ep_df["animal_id"]  = animal_id
        ep_df["session_id"] = session_id
        ep_df["group"]      = group
        ep_df["abf_file"]   = abf_file
        ep_df["timepoint"]  = "3m"

        all_epoch_rows.append(ep_df)

        # Session summary
        state_counts = ep_df["state"].value_counts()
        total = len(ep_df)
        summary = {
            "animal_id":   animal_id,
            "session_id":  session_id,
            "group":       group,
            "abf_file":    abf_file,
            "n_epochs":    total,
            "n_wake":      state_counts.get("Wake", 0),
            "n_nrem":      state_counts.get("NREM", 0),
            "n_rem":       state_counts.get("REM", 0),
            "pct_wake":    round(state_counts.get("Wake",0)/total*100, 1),
            "pct_nrem":    round(state_counts.get("NREM",0)/total*100, 1),
            "pct_rem":     round(state_counts.get("REM", 0)/total*100, 1),
        }
        session_summary.append(summary)
        del sig; gc.collect()

        print(f"  {session_id} {abf_file}: "
              f"W={summary['pct_wake']}% "
              f"N={summary['pct_nrem']}% "
              f"R={summary['pct_rem']}%")

    except Exception as e:
        print(f"  ERROR {abf_file}: {e}")

# ── Save epoch-level data ──────────────────────────────────────────────────
all_epochs = pd.concat(all_epoch_rows, ignore_index=True)
all_epochs.to_csv(os.path.join(DATA_DIR, "epochs_with_states_3m.csv"),
                   index=False)
print(f"\nSaved: epochs_with_states_3m.csv ({len(all_epochs)} epochs)")

summary_df = pd.DataFrame(session_summary)
summary_df.to_csv(os.path.join(DATA_DIR, "sleep_session_summary_3m.csv"),
                   index=False)
print(f"Saved: sleep_session_summary_3m.csv")

# ── Compute per-session state-specific EEG features ───────────────────────
print("\nComputing state-specific EEG features...")

state_feat_rows = []
for _, file_row in inventory.iterrows():
    abf_path   = file_row["abf_path"]
    animal_id  = str(file_row["animal_id"])
    session_id = str(file_row["session_id"])
    group      = file_row["group"]

    # Get epochs for this file
    mask = ((all_epochs["session_id"]==session_id) &
            (all_epochs["abf_file"]==file_row["abf_file"]))
    ep_sub = all_epochs[mask]
    if len(ep_sub) == 0: continue

    row = {"animal_id":animal_id,"session_id":session_id,"group":group,"timepoint":"3m"}
    for state in ["Wake","NREM","REM","All"]:
        if state == "All":
            sub = ep_sub
        else:
            sub = ep_sub[ep_sub["state"]==state]
        if len(sub) < MIN_EPOCHS_PER_STATE: continue
        prefix = state.lower() + "_"
        for feat in ["bp_delta","bp_theta","bp_alpha","bp_beta","bp_gamma"]:
            row[f"{prefix}{feat}"] = float(sub[feat].mean())
        row[f"{prefix}n_epochs"] = len(sub)
    state_feat_rows.append(row)

state_feats = pd.DataFrame(state_feat_rows)
state_feats.to_csv(os.path.join(DATA_DIR,
    "state_specific_features_3m.csv"), index=False)
print(f"Saved: state_specific_features_3m.csv ({len(state_feats)} sessions)")

# ── Statistics — Scenario A (n=36 sessions) ───────────────────────────────
print("\nSTATISTICS — Scenario A (n=36 session-level)")
print("="*60)

feat_cols_A = [c for c in state_feats.columns
               if c not in ["animal_id","session_id","group","timepoint"]
               and not c.endswith("_n_epochs")]

stat_rows_A = []
for feat in feat_cols_A:
    wt = state_feats[state_feats["group"]=="WT"][feat].dropna()
    ko = state_feats[state_feats["group"]=="KO"][feat].dropna()
    if len(wt)<2 or len(ko)<2: continue
    _,p = mannwhitneyu(wt,ko,alternative="two-sided")
    d   = (wt.mean()-ko.mean())/np.sqrt((wt.std()**2+ko.std()**2)/2+1e-10)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    stat_rows_A.append({"feature":feat,"wt_mean":round(float(wt.mean()),5),
                          "ko_mean":round(float(ko.mean()),5),
                          "pval":round(p,4),"cohens_d":round(float(d),3),
                          "sig":sig,"n_wt":len(wt),"n_ko":len(ko)})

stats_A = pd.DataFrame(stat_rows_A)
if len(stats_A)>1:
    rej,pfd,_,_ = multipletests(stats_A["pval"],method="fdr_bh",alpha=0.05)
    stats_A["pval_fdr"]=pfd; stats_A["fdr_sig"]=rej
stats_A.to_csv(os.path.join(RESULTS_DIR,"statistics_n36_3m.csv"),index=False)

sig_A = stats_A[stats_A["pval"]<0.05].sort_values("pval")
print(f"Significant (p<0.05): {len(sig_A)}/{len(stats_A)}")
for _,r in sig_A.iterrows():
    fdr=" FDR*" if r.get("fdr_sig",False) else ""
    print(f"  {r.feature}: WT={r.wt_mean:.4f} KO={r.ko_mean:.4f} "
          f"p={r.pval:.4f} {r.sig}{fdr}")

# ── Statistics — Scenario B (n=18 animals) ────────────────────────────────
print("\nSTATISTICS — Scenario B (n=18 animal-level)")
print("="*60)

# Average sessions within animal
animal_feats = (state_feats.groupby(["animal_id","group"])[feat_cols_A]
                .mean().reset_index())

stat_rows_B = []
for feat in feat_cols_A:
    wt = animal_feats[animal_feats["group"]=="WT"][feat].dropna()
    ko = animal_feats[animal_feats["group"]=="KO"][feat].dropna()
    if len(wt)<2 or len(ko)<2: continue
    _,p = mannwhitneyu(wt,ko,alternative="two-sided")
    d   = (wt.mean()-ko.mean())/np.sqrt((wt.std()**2+ko.std()**2)/2+1e-10)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    stat_rows_B.append({"feature":feat,"wt_mean":round(float(wt.mean()),5),
                          "ko_mean":round(float(ko.mean()),5),
                          "pval":round(p,4),"cohens_d":round(float(d),3),
                          "sig":sig,"n_wt":len(wt),"n_ko":len(ko)})

stats_B = pd.DataFrame(stat_rows_B)
if len(stats_B)>1:
    rej,pfd,_,_ = multipletests(stats_B["pval"],method="fdr_bh",alpha=0.05)
    stats_B["pval_fdr"]=pfd; stats_B["fdr_sig"]=rej
stats_B.to_csv(os.path.join(RESULTS_DIR,"statistics_n18_3m.csv"),index=False)

sig_B = stats_B[stats_B["pval"]<0.05].sort_values("pval")
print(f"Significant (p<0.05): {len(sig_B)}/{len(stats_B)}")
for _,r in sig_B.iterrows():
    fdr=" FDR*" if r.get("fdr_sig",False) else ""
    print(f"  {r.feature}: WT={r.wt_mean:.4f} KO={r.ko_mean:.4f} "
          f"p={r.pval:.4f} {r.sig}{fdr}")

# ── Figures ────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

# Fig 1: Sleep state proportions
fig,axes=plt.subplots(1,3,figsize=(13,5))
for ax,state,color_s in zip(axes,["Wake","NREM","REM"],
                               ["#E67E22","#3498DB","#9B59B6"]):
    pct_col = f"pct_{state.lower()}"
    if pct_col not in summary_df.columns: continue
    # Scenario A
    wt = summary_df[summary_df["group"]=="WT"][pct_col].dropna()
    ko = summary_df[summary_df["group"]=="KO"][pct_col].dropna()
    for g,color,marker,x in [("WT",COLORS["WT"],"o",0),
                               ("KO",COLORS["KO"],"s",1)]:
        vals = summary_df[summary_df["group"]==g][pct_col].dropna()
        ax.scatter([x]*len(vals),vals,color=color,alpha=0.5,
                   s=40,marker=marker,zorder=3)
        ax.errorbar(x,vals.mean(),yerr=vals.sem(),
                    fmt="_",markersize=24,markeredgewidth=2.5,
                    color=color,capsize=4,capthick=2)
    if len(wt)>=2 and len(ko)>=2:
        _,p=mannwhitneyu(wt,ko,alternative="two-sided")
        sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        color_sig="red" if sig!="ns" else "#555555"
        ymax=max(wt.max(),ko.max())
        rng=ymax-min(wt.min(),ko.min())
        ax.set_ylim(min(wt.min(),ko.min())-rng*0.05,ymax+rng*0.40)
        ax.annotate(f"{sig}\np={p:.3f}",xy=(0.5,0.90),
                    xycoords="axes fraction",ha="center",fontsize=10,
                    color=color_sig,
                    bbox=dict(boxstyle="round,pad=0.2",facecolor="white",
                              alpha=0.85,edgecolor="lightgray"))
    ax.set_xticks([0,1]); ax.set_xticklabels(["WT","KO"])
    ax.set_ylabel(f"% time in {state}",fontsize=10)
    ax.set_title(f"{state} state\n3m baseline",fontsize=10)
fig.suptitle("Sleep state proportions at 3m — C9orf72-KO vs WT\n"
             "Scenario A (n=36 sessions)",fontsize=11,y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR,"fig1_sleep_proportions_3m.png"),
            dpi=300,bbox_inches="tight")
plt.close()
print("  Saved: fig1_sleep_proportions_3m.png")

# Fig 2: State-specific band power comparison
key_feats = [
    ("wake_bp_theta","Wake Theta"),("wake_bp_beta","Wake Beta"),
    ("nrem_bp_delta","NREM Delta"),("nrem_bp_theta","NREM Theta"),
    ("rem_bp_theta","REM Theta"), ("rem_bp_delta","REM Delta"),
]
key_feats = [(f,t) for f,t in key_feats if f in state_feats.columns]
if key_feats:
    fig,axes=plt.subplots(2,3,figsize=(13,9))
    for ax,(feat,title) in zip(axes.flat,key_feats):
        wt=state_feats[state_feats["group"]=="WT"][feat].dropna()
        ko=state_feats[state_feats["group"]=="KO"][feat].dropna()
        all_v=pd.concat([wt,ko])
        rng=all_v.max()-all_v.min()
        ax.set_ylim(all_v.min()-rng*0.05,all_v.max()+rng*0.40)
        for g,color,marker,x in [("WT",COLORS["WT"],"o",0),
                                   ("KO",COLORS["KO"],"s",1)]:
            vals=state_feats[state_feats["group"]==g][feat].dropna()
            ax.scatter([x]*len(vals),vals,color=color,alpha=0.6,
                       s=50,marker=marker,zorder=3)
            if len(vals)>=1:
                ax.errorbar(x,vals.mean(),
                            yerr=vals.sem() if len(vals)>1 else 0,
                            fmt="_",markersize=22,markeredgewidth=2.5,
                            color=color,capsize=4,capthick=2)
        if len(wt)>=2 and len(ko)>=2:
            _,p=mannwhitneyu(wt,ko,alternative="two-sided")
            sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            color_sig="red" if sig!="ns" else "#555555"
            ax.annotate(f"{sig}\np={p:.3f}\nWT n={len(wt)} KO n={len(ko)}",
                        xy=(0.5,0.88),xycoords="axes fraction",
                        ha="center",fontsize=9,color=color_sig,
                        bbox=dict(boxstyle="round,pad=0.2",facecolor="white",
                                  alpha=0.85,edgecolor="lightgray"))
        ax.set_xticks([0,1]); ax.set_xticklabels(["WT","KO"])
        ax.set_title(title,fontsize=10)
    for ax in list(axes.flat)[len(key_feats):]: ax.set_visible(False)
    fig.suptitle("State-specific EEG band power at 3m — C9orf72-KO vs WT\n"
                 "Scenario A (n=36 sessions)",fontsize=11,y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR,"fig2_state_specific_power_3m.png"),
                dpi=300,bbox_inches="tight")
    plt.close()
    print("  Saved: fig2_state_specific_power_3m.png")

print("\nSLEEP STATE CLASSIFICATION COMPLETE")
print(f"Outputs in: {PORT_DIR}")
