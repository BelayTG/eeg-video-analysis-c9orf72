"""
05_sleep_classification_all_timepoints.py
==========================================
Sleep state classification and comprehensive EEG feature extraction
across all 6 timepoints, for BOTH scenarios.

SCENARIO A — Animal-level (5-digit animal_id as statistical unit)
  - All ABF files for an animal at a timepoint are pooled together
  - Feature averages computed across all files → one value per animal
  - Statistics: n = number of animals (WT vs KO per timepoint)

SCENARIO B — Session-level (6-digit session_id as statistical unit)
  - Files split into Session 1 (first half) and Session 2 (second half)
  - EXCEPT 4m: ALL files duplicated into both sessions (identical)
  - Feature averages computed per session → two values per animal
  - Statistics: n = 2 × number of animals

Features extracted per 4-second epoch:
  Spectral:   delta/theta/alpha/beta/gamma (absolute + relative)
              theta/delta ratio, spectral entropy, aperiodic exponent
              theta/alpha ratio, slow/fast ratio, high-frequency index
  Temporal:   variance, RMS, peak-to-peak, zero-crossing rate, CV
  Complexity: Hjorth mobility/complexity, Lempel-Ziv complexity

Outputs (per timepoint, per scenario):
  data/epochs_with_states_{tp}.csv            epoch-level (shared)
  data/sleep_session_summary_{tp}_{sc}.csv    file-level sleep %
  data/state_specific_features_{tp}_{sc}.csv  aggregated features
  results/statistics_{tp}_{sc}.csv            WT vs KO stats

Run:
    python src/05_sleep_classification_all_timepoints.py
    python src/05_sleep_classification_all_timepoints.py --timepoint 3m
    python src/05_sleep_classification_all_timepoints.py --timepoint 4m --scenario B
"""

import os, gc, sys, argparse, warnings
import numpy as np
import pandas as pd
import pyabf
from scipy.signal import welch, decimate, butter
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.signal import sosfiltfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")
for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

COLORS          = {"WT": "#378ADD", "KO": "#D85A30"}
EPOCH_S         = 4.0
FS_DS           = 500
MIN_EPOCHS      = 10
ARTIFACT_THRESH = 500
FLAT_THRESH     = 0.001

TP_ORDER        = ["3m", "4m", "6m", "7m", "9m", "12m"]
TP_4M_DUPLICATE = "4m"      # special duplication rule for Scenario B

SCENARIOS       = ["A", "B"]   # run both by default

# ── Signal helpers ─────────────────────────────────────────────────────────

def load_signal(abf_path, channel=0, target_fs=FS_DS):
    try:
        abf = pyabf.ABF(abf_path)
        fs  = float(abf.dataRate)
        abf.setSweep(0, channel=channel)
        sig = abf.sweepY.copy().astype(np.float64)
        del abf; gc.collect()
        if len(sig) < int(fs * 30):
            return None, None
        factor = max(1, int(round(fs / target_fs)))
        if factor > 1:
            sig = decimate(sig - sig.mean(), factor, zero_phase=True)
        return sig, float(target_fs)
    except Exception:
        return None, None


def band_power_abs(psd, freqs, lo, hi):
    m = (freqs >= lo) & (freqs <= hi)
    return float(np.trapz(psd[m], freqs[m])) if m.any() else 0.0


def aperiodic_exponent(psd, freqs, lo=2, hi=40):
    m = (freqs >= lo) & (freqs <= hi) & (psd > 0)
    if m.sum() < 5:
        return np.nan
    slope, _ = np.polyfit(np.log10(freqs[m]), np.log10(psd[m]), 1)
    return float(-slope)


def lzc(sig):
    s = (sig > np.median(sig)).astype(int).tolist()
    n = len(s)
    i, c, l, k = 0, 1, 1, 1
    while True:
        if s[i + k - 1] != s[l + k - 1]:
            if i + k > l:
                c += 1; l = i + k; i = 0; k = 1
            else:
                i += 1; k = 1
        else:
            k += 1
            if l + k > n:
                c += 1; break
    return c * np.log2(n + 1) / n if n > 1 else 0.0


def hjorth(sig):
    d1 = np.diff(sig); d2 = np.diff(d1)
    v0 = np.var(sig) + 1e-12
    v1 = np.var(d1)  + 1e-12
    v2 = np.var(d2)  + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = (np.sqrt(v2 / v1) / mob) if mob > 0 else 0.0
    return float(mob), float(comp)


def zero_cr(sig):
    s = np.sign(sig - np.mean(sig))
    return float(np.sum(np.diff(s) != 0) / len(sig))


def compute_epoch_features(sig, fs, epoch_s=EPOCH_S):
    epoch_n  = int(epoch_s * fs)
    n_epochs = len(sig) // epoch_n
    rows = []
    for i in range(n_epochs):
        ep = sig[i*epoch_n:(i+1)*epoch_n]
        if np.abs(ep).max() > ARTIFACT_THRESH or np.std(ep) < FLAT_THRESH:
            continue
        nperseg = min(int(fs * 2), epoch_n)
        f, psd  = welch(ep, fs=fs, nperseg=nperseg, detrend="constant")

        bp_d = band_power_abs(psd, f, 0.5,  4)
        bp_t = band_power_abs(psd, f, 4,    8)
        bp_a = band_power_abs(psd, f, 8,   13)
        bp_b = band_power_abs(psd, f, 13,  30)
        bp_g = band_power_abs(psd, f, 30,  80)
        tot  = bp_d + bp_t + bp_a + bp_b + bp_g + 1e-12

        rd, rt, ra, rb, rg = bp_d/tot, bp_t/tot, bp_a/tot, bp_b/tot, bp_g/tot

        bp_arr = np.clip([rd, rt, ra, rb, rg], 1e-12, None)
        bp_arr /= bp_arr.sum()
        sp_ent = float(-np.sum(bp_arr * np.log2(bp_arr)) / np.log2(5))

        mob, comp = hjorth(ep)

        rows.append({
            "epoch_idx":        i,
            "onset_s":          i * epoch_s,
            # absolute band power
            "bp_delta": bp_d, "bp_theta": bp_t, "bp_alpha": bp_a,
            "bp_beta":  bp_b, "bp_gamma": bp_g,
            # relative band power
            "rbp_delta": rd, "rbp_theta": rt, "rbp_alpha": ra,
            "rbp_beta":  rb, "rbp_gamma": rg,
            # ratios
            "td_ratio":      bp_t / (bp_d + 1e-12),
            "theta_alpha":   rt   / (ra  + 1e-12),
            "theta_beta":    rt   / (rb  + 1e-12),
            "slow_fast":     (bp_d + bp_t) / (bp_b + bp_g + 1e-12),
            "high_freq_idx": (bp_b + bp_g) / tot,
            # spectral
            "spectral_entropy": sp_ent,
            "ap_exp":           aperiodic_exponent(psd, f),
            # time-domain
            "total_var": float(np.var(ep)),
            "rms":       float(np.sqrt(np.mean(ep**2))),
            "p2p":       float(np.ptp(ep)),
            "zcr":       zero_cr(ep),
            "coeff_var": float(np.std(ep) / (np.abs(np.mean(ep)) + 1e-12)),
            # complexity
            "hjorth_mob":  mob,
            "hjorth_comp": comp,
            "lzc":         lzc(ep[::4]),   # downsample 500→125 Hz for speed
        })
    return pd.DataFrame(rows)


def classify_states(epoch_df):
    if len(epoch_df) < MIN_EPOCHS:
        epoch_df["state"] = "unknown"
        return epoch_df
    df = epoch_df.copy()
    for col in ["bp_delta","bp_theta","bp_alpha","bp_beta","bp_gamma",
                "total_var","hjorth_mob"]:
        if col in df.columns:
            m = df[col].mean(); s = df[col].std() + 1e-8
            df[f"z_{col}"] = (df[col] - m) / s
    dtm    = df["td_ratio"].median()
    var_75 = df["total_var"].quantile(0.75)
    states = []
    for _, r in df.iterrows():
        if (r["total_var"] > var_75 or
                r.get("z_bp_beta",  0) > 1.0 or
                r.get("z_bp_gamma", 0) > 1.0 or
                r.get("z_hjorth_mob", 0) > 1.2):
            states.append("Wake")
        elif (r["td_ratio"] < dtm * 0.5 and
              r.get("z_bp_theta", 0) > 0 and
              r["total_var"] < var_75):
            states.append("REM")
        elif r["td_ratio"] > dtm:
            states.append("NREM")
        else:
            states.append("Wake")
    df["state"] = states
    return df


FEAT_COLS = [
    "bp_delta","bp_theta","bp_alpha","bp_beta","bp_gamma",
    "rbp_delta","rbp_theta","rbp_alpha","rbp_beta","rbp_gamma",
    "td_ratio","theta_alpha","theta_beta","slow_fast","high_freq_idx",
    "spectral_entropy","ap_exp",
    "total_var","rms","p2p","zcr","coeff_var",
    "hjorth_mob","hjorth_comp","lzc",
]

# ── Core: process one ABF file → epoch DataFrame ──────────────────────────

def process_one_file(abf_path, animal_id, session_id, group, abf_file, tp):
    """Load, epoch, classify one ABF. Returns epoch DataFrame or None."""
    sig, fs = load_signal(abf_path)
    if sig is None:
        return None
    ep_df = compute_epoch_features(sig, fs)
    del sig; gc.collect()
    if len(ep_df) < MIN_EPOCHS:
        return None
    ep_df = classify_states(ep_df)
    ep_df["animal_id"]  = animal_id
    ep_df["session_id"] = session_id
    ep_df["group"]      = group
    ep_df["abf_file"]   = abf_file
    ep_df["timepoint"]  = tp
    return ep_df


# ── Aggregate epoch-level data → session/animal-level features ────────────

def aggregate_features(all_epochs_df, unit_col):
    """
    Average FEAT_COLS within each (unit_col, group, timepoint, state).
    unit_col is 'animal_id' (Scenario A) or 'session_id' (Scenario B).
    Returns wide DataFrame: one row per unit per ABF file per state.
    """
    rows = []
    # Iterate per unit × ABF file (keep file-level granularity first)
    for (unit, group, tp, abf_file), sub in all_epochs_df.groupby(
            [unit_col, "group", "timepoint", "abf_file"]):
        row = {unit_col: unit, "group": group,
               "timepoint": tp, "abf_file": abf_file}
        # Also carry animal_id if unit_col is session_id
        if unit_col == "session_id":
            row["animal_id"] = sub["animal_id"].iloc[0]

        for state in ["Wake", "NREM", "REM", "All"]:
            sub_s = sub if state == "All" else sub[sub.state == state]
            if len(sub_s) < MIN_EPOCHS:
                continue
            prefix = state.lower() + "_"
            for feat in FEAT_COLS:
                if feat in sub_s.columns:
                    row[f"{prefix}{feat}"]     = float(sub_s[feat].mean())
                    row[f"{prefix}{feat}_std"] = float(sub_s[feat].std())
            row[f"{prefix}n_epochs"] = len(sub_s)
            row[f"{prefix}pct"]      = len(sub_s) / len(sub) * 100

        rows.append(row)

    return pd.DataFrame(rows)


def run_statistics(feat_df, unit_col, tp, scenario_label):
    """Mann-Whitney U at the correct statistical unit level with FDR."""
    feat_cols_stat = [c for c in feat_df.columns
                      if c not in [unit_col,"animal_id","group","timepoint","abf_file"]
                      and not c.endswith("_std") and not c.endswith("_n_epochs")
                      and not c.endswith("_pct")]

    # Average across ABF files within unit
    unit_means = feat_df.groupby([unit_col,"group"])[feat_cols_stat].mean().reset_index()

    rows = []
    for feat in feat_cols_stat:
        wt = unit_means[unit_means.group=="WT"][feat].dropna()
        ko = unit_means[unit_means.group=="KO"][feat].dropna()
        if len(wt) < 2 or len(ko) < 2:
            continue
        _, p = mannwhitneyu(wt, ko, alternative="two-sided")
        d = (ko.mean()-wt.mean()) / np.sqrt((wt.std()**2+ko.std()**2)/2+1e-10)
        rows.append({
            "feature": feat, "timepoint": tp, "scenario": scenario_label,
            "unit": unit_col,
            "wt_mean": round(float(wt.mean()),6), "wt_std": round(float(wt.std()),6),
            "ko_mean": round(float(ko.mean()),6), "ko_std": round(float(ko.std()),6),
            "cohens_d": round(float(d),4), "pval": round(p,6),
            "n_wt": len(wt), "n_ko": len(ko),
        })

    stats = pd.DataFrame(rows)
    if len(stats) > 1:
        _, pfd, _, _ = multipletests(stats["pval"], method="fdr_bh")
        stats["pval_fdr"] = pfd.round(6)
    stats["sig"] = stats["pval"].apply(
        lambda x: "***" if x<0.001 else "**" if x<0.01 else "*" if x<0.05 else "ns")

    out = os.path.join(RESULTS_DIR, f"statistics_{tp}_{scenario_label}.csv")
    stats.to_csv(out, index=False)

    sig = stats[stats.pval < 0.05].sort_values("pval")
    print(f"  [{scenario_label}] Significant (p<0.05): {len(sig)}/{len(stats)}")
    for _, r in sig.head(8).iterrows():
        fdr = " FDR*" if r.get("pval_fdr", 1) < 0.05 else ""
        print(f"    {r.feature:<35} WT={r.wt_mean:.5f} KO={r.ko_mean:.5f} "
              f"d={r.cohens_d:.3f} p={r.pval:.5f}{fdr}")
    return stats


# ── Main per-timepoint pipeline ───────────────────────────────────────────

def process_timepoint(tp, scenarios_to_run=None):
    if scenarios_to_run is None:
        scenarios_to_run = SCENARIOS

    print(f"\n{'='*65}")
    print(f"Timepoint: {tp}")
    if tp == TP_4M_DUPLICATE:
        print("  4m special: Scenario B will duplicate files into both sessions")
    print(f"{'='*65}")

    # ── Load inventories ──────────────────────────────────────────────────
    inv = {}
    for sc in scenarios_to_run:
        path = os.path.join(DATA_DIR, f"file_inventory_{tp}_{sc}.csv")
        if not os.path.exists(path):
            # Try combined inventory
            path_all = os.path.join(DATA_DIR, "file_inventory_all_timepoints.csv")
            if os.path.exists(path_all):
                df_all = pd.read_csv(path_all)
                inv[sc] = df_all[(df_all.timepoint==tp) & (df_all.scenario==sc)].copy()
            else:
                print(f"  No inventory for {tp} Scenario {sc} — run script 04 first")
                inv[sc] = pd.DataFrame()
        else:
            inv[sc] = pd.read_csv(path)

    # Collect all unique ABF paths to avoid processing duplicates
    all_abf_paths = set()
    for sc in scenarios_to_run:
        if not inv[sc].empty:
            all_abf_paths.update(inv[sc]["abf_path"].tolist())

    if not all_abf_paths:
        print(f"  No ABF files found for {tp}")
        return

    print(f"\n  Unique ABF files to process: {len(all_abf_paths)}")

    # ── Process each ABF once (shared across both scenarios) ─────────────
    # We process once and tag each epoch with both animal_id AND session_id
    # Then aggregate differently for each scenario.

    # Build a lookup: abf_path → {animal_id, session_id_A, session_id_B, group}
    abf_meta = {}
    for sc in scenarios_to_run:
        for _, row in inv[sc].iterrows():
            abf = row["abf_path"]
            if abf not in abf_meta:
                abf_meta[abf] = {
                    "animal_id": str(row["animal_id"]),
                    "group":     row["group"],
                    "abf_file":  row["abf_file"],
                }
            abf_meta[abf][f"session_id_{sc}"] = str(row["session_id"])

    all_epoch_rows = []

    for abf_path, meta in abf_meta.items():
        if not os.path.exists(abf_path):
            print(f"  SKIP (not found): {abf_path}")
            continue

        animal_id = meta["animal_id"]
        group     = meta["group"]
        abf_file  = meta["abf_file"]

        # Use session_id from Scenario B as the file-level tag
        # (Scenario A ignores session_id anyway)
        session_id_A = meta.get("session_id_A", animal_id)
        session_id_B = meta.get("session_id_B", f"{animal_id}1")

        sig, fs = load_signal(abf_path)
        if sig is None:
            print(f"  SKIP (bad signal): {abf_file}")
            continue

        ep_df = compute_epoch_features(sig, fs)
        del sig; gc.collect()

        if len(ep_df) < MIN_EPOCHS:
            print(f"  SKIP (too few epochs {len(ep_df)}): {abf_file}")
            continue

        ep_df = classify_states(ep_df)

        # Tag with all IDs — will be used for both scenario aggregations
        ep_df["animal_id"]     = animal_id
        ep_df["session_id_A"]  = session_id_A    # always = animal_id for Sc A
        ep_df["session_id_B"]  = session_id_B    # session 1 or 2 for Sc B
        ep_df["group"]         = group
        ep_df["abf_file"]      = abf_file
        ep_df["timepoint"]     = tp

        all_epoch_rows.append(ep_df)

        # Quick progress
        vc = ep_df.state.value_counts()
        n  = len(ep_df)
        print(f"  {abf_file[:20]:<20} [{group}] {animal_id} "
              f"W={vc.get('Wake',0)/n*100:.0f}% "
              f"N={vc.get('NREM',0)/n*100:.0f}% "
              f"R={vc.get('REM',0)/n*100:.0f}% "
              f"({n} epochs)")

    if not all_epoch_rows:
        print(f"  No epochs processed for {tp}")
        return

    all_epochs = pd.concat(all_epoch_rows, ignore_index=True)

    # Save epoch-level data once (shared)
    ep_out = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
    all_epochs.to_csv(ep_out, index=False)
    print(f"\n  Saved: epochs_with_states_{tp}.csv ({len(all_epochs):,} epochs)")

    # ── Aggregate and statistics for each scenario ─────────────────────────
    for sc in scenarios_to_run:
        print(f"\n  --- Scenario {sc} ---")
        if sc == "A":
            unit_col   = "session_id_A"   # = animal_id in the epoch frame
            label_col  = "animal_id"      # what we call it in output
        else:
            unit_col   = "session_id_B"
            label_col  = "session_id"

        # Build a version of epochs with the right unit column named consistently
        ep_sc = all_epochs.copy()
        ep_sc["_unit"] = ep_sc[unit_col]

        # Aggregate: file-level features within unit
        feat_rows = []
        for (unit_val, group, abf_file), sub in ep_sc.groupby(
                ["_unit", "group", "abf_file"]):
            row = {label_col: unit_val, "group": group,
                   "timepoint": tp, "abf_file": abf_file}
            if sc == "B":
                row["animal_id"] = sub["animal_id"].iloc[0]

            for state in ["Wake","NREM","REM","All"]:
                sub_s = sub if state == "All" else sub[sub.state == state]
                if len(sub_s) < MIN_EPOCHS:
                    continue
                prefix = state.lower() + "_"
                for feat in FEAT_COLS:
                    if feat in sub_s.columns:
                        row[f"{prefix}{feat}"]     = float(sub_s[feat].mean())
                        row[f"{prefix}{feat}_std"] = float(sub_s[feat].std())
                row[f"{prefix}n_epochs"] = len(sub_s)
                row[f"{prefix}pct"]      = len(sub_s) / len(sub) * 100
            feat_rows.append(row)

        feat_df = pd.DataFrame(feat_rows)

        # Save features
        feat_out = os.path.join(DATA_DIR, f"state_specific_features_{tp}_{sc}.csv")
        feat_df.to_csv(feat_out, index=False)
        print(f"  Saved: state_specific_features_{tp}_{sc}.csv "
              f"({feat_df[label_col].nunique()} {label_col}s, {len(feat_df)} rows)")

        # Session-level sleep summary
        sleep_rows = []
        for (unit_val, group, abf_file), sub in ep_sc.groupby(
                ["_unit","group","abf_file"]):
            vc = sub.state.value_counts(); n = len(sub)
            sr = {label_col: unit_val, "group": group,
                  "timepoint": tp, "abf_file": abf_file,
                  "n_epochs": n,
                  "pct_wake": vc.get("Wake",0)/n*100,
                  "pct_nrem": vc.get("NREM",0)/n*100,
                  "pct_rem":  vc.get("REM", 0)/n*100}
            if sc == "B":
                sr["animal_id"] = sub["animal_id"].iloc[0]
            sleep_rows.append(sr)

        sleep_df = pd.DataFrame(sleep_rows)
        sleep_df.to_csv(
            os.path.join(DATA_DIR, f"sleep_session_summary_{tp}_{sc}.csv"),
            index=False)

        # Statistics
        print(f"\n  Statistics [{sc}] — WT vs KO:")
        run_statistics(feat_df, label_col, tp, sc)

        # QC figure
        _plot_sleep_qc(sleep_df, label_col, tp, sc)

    return all_epochs


def _plot_sleep_qc(sleep_df, unit_col, tp, sc):
    """QC figure: sleep state proportions per group."""
    # Average per unit (animal or session)
    unit_means = sleep_df.groupby([unit_col,"group"])[["pct_wake","pct_nrem","pct_rem"]].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (col, state) in zip(axes, [("pct_wake","Wake"),
                                        ("pct_nrem","NREM"),
                                        ("pct_rem", "REM")]):
        for g, color, x in [("WT",COLORS["WT"],0),("KO",COLORS["KO"],1)]:
            vals = unit_means[unit_means.group==g][col].dropna()
            ax.scatter([x]*len(vals), vals, color=color, alpha=0.5, s=40, zorder=3)
            ax.errorbar(x, vals.mean(), yerr=vals.sem() if len(vals)>1 else 0,
                        fmt="_", markersize=20, markeredgewidth=2,
                        color=color, capsize=4, capthick=2)
        wt = unit_means[unit_means.group=="WT"][col].dropna()
        ko = unit_means[unit_means.group=="KO"][col].dropna()
        if len(wt)>=2 and len(ko)>=2:
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            sig  = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            ax.set_title(f"{state} — {sig} p={p:.3f}", fontsize=9)
        else:
            ax.set_title(state, fontsize=9)
        ax.set_xticks([0,1]); ax.set_xticklabels(["WT","KO"])
        ax.set_ylabel(f"% {state}", fontsize=9)

    unit_label = "animal" if sc=="A" else "session"
    fig.suptitle(f"Sleep proportions — {tp} | Scenario {sc} (n per {unit_label})", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"qc_sleep_{tp}_{sc}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: qc_sleep_{tp}_{sc}.png")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timepoint", default=None,
                        help="Single timepoint (e.g. 3m). Default: all.")
    parser.add_argument("--scenario", default=None,
                        help="A or B. Default: both.")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIOS
    tp_list   = [args.timepoint] if args.timepoint else TP_ORDER

    # Verify inventories exist
    inv_all = os.path.join(DATA_DIR, "file_inventory_all_timepoints.csv")
    inv_A   = os.path.join(DATA_DIR, "file_inventory_scenarioA.csv")
    if not os.path.exists(inv_all) and not os.path.exists(inv_A):
        print("ERROR: Run script 04_multi_timepoint_inventory.py first.")
        sys.exit(1)

    print("=" * 65)
    print("Sleep Classification — All Timepoints | Scenarios A + B")
    print("=" * 65)
    print(f"Scenarios:   {scenarios}")
    print(f"Timepoints:  {tp_list}")
    print()
    print("Scenario A: 5-digit animal_id as unit (n = animals)")
    print("Scenario B: 6-digit session_id as unit (n = 2 × animals)")
    print("            4m: files DUPLICATED into session 1 and session 2")

    for tp in tp_list:
        process_timepoint(tp, scenarios)

    print("\n" + "="*65)
    print("ALL DONE")
    print("="*65)
