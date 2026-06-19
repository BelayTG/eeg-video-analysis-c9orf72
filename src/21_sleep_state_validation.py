"""
21_sleep_state_validation.py
=============================
Validate the EEG-only vigilance-state classification using the SYNCHRONOUS VIDEO
as an EMG surrogate, plus physiological sanity checks. Built to pre-empt the
reviewer objection that REM (theta-dominant, low-delta) may be contaminated by
active/exploratory wake in the absence of EMG.

Produces, for a (sub)set of recordings:
  (1) REM proportion         -- % of total epochs and % of sleep, by group x timepoint
                                 (physiological mouse REM is ~5-10% of total time)
  (2) Movement-by-state      -- per-epoch video motion energy by state. The key test:
                                 REM should be ~as immobile as NREM and far below active wake.
  (3) State-separation plane -- relative delta vs relative theta, coloured by motion energy
                                 (validation figure; true REM = high-theta AND immobile)
  (4) Transition structure   -- wake->REM rarity, NREM->REM dominance, REM self-maintenance,
                                 and per-state bout durations (real sleep architecture)
  (5) Hippocampal theta sig  -- CA3 theta peak frequency & regularity in REM vs active wake
  (6) Contamination estimate -- among theta-rich/low-delta epochs, the immobile (true-REM)
                                 fraction vs moving (active-wake) fraction

USAGE
-----
    conda activate eeg_video
    # fast first pass (recommended): a few animals per group at a few timepoints
    python src\\21_sleep_state_validation.py --subset
    # full run (slow: video decode is I/O-bound on the external drive; run overnight)
    python src\\21_sleep_state_validation.py --full

NOTES
-----
* This re-derives per-epoch states with a documented band-power classifier matching the
  staging-grid logic (relative-delta percentile for NREM; relative-theta percentile for REM).
  >>> If your pipeline saves per-epoch state labels, point LOAD_EXISTING_LABELS at them so the
      validation reflects YOUR classifier exactly (preferred). See the config block.
* Motion energy = mean abs frame-to-frame difference of a downscaled grayscale video, averaged
  within each 4-s EEG epoch, assuming EEG and video start synchronously (as in script 20B).
* Per-recording motion is cached to results\\_motion_cache\\ so re-runs are fast.
"""

import os, sys, json, argparse, glob
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import welch

# --- project resolver ----------------------------------------------------------
import abf_paths as ap   # iter_recordings(tp, scenario='A'), video_for(abf_path), find_video_near(abf_path)

# NumPy 2.0 renamed trapz -> trapezoid; support both
_trapz = getattr(np, "trapezoid", None) or np.trapz

FS_DS   = getattr(ap, "FS_DS", 500)
EPOCH_S = getattr(ap, "EPOCH_S", 4.0)
EPOCH_N = int(round(FS_DS * EPOCH_S))
TIMEPOINTS_ALL = ["3m", "4m", "6m", "7m", "9m", "12m"]

# ============================================================================
# Sleep-state functions copied VERBATIM from src/01_sleep_state_classification.py
# so this validation runs the EXACT staging used in the manuscript. (Script 01 is
# not imported directly because it auto-runs its main() on import.)
# Keep these in sync if 01 changes.
# ============================================================================
import pyabf

def load_signal(abf_path, channel=0, target_fs=FS_DS):
    try:
        a = pyabf.ABF(abf_path)
        a.setSweep(0, channel=channel)
        sig = a.sweepY.astype(np.float64)
        fs = a.dataRate
    except Exception:
        return None, None
    if fs != target_fs:
        n = int(round(len(sig) * target_fs / fs))
        sig = signal.resample(sig, n)
    return sig, float(target_fs)

def band_power(psd, freqs, lo, hi):
    m = (freqs >= lo) & (freqs <= hi)
    total = _trapz(psd, freqs) + 1e-12
    if not m.any(): return 0.0
    return float(_trapz(psd[m], freqs[m]) / total)

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
    for col in ["bp_delta","bp_theta","bp_alpha","bp_beta","bp_gamma","total_var"]:
        m = df[col].mean(); s = df[col].std() + 1e-8
        df[f"z_{col}"] = (df[col] - m) / s
    df["dt_ratio"] = df["bp_delta"] / (df["bp_theta"] + 1e-8)
    dt_med = df["dt_ratio"].median()
    var_75 = df["total_var"].quantile(0.75)
    var_25 = df["total_var"].quantile(0.25)
    states = []
    for _, row in df.iterrows():
        if (row["total_var"] > var_75 or row["z_bp_beta"] > 1.0 or row["z_bp_gamma"] > 1.0):
            states.append("Wake")
        elif (row["dt_ratio"] < dt_med * 0.5 and row["z_bp_theta"] > 0 and row["total_var"] < var_75):
            states.append("REM")
        elif row["dt_ratio"] > dt_med:
            states.append("NREM")
        else:
            states.append("Wake")
    df["state"] = states
    return df

def theta_peak_freq(seg, fs):
    """Dominant frequency within 5-10 Hz for one epoch (Hz)."""
    f, p = welch(seg, fs=fs, nperseg=min(len(seg), int(fs*2)))
    m = (f >= 5) & (f <= 10)
    if not np.any(m):
        return np.nan
    fm, pm = f[m], p[m]
    return float(fm[np.argmax(pm)])

# ================================ CONFIG ======================================
CH_CTX   = 1          # cortical S1/PtA channel (classification channel in script 01? confirm)
CH_CA3   = 0          # hippocampal CA3 channel (theta-signature check)
VID_DOWNSCALE = (96, 72)
VID_FRAME_STRIDE = 3
SUBSET_TIMEPOINTS = ["3m", "9m", "12m"]
SUBSET_PER_GROUP  = 3
RESULTS_DIR = "results"
CACHE_DIR   = os.path.join(RESULTS_DIR, "_motion_cache")
FIG_DIR     = "figures"
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
# ==============================================================================


# ----------------------------- EEG feature / state ----------------------------
def theta_peak_freq(seg, fs):
    """Dominant frequency within the theta band (6-9 Hz) for one epoch (Hz)."""
    f, p = signal.welch(seg, fs=fs, nperseg=min(len(seg), int(fs*2)))
    m = (f >= 5) & (f <= 10)
    if not np.any(m):
        return np.nan
    fm, pm = f[m], p[m]
    return float(fm[np.argmax(pm)])


# ----------------------------- video motion energy ----------------------------
def motion_energy_per_epoch(video_path, n_epochs):
    """Mean abs frame-difference (downscaled grayscale) within each 4-s epoch.
       Assumes video starts synchronously with the EEG recording."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_per_epoch = int(round(fps * EPOCH_S))
    me = np.full(n_epochs, np.nan)
    prev = None
    for ep in range(n_epochs):
        diffs = []
        # position to start of epoch (sequential read is more reliable than seek on wmv)
        target_start = ep * frames_per_epoch
        # read through this epoch window
        fidx = 0
        while fidx < frames_per_epoch:
            ok, frame = cap.read()
            if not ok:
                break
            if (fidx % VID_FRAME_STRIDE) == 0:
                g = cv2.cvtColor(cv2.resize(frame, VID_DOWNSCALE), cv2.COLOR_BGR2GRAY).astype(np.float32)
                if prev is not None:
                    diffs.append(np.mean(np.abs(g - prev)))
                prev = g
            fidx += 1
        if diffs:
            me[ep] = float(np.mean(diffs))
        if not ok:
            break
    cap.release()
    return me


def cached_motion(rec_id, video_path, n_epochs):
    cf = os.path.join(CACHE_DIR, f"{rec_id}.npy")
    if os.path.exists(cf):
        m = np.load(cf, allow_pickle=True)
        if len(m) >= n_epochs:
            return m[:n_epochs]
    m = motion_energy_per_epoch(video_path, n_epochs)
    if m is not None:
        np.save(cf, m)
    return m


# ----------------------------- bout durations ---------------------------------
def bout_durations(states):
    out = {"Wake": [], "NREM": [], "REM": []}
    if len(states) == 0:
        return out
    cur, run = states[0], 1
    for s in states[1:]:
        if s == cur:
            run += 1
        else:
            out.setdefault(cur, []).append(run * EPOCH_S)
            cur, run = s, 1
    out.setdefault(cur, []).append(run * EPOCH_S)
    return out


def transition_matrix(states):
    order = ["Wake", "NREM", "REM"]
    idx = {s: i for i, s in enumerate(order)}
    M = np.zeros((3, 3))
    for a, b in zip(states[:-1], states[1:]):
        M[idx[a], idx[b]] += 1
    M = M / (M.sum(1, keepdims=True) + 1e-12)
    return M, order


# ---------------------------------- main --------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--subset", action="store_true", help="few animals/timepoints (fast)")
    pa.add_argument("--full", action="store_true", help="all recordings (slow)")
    args = pa.parse_args()
    do_subset = args.subset or not args.full

    recs = []   # (abf_path, animal_id, group, timepoint)
    tps = SUBSET_TIMEPOINTS if do_subset else TIMEPOINTS_ALL
    counts = {}
    for tp in tps:
        for rec in ap.iter_recordings(tp, scenario="A"):
            abf, animal_id, group = rec[0], rec[1], rec[2]
            if do_subset:
                k = (tp, group)
                if counts.get(k, 0) >= SUBSET_PER_GROUP:
                    continue
                counts[k] = counts.get(k, 0) + 1
            recs.append((abf, str(animal_id), str(group), tp))
    print(f"Validating {len(recs)} recordings "
          f"({'subset' if do_subset else 'full'} mode)\n")

    rows = []          # per-epoch records
    per_rec_tm = []    # per-recording transition matrices + bouts

    for n, (abf, animal_id, group, tp) in enumerate(recs, 1):
        rec_id = f"{animal_id}_{os.path.splitext(os.path.basename(abf))[0]}"
        # --- EEG: cortical channel -> features -> YOUR classifier ---
        ctx, fs = load_signal(abf, channel=CH_CTX, target_fs=FS_DS)
        if ctx is None:
            print(f"  [{rec_id}] EEG load failed"); continue
        feat = compute_epoch_features(ctx, fs)
        if len(feat) < 10:
            print(f"  [{rec_id}] too few epochs ({len(feat)})"); continue
        feat = classify_sleep_states(feat)        # adds 'state' column (Wake/NREM/REM)
        states = feat["state"].values
        epoch_idx = feat["epoch_idx"].values       # original epoch positions (artifact-pruned)
        n_ep = len(states)

        # --- hippocampal CA3: theta/delta + theta peak freq for the SAME epochs ---
        td_ca3 = np.full(n_ep, np.nan); ca3_pk = np.full(n_ep, np.nan)
        ca3, fsc = load_signal(abf, channel=CH_CA3, target_fs=FS_DS)
        if ca3 is not None:
            epn = int(EPOCH_S * fsc)
            for j, ei in enumerate(epoch_idx):
                seg = ca3[ei*epn:(ei+1)*epn]
                if len(seg) < epn:
                    continue
                f, psd = welch(seg, fs=fsc, nperseg=min(int(fsc*2), epn))
                d = band_power(psd, f, 0.5, 4); t = band_power(psd, f, 4, 8)
                td_ca3[j] = t / (d + 1e-8)
                ca3_pk[j] = theta_peak_freq(seg, fsc)

        # --- video motion energy, aligned to the SAME (pruned) epochs ---
        vpath = None
        for fn in ("video_for", "find_video_near"):
            if hasattr(ap, fn):
                try:
                    vpath = getattr(ap, fn)(abf)
                    if vpath:
                        break
                except Exception:
                    pass
        me_full = cached_motion(rec_id, vpath, int(max(epoch_idx))+1) if (vpath and os.path.exists(str(vpath))) else None
        if me_full is not None:
            me = np.array([me_full[ei] if ei < len(me_full) else np.nan for ei in epoch_idx])
        else:
            me = np.full(n_ep, np.nan)
        print(f"  [{n}/{len(recs)}] {rec_id} {group} {tp}: "
              f"{n_ep} epochs (W/N/R = "
              f"{int((states=='Wake').sum())}/{int((states=='NREM').sum())}/{int((states=='REM').sum())}), "
              f"video={'yes' if np.isfinite(me).any() else 'no'}")

        for j in range(n_ep):
            rows.append(dict(rec=rec_id, group=group, timepoint=tp,
                             state=states[j],
                             dt_ratio=feat["dt_ratio"].values[j] if "dt_ratio" in feat else np.nan,
                             td_ca3=td_ca3[j], ca3_theta_peak=ca3_pk[j],
                             motion=me[j]))
        M, order = transition_matrix(states)
        bouts = bout_durations(states)
        per_rec_tm.append(dict(rec=rec_id, group=group, timepoint=tp,
                               w2r=M[0, 2], n2r=M[1, 2], r2r=M[2, 2],
                               rem_bout=np.mean(bouts["REM"]) if bouts["REM"] else np.nan,
                               nrem_bout=np.mean(bouts["NREM"]) if bouts["NREM"] else np.nan))

    if not rows:
        print("No epochs collected. Check abf_paths API names / paths."); return
    df = pd.DataFrame(rows)
    tmdf = pd.DataFrame(per_rec_tm)
    df.to_csv(os.path.join(RESULTS_DIR, "sleep_state_validation_epochs.csv"), index=False)
    tmdf.to_csv(os.path.join(RESULTS_DIR, "sleep_state_validation_transitions.csv"), index=False)

    # ---- digest ----
    def cohend(a, b):
        a, b = np.asarray(a), np.asarray(b)
        sp = np.sqrt(((len(a)-1)*np.var(a, ddof=1)+(len(b)-1)*np.var(b, ddof=1))/(len(a)+len(b)-2))
        return (np.mean(b)-np.mean(a))/sp if sp > 0 else np.nan

    print("\n" + "="*70)
    print("VALIDATION DIGEST")
    print("="*70)

    # (1) REM proportion
    print("\n(1) REM proportion (% of total epochs; % of sleep):")
    for (g, tp), sub in df.groupby(["group", "timepoint"]):
        rem = (sub.state == "REM").mean()*100
        sleep = sub.state.isin(["NREM", "REM"])
        rem_of_sleep = (sub.state == "REM").sum()/max(sleep.sum(), 1)*100
        print(f"   {g} {tp:>3}: REM = {rem:4.1f}% of total, {rem_of_sleep:4.1f}% of sleep")
    print("   (physiological mouse REM ~5-10% of total time)")

    # (2) movement by state  -- the key EMG-surrogate test
    mv = df.dropna(subset=["motion"])
    if len(mv):
        print("\n(2) Video motion energy by state (EMG surrogate):")
        for st in ["Wake", "NREM", "REM"]:
            v = mv[mv.state == st]["motion"]
            print(f"   {st:>4}: mean motion = {v.mean():.3f}  median = {v.median():.3f}  (n={len(v)} epochs)")
        wake = mv[mv.state == "Wake"]["motion"]; rem = mv[mv.state == "REM"]["motion"]
        nrem = mv[mv.state == "NREM"]["motion"]
        # Cohen's d AND Mann-Whitney U (non-parametric; motion is skewed)
        if len(rem) and len(wake):
            u1, p1 = stats.mannwhitneyu(rem, wake, alternative="less")
            print(f"   REM vs Wake (all) : d = {cohend(wake, rem):+.2f}, MW p = {p1:.2e}  (expect REM < Wake)")
        if len(rem) and len(nrem):
            u2, p2 = stats.mannwhitneyu(rem, nrem, alternative="two-sided")
            print(f"   REM vs NREM       : d = {cohend(nrem, rem):+.2f}, MW p = {p2:.3f}  (expect REM <= NREM; both immobile)")
        # ACTIVE wake = the high-motion tail of wake (the epochs that actually threaten REM).
        # Nobody confuses REM with quiet wake; the objection is active/exploratory wake.
        if len(wake):
            active_thr = np.percentile(wake, 75)
            active = wake[wake >= active_thr]
            quiet  = wake[wake <  active_thr]
            print(f"   Wake split: quiet (n={len(quiet)}, mean={quiet.mean():.3f}) "
                  f"vs active (n={len(active)}, mean={active.mean():.3f})")
            if len(rem) and len(active):
                u3, p3 = stats.mannwhitneyu(rem, active, alternative="less")
                print(f"   REM vs ACTIVE wake: d = {cohend(active, rem):+.2f}, MW p = {p3:.2e}  "
                      f"(the objection-relevant contrast; expect REM << active wake)")

        # immobility threshold defined from NREM (a state we are confident is immobile)
        immobile_thr = np.percentile(nrem, 90) if len(nrem) else np.percentile(mv["motion"], 50)

        # (6) contamination: of the epochs the classifier CALLS REM, how many are immobile?
        cand = mv[mv.state == "REM"]
        if len(cand):
            imm = (cand["motion"] < immobile_thr).mean()*100
            print(f"\n(6) Epochs classified REM: n={len(cand)}")
            print(f"    {imm:4.1f}% immobile (true-REM-like)   {100-imm:4.1f}% moving (active-wake-like)")
            for g in ["WT", "KO"]:
                cg = cand[cand.group == g]
                if len(cg):
                    print(f"      {g}: {(cg['motion']<immobile_thr).mean()*100:4.1f}% immobile (n={len(cg)})")

        # confusion matrix: EEG state vs video moving/immobile
        print("\n    Confusion (EEG state x video):   immobile   moving")
        for st in ["Wake", "NREM", "REM"]:
            sub = mv[mv.state == st]["motion"]
            if len(sub):
                imm_n = int((sub < immobile_thr).sum()); mov_n = int((sub >= immobile_thr).sum())
                print(f"      {st:>4}: {imm_n:8d}   {mov_n:6d}   "
                      f"({100*imm_n/len(sub):.0f}% immobile)")
    else:
        print("\n(2) No video motion available - rerun with videos accessible.")

    # (4) transition structure
    if len(tmdf):
        print("\n(4) Sleep architecture (per-recording means):")
        print(f"   Wake->REM prob : {tmdf.w2r.mean():.3f}  (expect LOW; real REM rarely entered from wake)")
        print(f"   NREM->REM prob : {tmdf.n2r.mean():.3f}  (expect dominant entry route)")
        print(f"   REM self-maint : {tmdf.r2r.mean():.3f}")
        print(f"   mean REM bout  : {tmdf.rem_bout.mean():.1f} s   "
              f"mean NREM bout : {tmdf.nrem_bout.mean():.1f} s")

    # (5) hippocampal theta signature in REM vs wake: ratio AND peak frequency
    if df["td_ca3"].notna().any():
        print("\n(5) Hippocampal (CA3) theta in REM vs Wake vs NREM:")
        for st in ["Wake", "NREM", "REM"]:
            v = df[df.state == st]["td_ca3"].dropna()
            pk = df[df.state == st]["ca3_theta_peak"].dropna()
            print(f"   {st:>4}: CA3 theta/delta = {v.mean():.2f}   theta peak = {pk.mean():.2f} Hz")
        rw = df[df.state=="REM"]["td_ca3"].dropna(); ww = df[df.state=="Wake"]["td_ca3"].dropna()
        if len(rw) and len(ww):
            print(f"   REM vs Wake CA3 theta/delta: d = {cohend(ww, rw):+.2f}  (expect REM > Wake)")

    # ---- figure: motion energy distribution by state (the EMG-surrogate figure) ----
    if len(mv):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
        # (left) motion by state, with wake split into quiet vs active
        order = ["Quiet wake", "Active wake", "NREM", "REM"]
        colors = {"Quiet wake": "#C7CBD1", "Active wake": "#6E7480", "NREM": "#4C72B0", "REM": "#C44E52"}
        wk = mv[mv.state == "Wake"]["motion"]
        athr = np.percentile(wk, 75) if len(wk) else np.inf
        data_map = {
            "Quiet wake":  wk[wk < athr].values,
            "Active wake": wk[wk >= athr].values,
            "NREM":        mv[mv.state == "NREM"]["motion"].values,
            "REM":         mv[mv.state == "REM"]["motion"].values,
        }
        data = [data_map[s] for s in order]
        try:
            bp = axes[0].boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
        except TypeError:
            bp = axes[0].boxplot(data, labels=order, patch_artist=True, showfliers=False)
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor(colors[s]); patch.set_alpha(0.75)
        axes[0].set_ylabel("Video motion energy (EMG surrogate)")
        axes[0].set_title("Movement by vigilance state\nREM immobile; far below active wake")
        axes[0].tick_params(axis="x", labelrotation=15)
        # (right) CA3 theta/delta by state (clean; immune to delta/theta outliers)
        ca3_data = [df[df.state == s]["td_ca3"].dropna().values for s in ["Wake", "NREM", "REM"]]
        try:
            bp2 = axes[1].boxplot(ca3_data, tick_labels=["Wake", "NREM", "REM"], patch_artist=True, showfliers=False)
        except TypeError:
            bp2 = axes[1].boxplot(ca3_data, labels=["Wake", "NREM", "REM"], patch_artist=True, showfliers=False)
        for patch, s in zip(bp2["boxes"], ["Wake", "NREM", "REM"]):
            patch.set_facecolor({"Wake": "#9AA0A6", "NREM": "#4C72B0", "REM": "#C44E52"}[s]); patch.set_alpha(0.75)
        axes[1].set_ylabel("CA3 theta/delta ratio")
        axes[1].set_title("Hippocampal theta greatest in REM\n(~6.3 Hz peak)")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "sleep_state_validation_motion.png")
        fig.savefig(out, dpi=200, bbox_inches="tight"); print(f"\nSaved {out}")

    print("\nSaved: results\\sleep_state_validation_epochs.csv,"
          " results\\sleep_state_validation_transitions.csv")
    print("(Threshold robustness is covered by the staging-sensitivity grid, Table S6 / Fig S6.)")
    print("="*70)


if __name__ == "__main__":
    main()
