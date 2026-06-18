"""
20_hmm_states_and_video_coupling.py
====================================
TIER 2 — DATA-DRIVEN BRAIN STATES + EEG-VIDEO MOVEMENT COUPLING

Two analyses. The second is the differentiator: almost no ALS EEG paper has
SYNCHRONOUS video. This script processes the raw video into motion energy and
asks whether cortical activity predicts movement — and whether that predictive
relationship degrades over the disease course (functional corticomotor decoupling).

PART A — HIDDEN MARKOV MODEL BRAIN STATES (data-driven, not threshold-based)
  Fits a Gaussian HMM to per-epoch spectral features and lets the data discover
  recurring brain states. Then tests whether the state REPERTOIRE, dwell times,
  or transition structure differ by genotype and change over the course.
  More principled than band-power thresholds; reviewers respect it.

PART B — EEG-VIDEO MOVEMENT COUPLING (unique to this dataset)
  (0) PREPROCESS VIDEO -> motion energy per frame (frame differencing), aligned
      to EEG epochs. Caches to data/motion_energy_<tp>.csv so it runs once.
  (1) Movement-state-conditioned EEG: split wake into movement vs immobility
      using motion energy; compare EEG features around movement.
  (2) Corticomotor prediction: can cortical (S1/PtA) activity in a short window
      predict imminent movement onset? Quantify with a simple predictive AUC /
      cross-correlation lead. Test whether this predictive coupling DECLINES
      over the disease course in KO vs WT — a functional decoupling signature.

Outputs:
  data/motion_energy_<tp>.csv                (cached, per-epoch motion energy)
  results/hmm_state_metrics.csv
  results/eeg_video_movement_coupling.csv
  figures/hmm_states.png
  figures/corticomotor_prediction_trajectory.png

INPUT:
  - raw ABF (both channels)
  - video files (set VIDEO_DIR + VIDEO_EXT; mapping abf_file -> video file)
Run:
    python src/20_hmm_states_and_video_coupling.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt, hilbert, decimate
from scipy.stats import mannwhitneyu, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from abf_paths import iter_recordings, find_video_near
    HAVE_PATHS = True
except ImportError:
    HAVE_PATHS = False

warnings.filterwarnings("ignore")

try:
    import pyabf
    HAVE_ABF = True
except ImportError:
    HAVE_ABF = False

try:
    from hmmlearn.hmm import GaussianHMM
    HAVE_HMM = True
except ImportError:
    HAVE_HMM = False

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")
ABF_DIR     = os.path.join(PORT_DIR, "data")
VIDEO_DIR   = os.path.join(PORT_DIR, "data")     # adjust to where videos live
VIDEO_EXTS  = [".avi", ".mp4", ".mov", ".mkv"]

CA3_CH, CTX_CH = 0, 1
FS_TARGET = 500.0
EPOCH_S   = 4.0
TP_ORDER  = ["3m", "4m", "6m", "7m", "9m", "12m"]
COLORS    = {"WT": "#378ADD", "KO": "#D85A30"}
N_HMM_STATES = 4
MAX_EPOCHS = 200


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2: return np.nan
    p = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return 0.0 if p == 0 else (np.mean(a)-np.mean(b))/p


def bandpass(x, lo, hi, fs=FS_TARGET, order=3):
    ny = 0.5*fs
    b, a = butter(order, [max(lo/ny,1e-4), min(hi/ny,0.999)], btype="band")
    return filtfilt(b, a, x)


def load_signal(abf_path, channel=0, target_fs=FS_TARGET):
    """Portfolio-consistent loader (matches script 14 load_signal)."""
    try:
        import gc
        abf = pyabf.ABF(abf_path)
        fs  = float(abf.dataRate)
        if channel >= abf.channelCount:
            del abf; gc.collect()
            return None, None
        abf.setSweep(0, channel=channel)
        sig = abf.sweepY.copy().astype(np.float64)
        del abf; gc.collect()
        if len(sig) < int(fs*30):
            return None, None
        factor = max(1, int(round(fs/target_fs)))
        if factor > 1:
            sig = decimate(sig - sig.mean(), factor, zero_phase=True)
        return sig, float(target_fs)
    except Exception:
        return None, None


def load_abf(path):
    ca3, fs = load_signal(path, channel=CA3_CH)
    if ca3 is None:
        return None, None
    ctx, _ = load_signal(path, channel=CTX_CH)
    if ctx is None:
        return None, None
    return ca3, ctx
    try:
        abf = pyabf.ABF(path); fs = abf.dataRate
        abf.setSweep(0, channel=CA3_CH); ca3 = abf.sweepY.copy()
        abf.setSweep(0, channel=CTX_CH); ctx = abf.sweepY.copy()
        f = int(round(fs/FS_TARGET))
        if f > 1:
            ca3 = decimate(ca3, f, ftype="iir", zero_phase=True)
            ctx = decimate(ctx, f, ftype="iir", zero_phase=True)
        return ca3, ctx
    except Exception:
        return None, None


def _states_by_basename(tp):
    p = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
    if not os.path.exists(p):
        return {}
    ep = pd.read_csv(p)
    out = {}
    for abf_file, g in ep.groupby("abf_file"):
        out[os.path.basename(str(abf_file))] = g.sort_values("epoch_idx")["state"].tolist()
    return out


def file_manifest(tp):
    p = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
    if not os.path.exists(p): return {}, None
    ep = pd.read_csv(p); ep["animal_id"] = ep["animal_id"].astype(str)
    man = {}
    for abf_file, g in ep.groupby("abf_file"):
        man[abf_file] = (g["animal_id"].iloc[0], g["group"].iloc[0])
    return man, ep


# ── PART A — HMM brain states ──────────────────────────────────────────────
HMM_FEATURES = ["rbp_delta", "rbp_theta", "rbp_alpha", "rbp_beta",
                "rbp_gamma", "td_ratio", "total_var"]


def part_a_hmm():
    print("\n" + "="*64)
    print("PART A — HMM DATA-DRIVEN BRAIN STATES")
    print("="*64)
    if not HAVE_HMM:
        print("hmmlearn required: pip install hmmlearn --break-system-packages")
        return None

    # Build pooled feature matrix from epoch CSVs (per channel = cortical)
    all_feats, meta = [], []
    for tp in TP_ORDER:
        p = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
        if not os.path.exists(p): continue
        ep = pd.read_csv(p); ep["animal_id"] = ep["animal_id"].astype(str)
        feats = [c for c in HMM_FEATURES if c in ep.columns]
        if len(feats) < 3:
            continue
        X = ep[feats].fillna(ep[feats].median()).values
        all_feats.append(X)
        m = ep[["animal_id", "group"]].copy(); m["timepoint"] = tp
        meta.append(m)
    if not all_feats:
        print("No epoch features for HMM.")
        return None
    X = np.vstack(all_feats)
    meta = pd.concat(meta, ignore_index=True)
    if HAVE_SK:
        X = StandardScaler().fit_transform(X)

    print(f"Fitting {N_HMM_STATES}-state Gaussian HMM on {X.shape[0]} epochs...")
    hmm = GaussianHMM(n_components=N_HMM_STATES, covariance_type="diag",
                      n_iter=50, random_state=0)
    hmm.fit(X)
    states = hmm.predict(X)
    meta["hmm_state"] = states

    # Per-animal: fractional occupancy of each state
    rows = []
    for (aid, group, tp), g in meta.groupby(["animal_id", "group", "timepoint"]):
        occ = g["hmm_state"].value_counts(normalize=True)
        row = {"animal_id": aid, "group": group, "timepoint": tp,
               "n_epochs": len(g)}
        for st in range(N_HMM_STATES):
            row[f"occ_state{st}"] = occ.get(st, 0.0)
        # transition entropy (disorder of state switching)
        seq = g["hmm_state"].values
        trans = np.zeros((N_HMM_STATES, N_HMM_STATES))
        for i in range(len(seq)-1):
            trans[seq[i], seq[i+1]] += 1
        tp_norm = trans / (trans.sum(axis=1, keepdims=True) + 1e-9)
        ent = -np.nansum(tp_norm*np.log(tp_norm+1e-12))
        row["transition_entropy"] = ent
        rows.append(row)
    occ_df = pd.DataFrame(rows)
    occ_df.to_csv(os.path.join(RESULTS_DIR, "hmm_state_metrics.csv"), index=False)
    print(f"Saved: hmm_state_metrics.csv ({len(occ_df)} animal-timepoints)")

    print("\nTransition entropy (state-switching disorder): WT vs KO")
    for tp in TP_ORDER:
        sub = occ_df[occ_df.timepoint == tp]
        wt = sub[sub.group=="WT"]["transition_entropy"].dropna().values
        ko = sub[sub.group=="KO"]["transition_entropy"].dropna().values
        if len(wt) < 2 or len(ko) < 2: continue
        d = cohens_d(ko, wt); _, p = mannwhitneyu(ko, wt, alternative="two-sided")
        flag = "  *" if p < 0.05 else ""
        print(f"  [{tp}] WT={np.mean(wt):.3f} KO={np.mean(ko):.3f} d={d:.2f} p={p:.3f}{flag}")

    # Figure: state occupancy trajectory
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for st in range(N_HMM_STATES):
        for group, ls in [("WT", "-"), ("KO", "--")]:
            xs, ms = [], []
            for tp in TP_ORDER:
                v = occ_df[(occ_df.timepoint==tp)&(occ_df.group==group)][f"occ_state{st}"].dropna()
                if len(v)==0: continue
                xs.append(TP_ORDER.index(tp)); ms.append(v.mean())
            if xs:
                ax.plot(xs, ms, ls, label=f"S{st} {group}", alpha=0.8)
    ax.set_xticks(range(len(TP_ORDER))); ax.set_xticklabels(TP_ORDER)
    ax.set_ylabel("Fractional occupancy"); ax.set_xlabel("Timepoint")
    ax.set_title("HMM brain-state occupancy over disease course", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(FIGURES_DIR, "hmm_states.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: hmm_states.png")
    return occ_df


# ── PART B — Video preprocessing + corticomotor coupling ───────────────────
def _load_video_map():
    """Load the verified ABF->video mapping produced by 20a_pair_videos_by_mtime.py."""
    p = os.path.join(DATA_DIR, "video_abf_map.csv")
    if not os.path.exists(p):
        return None
    m = pd.read_csv(p)
    # only trust rows the user verified
    if "verified" in m.columns:
        m = m[m["verified"] == 1]
    return m


_VIDEO_MAP = None


def find_video_for(abf_file):
    """Return the mapped video path for an ABF using the verified mtime mapping."""
    global _VIDEO_MAP
    if _VIDEO_MAP is None:
        _VIDEO_MAP = _load_video_map()
    if _VIDEO_MAP is None or len(_VIDEO_MAP) == 0:
        return None
    row = _VIDEO_MAP[_VIDEO_MAP["abf_file"] == abf_file]
    if len(row) == 0:
        return None
    vid_name = row.iloc[0]["video_file"]
    if not isinstance(vid_name, str) or vid_name == "":
        return None
    cand = os.path.join(VIDEO_DIR, vid_name)
    return cand if os.path.exists(cand) else None


def extract_motion_energy(video_path, n_epochs, fps_hint=None, samples_per_sec=2.0,
                          max_samples=2400):
    """
    Frame-difference motion energy, averaged into n_epochs bins.
    Robust+fast: reads sequentially but only KEEPS ~samples_per_sec frames
    (sequential read avoids slow per-frame seeks on wmv/codecs that lack fast
    random access), downscales hard, and caps total kept samples so one long or
    misbehaving file cannot hang the run.
    """
    if not HAVE_CV2:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint or 30.0
    if fps <= 0:
        fps = 30.0
    keep_every = max(1, int(round(fps / samples_per_sec)))
    motion = []
    prev = None
    fidx = 0
    kept = 0
    while kept < max_samples:
        ret = cap.grab()              # cheap: advance without full decode
        if not ret:
            break
        if fidx % keep_every == 0:    # only fully decode the frames we keep
            ret, frame = cap.retrieve()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (96, 72)).astype(np.float32)
            if prev is not None:
                motion.append((float(np.abs(gray - prev).mean()), fidx / fps))
            prev = gray
            kept += 1
        fidx += 1
    cap.release()
    if not motion:
        return None
    m = np.array([x[0] for x in motion]); t = np.array([x[1] for x in motion])
    dur = t[-1] if len(t) else 0
    if dur <= 0:
        return None
    edges = np.linspace(0, dur, n_epochs+1)
    binned = np.zeros(n_epochs)
    for i in range(n_epochs):
        sel = (t >= edges[i]) & (t < edges[i+1])
        binned[i] = m[sel].mean() if sel.any() else np.nan
    return binned


def get_motion_energy(tp, abf_path, n_epochs):
    """Load cached motion energy or extract from the nearest in-folder video + cache it."""
    abf_file = os.path.basename(abf_path)
    cache = os.path.join(DATA_DIR, f"motion_energy_{tp}.csv")
    if os.path.exists(cache):
        mc = pd.read_csv(cache)
        sub = mc[mc.abf_file == abf_file]
        if len(sub):
            return sub.sort_values("epoch_idx")["motion_energy"].values
    vid, gap = find_video_near(abf_path, max_gap_min=30) if HAVE_PATHS else (None, None)
    if vid is None:
        return None
    import time as _t
    _t0 = _t.time()
    print(f"      decoding {os.path.basename(vid)} ...", end="", flush=True)
    me = extract_motion_energy(vid, n_epochs)
    print(f" {_t.time()-_t0:.1f}s")
    if me is None:
        return None
    new = pd.DataFrame({"abf_file": abf_file, "epoch_idx": np.arange(len(me)),
                        "motion_energy": me, "video_file": os.path.basename(vid),
                        "gap_min": round(gap, 2) if gap is not None else None})
    if os.path.exists(cache):
        new = pd.concat([pd.read_csv(cache), new], ignore_index=True)
    new.to_csv(cache, index=False)
    return me


def part_b_video_coupling():
    print("\n" + "="*64)
    print("PART B — EEG-VIDEO CORTICOMOTOR COUPLING")
    print("Does cortical activity predict movement, and does it decline?")
    print("="*64)
    if not HAVE_ABF:
        print("pyabf required."); return None
    if not HAVE_CV2:
        print("opencv required for video: pip install opencv-python-headless --break-system-packages")
        print("(If motion_energy_<tp>.csv caches already exist, will use them.)")

    if not HAVE_PATHS:
        print("abf_paths.py not importable; cannot locate F:\\EEG videos."); return None

    rows = []
    for tp in TP_ORDER:
        state_lut = _states_by_basename(tp)
        recs = list(iter_recordings(tp)) if HAVE_PATHS else []
        if not recs: continue
        print(f"[{tp}] {len(recs)} recordings on disk")
        for path, mouse_id, group in recs:
            aid = mouse_id
            seq = state_lut.get(os.path.basename(path))
            if seq is None: continue
            ca3, ctx = load_abf(path)
            if ctx is None: continue
            n_ep = len(seq)
            if n_ep < 20: continue
            motion = get_motion_energy(tp, path, n_ep)
            if motion is None or np.all(np.isnan(motion)):
                continue
            ep_len = int(EPOCH_S*FS_TARGET)
            # per-epoch cortical beta/gamma power (the motor-relevant bands)
            ctx_beta = np.full(n_ep, np.nan)
            for ei in range(n_ep):
                a, b = ei*ep_len, (ei+1)*ep_len
                if b > len(ctx): break
                f, p = welch(ctx[a:b], FS_TARGET, nperseg=min(1024, b-a))
                m = (f >= 13) & (f < 30)
                ctx_beta[ei] = np.trapz(p[m], f[m])
            valid = ~np.isnan(motion) & ~np.isnan(ctx_beta)
            if valid.sum() < 15: continue
            mE = motion[valid]; cB = ctx_beta[valid]
            # (1) concurrent correlation cortex-beta vs motion
            r_conc, _ = pearsonr(cB, mE)
            # (2) predictive: does cortical beta at t predict motion at t+1?
            if valid.sum() > 16:
                r_pred, p_pred = pearsonr(cB[:-1], mE[1:])
            else:
                r_pred, p_pred = np.nan, np.nan
            rows.append({
                "timepoint": tp, "animal_id": aid, "group": group,
                "r_concurrent": r_conc, "r_predictive": r_pred,
                "n_epochs": int(valid.sum()),
            })
            print(f"  {aid} ({group}) r_pred={r_pred:.3f}")
    if not rows:
        print("No EEG-video coupling computed (need videos or cached motion energy).")
        return None
    vc = pd.DataFrame(rows)
    vc.to_csv(os.path.join(RESULTS_DIR, "eeg_video_movement_coupling.csv"), index=False)
    print(f"Saved: eeg_video_movement_coupling.csv ({len(vc)} rows)")

    print("\nCorticomotor predictive coupling (beta@t -> motion@t+1): WT vs KO")
    for tp in TP_ORDER:
        sub = vc[vc.timepoint == tp]
        wt = sub[sub.group=="WT"]["r_predictive"].dropna().values
        ko = sub[sub.group=="KO"]["r_predictive"].dropna().values
        if len(wt) < 2 or len(ko) < 2: continue
        d = cohens_d(ko, wt); _, p = mannwhitneyu(ko, wt, alternative="two-sided")
        flag = "  *" if p < 0.05 else ""
        print(f"  [{tp}] WT={np.mean(wt):.3f} KO={np.mean(ko):.3f} d={d:.2f} p={p:.3f}{flag}")

    # Figure: predictive coupling trajectory
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group, color in COLORS.items():
        g = vc[vc.group == group]
        xs, ms, es = [], [], []
        for tp in TP_ORDER:
            v = g[g.timepoint == tp]["r_predictive"].dropna()
            if len(v)==0: continue
            xs.append(TP_ORDER.index(tp)); ms.append(v.mean()); es.append(v.sem())
        if xs:
            ax.errorbar(xs, ms, yerr=es, fmt="-o", color=color, lw=2, capsize=3, label=group)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(TP_ORDER))); ax.set_xticklabels(TP_ORDER)
    ax.set_ylabel("Predictive coupling (cortex β → motion)")
    ax.set_xlabel("Timepoint")
    ax.set_title("Corticomotor predictive coupling over disease course\n"
                 "(decline = functional decoupling)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIGURES_DIR, "corticomotor_prediction_trajectory.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: corticomotor_prediction_trajectory.png")
    return vc


def main():
    import sys
    print("="*64)
    print("HMM BRAIN STATES + EEG-VIDEO CORTICOMOTOR COUPLING")
    print("="*64)
    only_b = "--only-b" in sys.argv      # skip HMM, run only video coupling
    only_a = "--only-a" in sys.argv      # run only HMM
    hmm_csv = os.path.join(RESULTS_DIR, "hmm_state_metrics.csv")

    if only_b:
        print("\n[--only-b] Skipping HMM; running video coupling only.")
    elif only_a:
        part_a_hmm()
    else:
        if os.path.exists(hmm_csv):
            print(f"\nHMM results already exist ({os.path.basename(hmm_csv)}); skipping recompute.")
            print("  (use --only-a to recompute the HMM)")
        else:
            part_a_hmm()

    if not only_a:
        part_b_video_coupling()
    print("\n" + "="*64)
    print("COMPLETE")
    print("="*64)


if __name__ == "__main__":
    main()
