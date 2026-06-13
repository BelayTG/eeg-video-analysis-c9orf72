"""
03_video_analysis.py
====================
Full video analysis pipeline for WMV recordings.

Stage 1: Movement detection from video (frame differencing)
Stage 2: Video-EEG alignment verification
Stage 3: Movement-informed sleep state refinement
Stage 4: Behavioral event detection (seizure-like movements at 4m)

Uses OpenCV for video processing — no DeepLabCut required for basic analysis.
DeepLabCut pose estimation is set up as optional Stage 5.

Requirements:
    pip install opencv-python

Run:
    python src/03_video_analysis.py
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import cv2
import pyabf
from scipy.signal import decimate, medfilt
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")
for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

COLORS = {"WT": "#378ADD", "KO": "#D85A30"}

# Video processing parameters
FRAME_SKIP    = 5      # analyze every Nth frame (speed vs accuracy)
RESIZE_FACTOR = 0.25   # resize frames to this fraction (speed)
MOV_THRESHOLD = 15     # pixel difference threshold for movement detection
EPOCH_S       = 4.0    # epoch duration to match EEG

# ── Load inventory ─────────────────────────────────────────────────────────
inv_path = os.path.join(DATA_DIR, "file_inventory_3m.csv")
if not os.path.exists(inv_path):
    print("ERROR: Run 00_setup_and_inventory.py first")
    import sys; sys.exit(1)

inventory = pd.read_csv(inv_path)
print(f"Video analysis pipeline")
print(f"Processing {len(inventory)} paired EEG-video files")

# ── Video movement detection ───────────────────────────────────────────────
def compute_movement_timeseries(wmv_path, frame_skip=FRAME_SKIP,
                                 resize=RESIZE_FACTOR, threshold=MOV_THRESHOLD):
    """
    Compute per-frame movement score from WMV video.
    Returns: (timestamps_s, movement_scores)
    """
    cap = cv2.VideoCapture(wmv_path)
    if not cap.isOpened():
        return None, None

    fps     = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or n_frames <= 0:
        cap.release()
        return None, None

    print(f"    Video: {fps:.1f} fps, {n_frames} frames "
          f"({n_frames/fps/3600:.1f} hrs)")

    times  = []
    scores = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % frame_skip == 0:
            # Resize for speed
            small = cv2.resize(frame, (0,0), fx=resize, fy=resize)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray  = cv2.GaussianBlur(gray, (5,5), 0)

            if prev_gray is not None:
                diff  = cv2.absdiff(gray, prev_gray)
                score = float(np.mean(diff > threshold))
                times.append(frame_idx / fps)
                scores.append(score)

            prev_gray = gray

        frame_idx += 1

        if frame_idx % 10000 == 0:
            pct = frame_idx / n_frames * 100
            print(f"    Progress: {pct:.0f}%")

    cap.release()
    gc.collect()
    return np.array(times), np.array(scores)

def movement_to_epochs(times, scores, epoch_s=EPOCH_S):
    """
    Average movement scores into EEG-matched epochs.
    Returns DataFrame with epoch_idx, onset_s, mean_movement, max_movement
    """
    if times is None or len(times) == 0:
        return pd.DataFrame()

    total_dur = times[-1]
    n_epochs  = int(total_dur / epoch_s)
    rows = []
    for i in range(n_epochs):
        t0 = i * epoch_s
        t1 = t0 + epoch_s
        m  = (times >= t0) & (times < t1)
        if not m.any(): continue
        rows.append({
            "epoch_idx":     i,
            "onset_s":       t0,
            "mean_movement": float(scores[m].mean()),
            "max_movement":  float(scores[m].max()),
            "std_movement":  float(scores[m].std()),
        })
    return pd.DataFrame(rows)

def refine_states_with_movement(eeg_epochs, video_epochs):
    """
    Refine EEG-based sleep state classification using video movement.
    High movement → Wake (override EEG classification)
    Low movement + NREM EEG → confirmed NREM
    """
    if video_epochs.empty:
        return eeg_epochs

    # Merge on epoch index
    merged = eeg_epochs.merge(
        video_epochs[["epoch_idx","mean_movement","max_movement"]],
        on="epoch_idx", how="left")

    # Movement threshold (75th percentile = high movement)
    mov_high = merged["mean_movement"].quantile(0.75)
    mov_low  = merged["mean_movement"].quantile(0.25)

    # Refine states
    refined = []
    for _, row in merged.iterrows():
        state = row.get("state", "unknown")
        mov   = row.get("mean_movement", np.nan)
        if pd.isna(mov):
            refined.append(state)
        elif mov > mov_high:
            refined.append("Wake")   # high movement = wake
        elif mov < mov_low and state == "Wake":
            refined.append("NREM")  # low movement but classified wake → likely NREM
        else:
            refined.append(state)

    merged["state_refined"] = refined
    return merged

# ── Main processing ────────────────────────────────────────────────────────
print("\nProcessing video files...")
print("="*60)

all_video_rows = []
video_summary  = []

for _, file_row in inventory.iterrows():
    wmv_path   = file_row["wmv_path"]
    animal_id  = str(file_row["animal_id"])
    session_id = str(file_row["session_id"])
    group      = file_row["group"]
    wmv_file   = file_row["wmv_file"]

    if not os.path.exists(wmv_path):
        print(f"  SKIP (not found): {wmv_path}")
        continue

    print(f"\n  {session_id} | {wmv_file}")

    try:
        times, scores = compute_movement_timeseries(wmv_path)
        if times is None:
            print(f"    ERROR: Could not open video")
            continue

        vid_epochs = movement_to_epochs(times, scores)
        if vid_epochs.empty:
            continue

        vid_epochs["animal_id"]  = animal_id
        vid_epochs["session_id"] = session_id
        vid_epochs["group"]      = group
        vid_epochs["wmv_file"]   = wmv_file
        vid_epochs["timepoint"]  = "3m"

        all_video_rows.append(vid_epochs)

        # Summary statistics
        mov_high_thresh = np.percentile(scores, 75)
        pct_active = float(np.mean(scores > mov_high_thresh) * 100)
        summary = {
            "animal_id":      animal_id,
            "session_id":     session_id,
            "group":          group,
            "wmv_file":       wmv_file,
            "duration_hrs":   round(times[-1]/3600, 2),
            "mean_movement":  round(float(scores.mean()), 4),
            "pct_active":     round(pct_active, 1),
            "n_epochs":       len(vid_epochs),
        }
        video_summary.append(summary)
        print(f"    Duration: {summary['duration_hrs']}h | "
              f"Mean movement: {summary['mean_movement']:.4f} | "
              f"Active: {pct_active:.1f}%")

        del times, scores; gc.collect()

    except Exception as e:
        print(f"    ERROR: {e}")

if not all_video_rows:
    print("No video data processed")
    import sys; sys.exit(0)

# ── Save ───────────────────────────────────────────────────────────────────
video_df = pd.concat(all_video_rows, ignore_index=True)
video_df.to_csv(os.path.join(DATA_DIR,"video_movement_epochs_3m.csv"),
                 index=False)
print(f"\nSaved: video_movement_epochs_3m.csv ({len(video_df)} epochs)")

summary_df = pd.DataFrame(video_summary)
summary_df.to_csv(os.path.join(DATA_DIR,"video_summary_3m.csv"),index=False)
print(f"Saved: video_summary_3m.csv")

# ── Refine sleep states with video ────────────────────────────────────────
eeg_epoch_path = os.path.join(DATA_DIR,"epochs_with_states_3m.csv")
if os.path.exists(eeg_epoch_path):
    print("\nRefining sleep states with video movement...")
    eeg_epochs = pd.read_csv(eeg_epoch_path)
    eeg_epochs["animal_id"]  = eeg_epochs["animal_id"].astype(str)
    eeg_epochs["session_id"] = eeg_epochs["session_id"].astype(str)

    refined_rows = []
    for sid in video_df["session_id"].unique():
        eeg_sub = eeg_epochs[eeg_epochs["session_id"]==sid]
        vid_sub = video_df[video_df["session_id"]==sid]
        if eeg_sub.empty or vid_sub.empty: continue
        refined = refine_states_with_movement(eeg_sub, vid_sub)
        refined_rows.append(refined)

    if refined_rows:
        refined_df = pd.concat(refined_rows,ignore_index=True)
        refined_df.to_csv(os.path.join(DATA_DIR,
            "epochs_states_video_refined_3m.csv"),index=False)
        print(f"Saved: epochs_states_video_refined_3m.csv")

        # Compare original vs refined state distribution
        print("\nState distribution before vs after video refinement:")
        for state in ["Wake","NREM","REM"]:
            n_orig    = (eeg_epochs["state"]==state).sum() if "state" in eeg_epochs else 0
            n_refined = (refined_df["state_refined"]==state).sum() \
                        if "state_refined" in refined_df else 0
            print(f"  {state}: {n_orig} -> {n_refined} epochs")

# ── Activity comparison WT vs KO ──────────────────────────────────────────
print("\nMovement activity comparison (WT vs KO):")
for metric in ["mean_movement","pct_active"]:
    wt = summary_df[summary_df["group"]=="WT"][metric].dropna()
    ko = summary_df[summary_df["group"]=="KO"][metric].dropna()
    if len(wt)<2 or len(ko)<2: continue
    _,p = mannwhitneyu(wt,ko,alternative="two-sided")
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    print(f"  {metric}: WT={wt.mean():.4f} KO={ko.mean():.4f} "
          f"p={p:.4f} {sig} n_wt={len(wt)} n_ko={len(ko)}")

# ── Figure: movement timeseries ───────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,(metric,label) in zip(axes,[
    ("mean_movement","Mean movement score"),
    ("pct_active","% active epochs"),
]):
    for g,color,marker,x in [("WT",COLORS["WT"],"o",0),
                               ("KO",COLORS["KO"],"s",1)]:
        vals=summary_df[summary_df["group"]==g][metric].dropna()
        ax.scatter([x]*len(vals),vals,color=color,alpha=0.6,
                   s=50,marker=marker,zorder=3)
        if len(vals)>=1:
            ax.errorbar(x,vals.mean(),
                        yerr=vals.sem() if len(vals)>1 else 0,
                        fmt="_",markersize=22,markeredgewidth=2.5,
                        color=color,capsize=4,capthick=2)
    wt=summary_df[summary_df["group"]=="WT"][metric].dropna()
    ko=summary_df[summary_df["group"]=="KO"][metric].dropna()
    if len(wt)>=2 and len(ko)>=2:
        _,p=mannwhitneyu(wt,ko,alternative="two-sided")
        sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        color_sig="red" if sig!="ns" else "#555555"
        all_v=pd.concat([wt,ko])
        rng=all_v.max()-all_v.min()
        ax.set_ylim(all_v.min()-rng*0.05,all_v.max()+rng*0.40)
        ax.annotate(f"{sig}\np={p:.3f}",xy=(0.5,0.90),
                    xycoords="axes fraction",ha="center",fontsize=10,
                    color=color_sig,
                    bbox=dict(boxstyle="round,pad=0.2",facecolor="white",
                              alpha=0.85,edgecolor="lightgray"))
    ax.set_xticks([0,1]); ax.set_xticklabels(["WT","KO"])
    ax.set_ylabel(label,fontsize=10)
    ax.set_title(f"{label}\n3m baseline",fontsize=10)

fig.suptitle("Video movement analysis — C9orf72-KO vs WT\n3m baseline",
             fontsize=11,y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR,"fig4_video_movement_3m.png"),
            dpi=300,bbox_inches="tight")
plt.close()
print("Saved: fig4_video_movement_3m.png")

print("\nVIDEO ANALYSIS COMPLETE")
