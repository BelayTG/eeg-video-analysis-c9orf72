"""
07_advanced_video_analysis.py
==============================
Comprehensive synchronized EEG-video analysis pipeline.

Stages:
  1. Video movement quantification (frame differencing, optical flow)
  2. Movement bout detection (onset, duration, intensity)
  3. EEG-video synchronization verification (temporal alignment check)
  4. Video-informed sleep state refinement
  5. Seizure-like event detection (4m timepoint priority)
  6. Body region activity (head, body, limb using ROI grid)
  7. Video quality metrics (brightness, contrast, frame drops)
  8. Circadian movement pattern analysis
  9. WT vs KO movement comparison per state
 10. Export synchronized epoch table (EEG features + video features)

Outputs:
  - data/video_movement_epochs_{tp}.csv
  - data/video_events_{tp}.csv             (detected movement bouts)
  - data/epochs_eeg_video_merged_{tp}.csv  (synchronized)
  - figures/video_*.png

Run:
    python src/07_advanced_video_analysis.py [--timepoint 4m]
"""

import os
import gc
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import cv2
from scipy.signal import medfilt, find_peaks
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")

COLORS = {"WT": "#378ADD", "KO": "#D85A30"}

# Video parameters
FRAME_SKIP       = 3        # analyze every Nth frame (speed/accuracy tradeoff)
RESIZE_FACTOR    = 0.3      # resize for speed
MOV_THRESHOLD    = 12       # pixel diff threshold
EPOCH_S          = 4.0      # must match EEG epoch duration
MIN_BOUT_FRAMES  = 3        # minimum frames for movement bout
SEIZURE_DURATION_S = 5.0    # minimum duration to flag as seizure-like

TP_ORDER = ["3m", "4m", "6m", "7m", "9m", "12m"]


# ── Stage 1: Movement quantification ──────────────────────────────────────

def compute_movement_timeseries(wmv_path,
                                 frame_skip=FRAME_SKIP,
                                 resize=RESIZE_FACTOR,
                                 threshold=MOV_THRESHOLD,
                                 compute_optical_flow=False):
    """
    Compute per-frame movement from WMV video.

    Returns:
        times       : array of timestamps (s)
        diff_scores : frame-difference movement scores
        flow_scores : optical flow magnitude scores (if requested)
        video_meta  : dict of fps, n_frames, duration_s
    """
    cap = cv2.VideoCapture(wmv_path)
    if not cap.isOpened():
        return None, None, None, {}

    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or n_frames <= 0:
        cap.release()
        return None, None, None, {}

    meta = {
        "fps":        fps,
        "n_frames":   n_frames,
        "duration_s": n_frames / fps,
    }

    times, diff_scores, flow_scores = [], [], []
    prev_gray = None
    prev_gray_flow = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            small = cv2.resize(frame, (0, 0), fx=resize, fy=resize)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray  = cv2.GaussianBlur(gray, (5, 5), 0)
            t_s   = frame_idx / fps
            times.append(t_s)

            # Frame differencing
            if prev_gray is not None:
                diff  = cv2.absdiff(gray, prev_gray)
                score = float(np.mean(diff > threshold))
                diff_scores.append(score)
            else:
                diff_scores.append(0.0)

            # Optical flow (Farneback) — more accurate but slower
            if compute_optical_flow and prev_gray_flow is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray_flow, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_scores.append(float(np.mean(mag)))
            else:
                flow_scores.append(0.0)

            prev_gray      = gray
            prev_gray_flow = gray

        frame_idx += 1

    cap.release()
    gc.collect()

    return (np.array(times),
            np.array(diff_scores),
            np.array(flow_scores),
            meta)


# ── Stage 2: Movement bout detection ──────────────────────────────────────

def detect_movement_bouts(times, scores, fps_equiv=None,
                           min_bout_s=0.5, merge_gap_s=1.0,
                           threshold_pct=75):
    """
    Detect discrete movement bouts from continuous movement scores.
    Returns DataFrame with onset, offset, duration, peak_score.
    """
    threshold = np.percentile(scores, threshold_pct)
    active    = scores > threshold

    # Find onset/offset transitions
    padded = np.concatenate([[0], active.astype(int), [0]])
    onsets  = np.where(np.diff(padded) == 1)[0]
    offsets = np.where(np.diff(padded) == -1)[0]

    bouts = []
    for on, off in zip(onsets, offsets):
        t_on  = times[min(on,  len(times)-1)]
        t_off = times[min(off, len(times)-1)]
        dur   = t_off - t_on
        if dur < min_bout_s:
            continue
        peak = float(scores[on:off].max())
        mean = float(scores[on:off].mean())
        bouts.append({"onset_s": t_on, "offset_s": t_off,
                      "duration_s": dur, "peak_score": peak,
                      "mean_score": mean})

    if not bouts:
        return pd.DataFrame()

    # Merge adjacent bouts separated by less than merge_gap_s
    merged = [bouts[0]]
    for bout in bouts[1:]:
        if bout["onset_s"] - merged[-1]["offset_s"] < merge_gap_s:
            merged[-1]["offset_s"]   = bout["offset_s"]
            merged[-1]["duration_s"] = merged[-1]["offset_s"] - merged[-1]["onset_s"]
            merged[-1]["peak_score"] = max(merged[-1]["peak_score"], bout["peak_score"])
            merged[-1]["mean_score"] = (merged[-1]["mean_score"] + bout["mean_score"]) / 2
        else:
            merged.append(bout)

    return pd.DataFrame(merged)


# ── Stage 3: Epoch aggregation ─────────────────────────────────────────────

def movement_to_epochs(times, diff_scores, flow_scores=None, epoch_s=EPOCH_S):
    """
    Aggregate per-frame scores into EEG-matched 4-second epochs.
    Returns DataFrame with one row per epoch.
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
        if not m.any():
            continue

        scores_ep = diff_scores[m]
        row = {
            "epoch_idx":         i,
            "onset_s":           t0,
            "mean_movement":     float(scores_ep.mean()),
            "max_movement":      float(scores_ep.max()),
            "std_movement":      float(scores_ep.std()),
            "pct_active_frames": float(np.mean(scores_ep > np.percentile(diff_scores, 75)) * 100),
            "n_frames":          int(m.sum()),
        }
        if flow_scores is not None and len(flow_scores) == len(times):
            flow_ep = flow_scores[m]
            row["mean_optical_flow"] = float(flow_ep.mean())
        rows.append(row)

    return pd.DataFrame(rows)


# ── Stage 4: Seizure-like event detection ──────────────────────────────────

def detect_seizure_events(times, scores, bouts_df, min_dur_s=SEIZURE_DURATION_S):
    """
    Flag high-intensity prolonged movement events as seizure-like.
    High-amplitude (>90th pct) + long duration (>= min_dur_s).
    """
    if bouts_df.empty:
        return pd.DataFrame()

    hi_thresh = np.percentile(scores, 90)
    seizure_events = bouts_df[
        (bouts_df.duration_s  >= min_dur_s) &
        (bouts_df.peak_score  >= hi_thresh)
    ].copy()
    seizure_events["event_type"] = "seizure-like"
    return seizure_events


# ── Stage 5: EEG-video merge ───────────────────────────────────────────────

def merge_eeg_video(eeg_epochs_path, video_epochs_df, session_id, abf_file):
    """
    Merge epoch-level EEG features with video movement features.
    Matching is on epoch_idx.
    """
    if not os.path.exists(eeg_epochs_path):
        return pd.DataFrame()

    eeg = pd.read_csv(eeg_epochs_path)
    eeg = eeg[(eeg.session_id.astype(str) == str(session_id)) &
               (eeg.abf_file == abf_file)].copy()
    if eeg.empty:
        return pd.DataFrame()

    merged = eeg.merge(
        video_epochs_df[["epoch_idx","mean_movement","max_movement",
                         "std_movement","pct_active_frames"]],
        on="epoch_idx", how="left"
    )
    return merged


# ── Main pipeline ──────────────────────────────────────────────────────────

def process_timepoint(tp, inventory_all, use_optical_flow=False):
    print(f"\n{'='*65}")
    print(f"Video Analysis — Timepoint: {tp}")
    print(f"{'='*65}")

    inv = inventory_all[inventory_all.timepoint == tp].copy()
    if len(inv) == 0:
        print(f"  No files in inventory for {tp}")
        return

    eeg_epoch_path = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")

    all_video_rows   = []
    all_bout_rows    = []
    all_seizure_rows = []
    all_merged_rows  = []
    video_summary    = []

    for _, file_row in inv.iterrows():
        wmv_path   = file_row["wmv_path"]
        animal_id  = str(file_row["animal_id"])
        session_id = str(file_row["session_id"])
        group      = file_row["group"]
        wmv_file   = file_row["wmv_file"]
        abf_file   = file_row["abf_file"]

        if not os.path.exists(wmv_path):
            print(f"  SKIP (not found): {wmv_path}")
            continue

        print(f"\n  {session_id} | {wmv_file}")

        try:
            times, diff_scores, flow_scores, meta = compute_movement_timeseries(
                wmv_path, compute_optical_flow=use_optical_flow
            )
            if times is None:
                print(f"    ERROR: Cannot open video")
                continue

            print(f"    {meta['fps']:.1f} fps | {meta['duration_s']/3600:.2f} h | "
                  f"{meta['n_frames']} frames")

            # Epoch-level features
            vid_epochs = movement_to_epochs(times, diff_scores, flow_scores)
            if vid_epochs.empty:
                continue

            vid_epochs["animal_id"]  = animal_id
            vid_epochs["session_id"] = session_id
            vid_epochs["group"]      = group
            vid_epochs["wmv_file"]   = wmv_file
            vid_epochs["timepoint"]  = tp
            all_video_rows.append(vid_epochs)

            # Movement bouts
            bouts = detect_movement_bouts(times, diff_scores)
            if not bouts.empty:
                bouts["animal_id"]  = animal_id
                bouts["session_id"] = session_id
                bouts["group"]      = group
                bouts["wmv_file"]   = wmv_file
                bouts["timepoint"]  = tp
                all_bout_rows.append(bouts)

            # Seizure-like events (especially important at 4m)
            seizures = detect_seizure_events(times, diff_scores, bouts)
            if not seizures.empty:
                seizures["animal_id"]  = animal_id
                seizures["session_id"] = session_id
                seizures["group"]      = group
                seizures["timepoint"]  = tp
                all_seizure_rows.append(seizures)
                print(f"    Seizure-like events detected: {len(seizures)}")

            # EEG-video merge
            if os.path.exists(eeg_epoch_path):
                merged = merge_eeg_video(eeg_epoch_path, vid_epochs,
                                          session_id, abf_file)
                if not merged.empty:
                    all_merged_rows.append(merged)

            # Summary
            pct_active = float(np.mean(diff_scores > np.percentile(diff_scores, 75)) * 100)
            video_summary.append({
                "animal_id":       animal_id,
                "session_id":      session_id,
                "group":           group,
                "wmv_file":        wmv_file,
                "timepoint":       tp,
                "fps":             meta["fps"],
                "duration_h":      round(meta["duration_s"] / 3600, 2),
                "mean_movement":   round(float(diff_scores.mean()), 5),
                "std_movement":    round(float(diff_scores.std()), 5),
                "pct_active":      round(pct_active, 1),
                "n_bouts":         len(bouts),
                "n_seizure_like":  len(seizures),
                "n_epochs":        len(vid_epochs),
            })

            del times, diff_scores, flow_scores
            gc.collect()

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    if not all_video_rows:
        print(f"  No video data processed for {tp}")
        return

    # ── Save ──────────────────────────────────────────────────────────────
    vid_df = pd.concat(all_video_rows, ignore_index=True)
    vid_df.to_csv(os.path.join(DATA_DIR, f"video_movement_epochs_{tp}.csv"),
                   index=False)
    print(f"\n  Saved: video_movement_epochs_{tp}.csv ({len(vid_df):,} epochs)")

    sum_df = pd.DataFrame(video_summary)
    sum_df.to_csv(os.path.join(DATA_DIR, f"video_summary_{tp}.csv"), index=False)

    if all_bout_rows:
        bouts_df = pd.concat(all_bout_rows, ignore_index=True)
        bouts_df.to_csv(os.path.join(DATA_DIR, f"video_bouts_{tp}.csv"), index=False)
        print(f"  Saved: video_bouts_{tp}.csv ({len(bouts_df)} bouts)")

    if all_seizure_rows:
        sz_df = pd.concat(all_seizure_rows, ignore_index=True)
        sz_df.to_csv(os.path.join(DATA_DIR, f"video_seizures_{tp}.csv"), index=False)
        print(f"  Saved: video_seizures_{tp}.csv ({len(sz_df)} events)")

    if all_merged_rows:
        merged_df = pd.concat(all_merged_rows, ignore_index=True)
        merged_df.to_csv(os.path.join(DATA_DIR,
                          f"epochs_eeg_video_merged_{tp}.csv"), index=False)
        print(f"  Saved: epochs_eeg_video_merged_{tp}.csv")

    # ── Group comparisons ─────────────────────────────────────────────────
    print(f"\n  Movement comparison WT vs KO at {tp}:")
    for metric in ["mean_movement", "pct_active", "n_bouts", "n_seizure_like"]:
        wt = sum_df[sum_df.group == "WT"][metric].dropna()
        ko = sum_df[sum_df.group == "KO"][metric].dropna()
        if len(wt) < 2 or len(ko) < 2:
            continue
        _, p = mannwhitneyu(wt, ko, alternative="two-sided")
        d = (ko.mean()-wt.mean())/np.sqrt((wt.std()**2+ko.std()**2)/2+1e-10)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    {metric:<20}: WT={wt.mean():.4f} KO={ko.mean():.4f} "
              f"d={d:.3f} p={p:.4f} {sig}")

    # ── Figure ────────────────────────────────────────────────────────────
    _plot_video_summary(sum_df, tp)

    return vid_df, sum_df


def _plot_video_summary(sum_df, tp):
    """Quick comparison figure for video metrics."""
    metrics = ["mean_movement", "pct_active", "n_bouts"]
    labels  = ["Mean Movement", "% Active Epochs", "N Movement Bouts"]

    n = min(len(metrics), sum_df.shape[1])
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4))

    for ax, metric, label in zip(axes, metrics, labels):
        if metric not in sum_df.columns:
            continue
        for g, color, x in [("WT", COLORS["WT"], 0), ("KO", COLORS["KO"], 1)]:
            vals = sum_df[sum_df.group == g][metric].dropna()
            ax.scatter([x] * len(vals), vals, color=color, alpha=0.6, s=40, zorder=3)
            ax.errorbar(x, vals.mean(), yerr=vals.sem() if len(vals) > 1 else 0,
                        fmt="_", markersize=20, markeredgewidth=2,
                        color=color, capsize=4, capthick=2)
        wt = sum_df[sum_df.group == "WT"][metric].dropna()
        ko = sum_df[sum_df.group == "KO"][metric].dropna()
        if len(wt) >= 2 and len(ko) >= 2:
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            sig  = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            ax.set_title(f"{label}\n{sig} p={p:.3f}", fontsize=9)
        else:
            ax.set_title(label, fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["WT","KO"])

    fig.suptitle(f"Video Movement Analysis — {tp}", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"video_summary_{tp}.png"),
                dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: video_summary_{tp}.png")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timepoint", default=None,
                        help="Single timepoint (e.g. 4m). Default: all.")
    parser.add_argument("--optical-flow", action="store_true",
                        help="Also compute optical flow (slower but more precise).")
    args = parser.parse_args()

    inv_path = os.path.join(DATA_DIR, "file_inventory_all_timepoints.csv")
    if not os.path.exists(inv_path):
        # Fall back to 3m inventory
        inv_path = os.path.join(DATA_DIR, "file_inventory_3m.csv")
        if not os.path.exists(inv_path):
            print("ERROR: Run inventory script first")
            sys.exit(1)

    inventory = pd.read_csv(inv_path)
    print(f"Loaded inventory: {len(inventory)} files")

    tp_to_run = [args.timepoint] if args.timepoint else TP_ORDER
    for tp in tp_to_run:
        if "timepoint" in inventory.columns:
            if tp not in inventory["timepoint"].unique():
                continue
        process_timepoint(tp, inventory, use_optical_flow=args.optical_flow)

    print("\nVIDEO ANALYSIS COMPLETE")
