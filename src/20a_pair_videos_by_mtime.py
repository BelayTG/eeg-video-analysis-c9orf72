"""
20a_pair_videos_by_mtime.py
============================
Pairs each ABF file to its matching video by MODIFIED-TIME proximity, since the
two share no common filename token. Builds a mapping CSV that script 20 reads.

WHY A SEPARATE STEP: mis-pairing an ABF with the wrong animal's video would
silently invalidate the entire EEG-video analysis (you would correlate one
animal's cortex with another's movement). So pairing is done once, with safety
checks, and written to a CSV you VERIFY BY EYE before any coupling analysis
trusts it.

SAFETY LOGIC
  - For each ABF, find the nearest video by |mtime difference|.
  - Reject a pairing if the gap exceeds MAX_GAP_MIN (default 30 min).
  - Reject if the same video is the nearest match for two different ABFs
    (ambiguous) — flag both for manual resolution.
  - Report every pairing with its time gap so you can scan for anomalies.
  - Writes data/video_abf_map.csv with a 'verified' column you set to 1
    after checking (script 20 will refuse unverified rows unless you override).

USAGE
  1) Set FOLDER below to where the interleaved ABF+video files live.
  2) python src/20a_pair_videos_by_mtime.py
  3) Open data/video_abf_map.csv, eyeball the gaps, fix any flagged rows,
     set verified=1 on the rows you trust.
  4) Then run script 20 (it reads this CSV).
"""

import os
import glob
import datetime as dt
import pandas as pd

PORT_DIR = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR = os.path.join(PORT_DIR, "data")

# Folder containing the interleaved ABF + video files
FOLDER   = os.path.join(PORT_DIR, "data")     # adjust if elsewhere
VIDEO_EXTS = [".avi", ".mp4", ".mov", ".mkv", ".wmv"]
ABF_EXT  = ".abf"

MAX_GAP_MIN = 30        # reject pairings with a larger modified-time gap
OUT_CSV  = os.path.join(DATA_DIR, "video_abf_map.csv")


def mtime(path):
    return dt.datetime.fromtimestamp(os.path.getmtime(path))


def main():
    if not os.path.isdir(FOLDER):
        print(f"FOLDER not found: {FOLDER}")
        print("Edit FOLDER at the top of this script to point at your data directory.")
        return

    abfs = sorted(glob.glob(os.path.join(FOLDER, f"*{ABF_EXT}")))
    videos = []
    for ext in VIDEO_EXTS:
        videos += glob.glob(os.path.join(FOLDER, f"*{ext}"))
    videos = sorted(videos)

    print(f"Found {len(abfs)} ABF files and {len(videos)} video files in:\n  {FOLDER}\n")
    if not abfs or not videos:
        print("Need both ABF and video files present. Check FOLDER and extensions.")
        return

    vid_mtimes = {v: mtime(v) for v in videos}

    rows = []
    nearest_video_count = {}  # video -> how many ABFs picked it (ambiguity check)
    for abf in abfs:
        at = mtime(abf)
        # nearest video by |mtime gap|
        best_v, best_gap = None, None
        for v, vt in vid_mtimes.items():
            gap = abs((vt - at).total_seconds())
            if best_gap is None or gap < best_gap:
                best_gap = gap; best_v = v
        gap_min = best_gap/60.0 if best_gap is not None else None
        within = (gap_min is not None) and (gap_min <= MAX_GAP_MIN)
        nearest_video_count[best_v] = nearest_video_count.get(best_v, 0) + 1
        rows.append({
            "abf_file": os.path.basename(abf),
            "abf_mtime": at.strftime("%Y-%m-%d %H:%M:%S"),
            "video_file": os.path.basename(best_v) if best_v else "",
            "video_mtime": vid_mtimes[best_v].strftime("%Y-%m-%d %H:%M:%S") if best_v else "",
            "gap_minutes": round(gap_min, 2) if gap_min is not None else None,
            "within_window": int(within),
            "ambiguous": 0,         # filled below
            "verified": 0,          # YOU set to 1 after eyeballing
        })

    # mark ambiguous pairings (a video chosen by >1 ABF)
    for r in rows:
        if r["video_file"] and nearest_video_count.get(
                os.path.join(FOLDER, r["video_file"]), 0) > 1:
            r["ambiguous"] = 1

    df = pd.DataFrame(rows).sort_values("abf_mtime")
    df.to_csv(OUT_CSV, index=False)

    # Report
    n_ok = int((df["within_window"] == 1).sum())
    n_amb = int((df["ambiguous"] == 1).sum())
    n_far = int((df["within_window"] == 0).sum())
    print(f"Wrote mapping: {OUT_CSV}\n")
    print(f"  {n_ok}/{len(df)} pairings within {MAX_GAP_MIN} min")
    if n_far:
        print(f"  {n_far} pairings EXCEED the window — likely missing/extra video:")
        for _, r in df[df.within_window == 0].iterrows():
            print(f"     {r.abf_file}  ->  {r.video_file or '(none)'}  gap={r.gap_minutes} min")
    if n_amb:
        print(f"  {n_amb} AMBIGUOUS pairings (one video matched to multiple ABFs):")
        for _, r in df[df.ambiguous == 1].iterrows():
            print(f"     {r.abf_file}  ->  {r.video_file}  gap={r.gap_minutes} min")
    print("\nNEXT STEPS:")
    print(f"  1. Open {OUT_CSV}")
    print("  2. Scan 'gap_minutes' — typical gaps should be similar; an outlier = a mis-pair.")
    print("  3. Resolve any within_window=0 or ambiguous=1 rows by hand (fix video_file).")
    print("  4. Set verified=1 on every row you trust.")
    print("  5. Run script 20 (it reads only verified=1 rows).")
    print("\nNOTE on timestamps: if files were copied/moved/cloud-synced, modified-time")
    print("may have changed. If the gaps look wrong, the timestamps were not preserved —")
    print("tell me and we will need another linking method (e.g. a recording log).")


if __name__ == "__main__":
    main()
