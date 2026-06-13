"""
00_setup_and_inventory.py
=========================
Sets up the eeg-video-analysis-c9orf72 portfolio and builds a file inventory
of all valid ABF and WMV files at the 3m baseline timepoint.

Rules:
  - ABF files: include only if size >= 100MB (> ~1 hour at 5000 Hz)
  - WMV files: include only if size >= 400MB (> ~1 hour video)
  - Files are sorted by name (chronological order)
  - Split into Session 1 (first half) and Session 2 (second half)
  - Animal ID = 5-digit folder prefix
  - Session ID = animal_id + "1" or animal_id + "2" (6-digit)

Run from portfolio root:
    python src/00_setup_and_inventory.py
"""

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
VIDEO_BASE = r"F:\EEG\Baseline (24hrs)_3monthsold_before_KA"
PORT_DIR   = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR   = os.path.join(PORT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Size thresholds
ABF_MIN_BYTES = 100 * 1024 * 1024   # 100 MB
WMV_MIN_BYTES = 400 * 1024 * 1024   # 400 MB

GROUPS = ["WT", "KO"]

print("="*60)
print("EEG-Video Portfolio — File Inventory")
print("="*60)

rows = []

for group in GROUPS:
    group_dir = os.path.join(VIDEO_BASE, group)
    if not os.path.isdir(group_dir):
        print(f"WARNING: {group_dir} not found"); continue

    animal_folders = sorted([
        d for d in os.listdir(group_dir)
        if os.path.isdir(os.path.join(group_dir, d))
    ])

    print(f"\n{group} ({len(animal_folders)} animals):")

    for mfolder in animal_folders:
        animal_id  = mfolder[:5]
        mouse_path = os.path.join(group_dir, mfolder)
        all_files  = sorted(os.listdir(mouse_path))

        # Filter valid ABF files
        abf_files = sorted([
            f for f in all_files
            if f.lower().endswith(".abf") and
            os.path.getsize(os.path.join(mouse_path, f)) >= ABF_MIN_BYTES
        ])

        # Filter valid WMV files
        wmv_files = sorted([
            f for f in all_files
            if f.lower().endswith(".wmv") and
            os.path.getsize(os.path.join(mouse_path, f)) >= WMV_MIN_BYTES
        ])

        n_abf = len(abf_files)
        n_wmv = len(wmv_files)

        if n_abf == 0:
            print(f"  {animal_id}: SKIP — no valid ABF files")
            continue
        if n_wmv == 0:
            print(f"  {animal_id}: SKIP — no valid WMV files")
            continue

        # Check alignment — need equal numbers or close
        n_paired = min(n_abf, n_wmv)
        if n_paired < 4:
            print(f"  {animal_id}: SKIP — too few paired files ({n_paired})")
            continue

        # Use only paired files (matched by position)
        abf_paired = abf_files[:n_paired]
        wmv_paired = wmv_files[:n_paired]

        # Split into two sessions
        half      = n_paired // 2
        sessions  = {
            f"{animal_id}1": (abf_paired[:half],  wmv_paired[:half]),
            f"{animal_id}2": (abf_paired[half:],   wmv_paired[half:]),
        }

        abf_sizes = [os.path.getsize(os.path.join(mouse_path,f))
                     for f in abf_paired]
        total_gb  = sum(abf_sizes) / (1024**3)

        print(f"  {animal_id}: {n_paired} pairs | "
              f"S1={half} files | S2={n_paired-half} files | "
              f"{total_gb:.1f} GB ABF")

        for session_id, (abfs, wmvs) in sessions.items():
            for abf, wmv in zip(abfs, wmvs):
                abf_path = os.path.join(mouse_path, abf)
                wmv_path = os.path.join(mouse_path, wmv)
                abf_gb   = os.path.getsize(abf_path) / (1024**3)
                wmv_gb   = os.path.getsize(wmv_path) / (1024**3)
                session  = int(session_id[-1])
                rows.append({
                    "animal_id":  animal_id,
                    "session_id": session_id,
                    "session":    session,
                    "group":      group,
                    "timepoint":  "3m",
                    "abf_file":   abf,
                    "wmv_file":   wmv,
                    "abf_path":   abf_path,
                    "wmv_path":   wmv_path,
                    "abf_gb":     round(abf_gb, 3),
                    "wmv_gb":     round(wmv_gb, 3),
                    "paired":     True,
                })

df = pd.DataFrame(rows)

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"INVENTORY SUMMARY")
print(f"{'='*60}")
print(f"\nTotal paired ABF-WMV files: {len(df)}")
print(f"\nScenario A — n=36 (session-level, 6-digit IDs):")
for group in GROUPS:
    n = df[df["group"]==group]["session_id"].nunique()
    print(f"  {group}: {n} sessions")

print(f"\nScenario B — n=18 (animal-level, 5-digit IDs):")
for group in GROUPS:
    n = df[df["group"]==group]["animal_id"].nunique()
    print(f"  {group}: {n} animals")

print(f"\nTotal recording time:")
print(f"  ABF: {df['abf_gb'].sum():.1f} GB")
print(f"  WMV: {df['wmv_gb'].sum():.1f} GB")

print(f"\nFiles per session:")
print(df.groupby(["group","session_id"])["abf_file"].count().to_string())

# ── Save ───────────────────────────────────────────────────────────────────
out = os.path.join(DATA_DIR, "file_inventory_3m.csv")
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(f"\nSession ID examples:")
print(df[["animal_id","session_id","session","group"]].drop_duplicates().head(10).to_string())
