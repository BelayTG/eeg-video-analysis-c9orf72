"""
04_multi_timepoint_inventory.py
================================
Builds a unified file inventory across ALL 6 timepoints for BOTH scenarios.

SCENARIO A — Animal-level (n = number of animals)
  Statistical unit: 5-digit animal_id (e.g. 16572)
  All ABF files for an animal at a timepoint are pooled.

SCENARIO B — Session-level (n = 2 × number of animals)
  Statistical unit: 6-digit session_id (e.g. 165721, 165722)
  Files for each animal are split equally into two sessions.
  EXCEPTION — 4m timepoint:
    Instead of splitting, ALL files are DUPLICATED into both sessions
    (session1 = all files, session2 = identical copy of all files).
    This preserves the acute KA day as a single coherent epoch while
    still allowing the session-level design to be used.

Timepoints:
  3m:  F:\EEG\Baseline (24hrs)_3monthsold_before_KA       WT=9, KO=9
  4m:  F:\EEG\EEG_After_20KA_i.p._Treatment               WT=9, KO=9
  6m:  F:\EEG\Baseline (24hrs)_6monthsold_20mgkgKA_Rx     WT=8, KO=8
  7m:  F:\EEG\Baseline_7month                             WT=8, KO=6
  9m:  F:\EEG\EEG 10mon                                   WT=7, KO=4
 12m:  F:\EEG\EEG 1yr                                     WT=6, KO=4

Exclusions:
  - Animal 16654: excluded from 6m onward (euthanized after 4m)
  - 12m: ignore "All recordings" and "All recordings.zip" subfolders

Run:
    python src/04_multi_timepoint_inventory.py
"""

import os
import pandas as pd
import numpy as np

PORT_DIR = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR = os.path.join(PORT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TIMEPOINTS = {
    "3m":  r"F:\EEG\Baseline (24hrs)_3monthsold_before_KA",
    "4m":  r"F:\EEG\EEG_After_20KA_i.p._Treatment",
    "6m":  r"F:\EEG\Baseline (24hrs)_6monthsold_20mgkgKA_Rx",
    "7m":  r"F:\EEG\Baseline_7month",
    "9m":  r"F:\EEG\EEG 10mon",
    "12m": r"F:\EEG\EEG 1yr",
}

# Animals excluded from specific timepoints onward
EXCLUSIONS = {
    "6m": {"16654"}, "7m": {"16654"},
    "9m": {"16654"}, "12m": {"16654"},
}

IGNORE_SUBFOLDERS = {"All recordings", "All recordings.zip"}
GROUPS            = ["WT", "KO"]
ABF_MIN_BYTES     = 100 * 1024 * 1024   # 100 MB
WMV_MIN_BYTES     = 400 * 1024 * 1024   # 400 MB

# The 4m timepoint duplicates all files into both sessions
TP_4M_DUPLICATE = "4m"


def find_valid_files(folder):
    """Recursively find valid ABF (≥100 MB) and WMV (≥400 MB) files."""
    abf_files, wmv_files = [], []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in IGNORE_SUBFOLDERS]
        for f in files:
            fpath = os.path.join(root, f)
            try:
                fsize = os.path.getsize(fpath)
            except OSError:
                continue
            if f.lower().endswith(".abf") and fsize >= ABF_MIN_BYTES:
                abf_files.append(fpath)
            elif f.lower().endswith(".wmv") and fsize >= WMV_MIN_BYTES:
                wmv_files.append(fpath)
    return sorted(abf_files), sorted(wmv_files)


def build_inventory_rows(animal_id, group, tp, abf_files, wmv_files):
    """
    Build inventory rows for both Scenario A and Scenario B.

    Scenario A: all files → single animal_id (no session split)
    Scenario B: files split 50/50 into session 1 and session 2
                EXCEPT at 4m: all files duplicated into both sessions
    """
    n_paired = min(len(abf_files), len(wmv_files))
    if n_paired == 0:
        return []

    abf_paired = abf_files[:n_paired]
    wmv_paired = wmv_files[:n_paired]

    # ── Scenario A: all files under one animal_id ──────────────────────────
    rows_A = []
    for abf, wmv in zip(abf_paired, wmv_paired):
        rows_A.append({
            "scenario":   "A",
            "animal_id":  animal_id,
            "session_id": animal_id,      # same as animal — no session suffix
            "session":    None,
            "group":      group,
            "timepoint":  tp,
            "abf_file":   os.path.basename(abf),
            "wmv_file":   os.path.basename(wmv),
            "abf_path":   abf,
            "wmv_path":   wmv,
            "abf_gb":     round(os.path.getsize(abf) / 1024**3, 3),
            "wmv_gb":     round(os.path.getsize(wmv) / 1024**3, 3),
        })

    # ── Scenario B: split into two sessions ───────────────────────────────
    rows_B = []

    if tp == TP_4M_DUPLICATE:
        # At 4m: duplicate all files into session 1 AND session 2
        sessions_B = {
            f"{animal_id}1": (abf_paired, wmv_paired),   # all files
            f"{animal_id}2": (abf_paired, wmv_paired),   # identical copy
        }
    else:
        # All other timepoints: split 50/50 by file order
        half = max(1, n_paired // 2)
        sessions_B = {
            f"{animal_id}1": (abf_paired[:half],  wmv_paired[:half]),
            f"{animal_id}2": (abf_paired[half:],  wmv_paired[half:]),
        }

    for session_id, (abfs, wmvs) in sessions_B.items():
        session_num = int(session_id[-1])
        for abf, wmv in zip(abfs, wmvs):
            rows_B.append({
                "scenario":   "B",
                "animal_id":  animal_id,
                "session_id": session_id,
                "session":    session_num,
                "group":      group,
                "timepoint":  tp,
                "abf_file":   os.path.basename(abf),
                "wmv_file":   os.path.basename(wmv),
                "abf_path":   abf,
                "wmv_path":   wmv,
                "abf_gb":     round(os.path.getsize(abf) / 1024**3, 3),
                "wmv_gb":     round(os.path.getsize(wmv) / 1024**3, 3),
            })

    return rows_A + rows_B


print("=" * 70)
print("EEG-Video Portfolio — Multi-Timepoint Inventory (Scenarios A + B)")
print("=" * 70)
print()
print("Scenario A: 5-digit animal_id as statistical unit")
print("Scenario B: 6-digit session_id (files split 50/50 per animal,")
print("            EXCEPT 4m where all files are duplicated into both sessions)")
print()

all_rows = []

for tp, tp_path in TIMEPOINTS.items():
    print(f"\n{'─'*65}")
    print(f"Timepoint: {tp}  →  {tp_path}")
    if tp == TP_4M_DUPLICATE:
        print(f"  *** 4m: Scenario B will DUPLICATE files into both sessions ***")
    print(f"{'─'*65}")

    excluded = EXCLUSIONS.get(tp, set())
    if excluded:
        print(f"  Excluded animals: {excluded}")

    if not os.path.isdir(tp_path):
        print(f"  WARNING: Directory not found — {tp_path}")
        continue

    for group in GROUPS:
        group_dir = os.path.join(tp_path, group)
        if not os.path.isdir(group_dir):
            print(f"  {group}: no WT/KO subfolder, trying flat structure")
            group_dir = tp_path

        animal_folders = sorted([
            d for d in os.listdir(group_dir)
            if os.path.isdir(os.path.join(group_dir, d))
            and d not in IGNORE_SUBFOLDERS
            and not d.startswith(".")
        ])

        print(f"\n  {group} ({len(animal_folders)} folders):")

        for afolder in animal_folders:
            animal_id = afolder[:5]

            if animal_id in excluded:
                print(f"    {animal_id}: EXCLUDED")
                continue

            animal_path = os.path.join(group_dir, afolder)
            abf_files, wmv_files = find_valid_files(animal_path)
            n_paired = min(len(abf_files), len(wmv_files))

            if n_paired == 0:
                print(f"    {animal_id}: SKIP — no valid paired files")
                continue

            rows = build_inventory_rows(
                animal_id, group, tp,
                abf_files, wmv_files
            )
            all_rows.extend(rows)

            # Print summary
            half = max(1, n_paired // 2)
            s2_n = n_paired if tp == TP_4M_DUPLICATE else n_paired - half
            total_gb = sum(os.path.getsize(f) / 1024**3 for f in abf_files[:n_paired])
            dup_flag = " [DUPLICATED]" if tp == TP_4M_DUPLICATE else ""
            print(f"    {animal_id}: {n_paired} pairs | "
                  f"S1={half} S2={s2_n} files{dup_flag} | "
                  f"{total_gb:.1f} GB ABF")

# ── Build DataFrames ───────────────────────────────────────────────────────
df_all = pd.DataFrame(all_rows)
df_A   = df_all[df_all.scenario == "A"].copy()
df_B   = df_all[df_all.scenario == "B"].copy()

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("INVENTORY SUMMARY")
print(f"{'='*70}")

for scenario, df, unit in [("A (animal-level)", df_A, "animal_id"),
                             ("B (session-level)", df_B, "session_id")]:
    print(f"\nScenario {scenario}:")
    print(f"  {'TP':<5} {'Group':<5} {'N files':<9} {'N units':<10} {'ABF GB':<10}")
    for tp in TIMEPOINTS:
        for group in GROUPS:
            sub = df[(df.timepoint==tp) & (df.group==group)]
            if len(sub) == 0: continue
            n_units = sub[unit].nunique()
            gb = sub["abf_gb"].sum()
            # Avoid double-counting duplicates at 4m for GB
            if scenario.startswith("B") and tp == TP_4M_DUPLICATE:
                gb = gb / 2
            print(f"  {tp:<5} {group:<5} {len(sub):<9} {n_units:<10} {gb:.1f}")

print(f"\nTotal ABF data (Scenario A): {df_A['abf_gb'].sum():.1f} GB")

# ── Save ───────────────────────────────────────────────────────────────────
out_all = os.path.join(DATA_DIR, "file_inventory_all_timepoints.csv")
df_all.to_csv(out_all, index=False)
print(f"\nSaved: file_inventory_all_timepoints.csv ({len(df_all)} rows, both scenarios)")

out_A = os.path.join(DATA_DIR, "file_inventory_scenarioA.csv")
df_A.to_csv(out_A, index=False)
print(f"Saved: file_inventory_scenarioA.csv ({len(df_A)} rows)")

out_B = os.path.join(DATA_DIR, "file_inventory_scenarioB.csv")
df_B.to_csv(out_B, index=False)
print(f"Saved: file_inventory_scenarioB.csv ({len(df_B)} rows)")

for tp in TIMEPOINTS:
    sub_A = df_A[df_A.timepoint==tp]
    sub_B = df_B[df_B.timepoint==tp]
    if len(sub_A) > 0:
        sub_A.to_csv(os.path.join(DATA_DIR, f"file_inventory_{tp}_A.csv"), index=False)
    if len(sub_B) > 0:
        sub_B.to_csv(os.path.join(DATA_DIR, f"file_inventory_{tp}_B.csv"), index=False)

print("\nINVENTORY COMPLETE")
