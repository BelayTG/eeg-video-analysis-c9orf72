"""
abf_paths.py  —  inventory-based resolver (matches the portfolio convention)
===========================================================================
The whole portfolio uses data/file_inventory_all_timepoints.csv as the single
source of truth. Every row has the full abf_path, the paired wmv_path (video),
plus animal_id, group (genotype), timepoint, scenario, session_id.

This module exposes the same iteration interface the Tier-1/2 scripts expect,
but sourced from the inventory rather than walking the disk.

  iter_recordings(tp, scenario="A")
      -> yields (abf_path, animal_id, group) for every row at that timepoint
  video_for(abf_path)
      -> the paired wmv_path from the inventory (None if absent/missing)
"""

import os
import pandas as pd

PORT_DIR = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR = os.path.join(PORT_DIR, "data")
INVENTORY = os.path.join(DATA_DIR, "file_inventory_all_timepoints.csv")

_INV = None


def _load_inventory():
    global _INV
    if _INV is None:
        if not os.path.exists(INVENTORY):
            _INV = pd.DataFrame()
        else:
            df = pd.read_csv(INVENTORY)
            df["animal_id"] = df["animal_id"].astype(str)
            _INV = df
    return _INV


def iter_recordings(tp, scenario="A"):
    """Yield (abf_path, animal_id, group) for one timepoint (Scenario A = animal-level)."""
    inv = _load_inventory()
    if inv.empty:
        return
    sub = inv[inv["timepoint"] == tp]
    if "scenario" in sub.columns and scenario is not None:
        sub = sub[sub["scenario"] == scenario]
    for _, r in sub.iterrows():
        ap = r.get("abf_path")
        if not isinstance(ap, str) or not ap:
            continue
        yield ap, str(r["animal_id"]), r["group"]


def video_for(abf_path):
    """Return the paired video path for an ABF, from the inventory."""
    inv = _load_inventory()
    if inv.empty:
        return None
    row = inv[inv["abf_path"] == abf_path]
    if len(row) == 0:
        # match on basename as fallback
        base = os.path.basename(abf_path)
        row = inv[inv["abf_file"] == base]
    if len(row) == 0:
        return None
    wp = row.iloc[0].get("wmv_path")
    if isinstance(wp, str) and wp:
        return wp
    return None


def find_video_near(abf_path, max_gap_min=None):
    """Compatibility shim: returns (video_path, 0.0) using the inventory pairing."""
    v = video_for(abf_path)
    return (v, 0.0) if v else (None, None)


def sanity_report(scenario="A"):
    inv = _load_inventory()
    if inv.empty:
        print(f"Inventory not found: {INVENTORY}")
        return
    print(f"Inventory: {INVENTORY}  ({len(inv)} rows)")
    for tp in ["3m", "4m", "6m", "7m", "9m", "12m"]:
        recs = list(iter_recordings(tp, scenario))
        wt = sum(1 for _, _, g in recs if g == "WT")
        ko = sum(1 for _, _, g in recs if g == "KO")
        n_animals = len(set(a for _, a, _ in recs))
        print(f"  {tp:>3}: {len(recs):>4} recordings  (WT={wt}, KO={ko})  "
              f"{n_animals} unique animals")


if __name__ == "__main__":
    sanity_report()
