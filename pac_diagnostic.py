import pandas as pd
import numpy as np

# Check 1: How many REM epochs per animal at 3m?
print("=== REM EPOCH COUNTS AT 3m ===")
ep = pd.read_csv("data/epochs_with_states_3m.csv")
ep["animal_id"] = ep["animal_id"].astype(str)
rem = ep[ep.state=="REM"]
print(f"Total REM epochs: {len(rem):,} out of {len(ep):,} ({len(rem)/len(ep)*100:.1f}%)")
print(f"Mean REM epochs per animal: {rem.groupby('animal_id').size().mean():.0f}")
print(f"Min REM epochs per animal:  {rem.groupby('animal_id').size().min():.0f}")
print(f"Max REM epochs per animal:  {rem.groupby('animal_id').size().max():.0f}")
print()

# Check 2: What values did script 14 produce?
print("=== STATE-SPECIFIC PAC VALUES (3m) ===")
pac = pd.read_csv("results/pac_state_specific_3m.csv")
pac["animal_id"] = pac["animal_id"].astype(str)

# Check which columns exist
ctx_cols = [c for c in pac.columns if c.startswith("ctx_")]
ca3_cols = [c for c in pac.columns if c.startswith("ca3_")]
print(f"CTX columns: {ctx_cols}")
print(f"CA3 columns: {ca3_cols}")
print()

# Check n_epochs per state
epoch_cols = [c for c in pac.columns if "n_epochs" in c]
print(f"Epoch count columns: {epoch_cols}")
if epoch_cols:
    print(pac.groupby("group")[epoch_cols].mean().to_string())
print()

# Check T_HG values specifically
thg_cols = [c for c in pac.columns if "T_HG" in c or "t_hg" in c.lower()]
print(f"High gamma PAC columns: {thg_cols}")
if thg_cols:
    am = pac.groupby(["animal_id","group"])[thg_cols].mean().reset_index()
    print("\nAnimal-level T_HG means:")
    print(am[["group"]+thg_cols].groupby("group").mean().to_string())
    print("\nWT values:")
    print(am[am.group=="WT"][thg_cols].to_string())
    print("\nKO values:")
    print(am[am.group=="KO"][thg_cols].to_string())
print()

# Check if ctx columns have valid data
print("=== CTX DATA AVAILABILITY ===")
for col in ctx_cols:
    n_valid = pac[col].notna().sum()
    n_total = len(pac)
    print(f"  {col}: {n_valid}/{n_total} valid ({n_valid/n_total*100:.0f}%)")
    if n_valid > 0:
        print(f"    mean={pac[col].mean():.6f} std={pac[col].std():.6f}")
