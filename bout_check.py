import pandas as pd
from scipy.stats import mannwhitneyu

b = pd.read_csv('data/video_bouts_4m.csv')
print("=== BOUT DURATION ===")
for g in ['WT','KO']:
    sub = b[b.group==g]['duration_s']
    print(f"{g}: n={len(sub)} bouts | mean={sub.mean():.2f}s | median={sub.median():.2f}s | max={sub.max():.2f}s | total={sub.sum()/3600:.2f}h")

_, p = mannwhitneyu(b[b.group=='WT']['duration_s'], b[b.group=='KO']['duration_s'])
print(f"Duration WT vs KO: p={p:.5f}")

_, p2 = mannwhitneyu(b[b.group=='WT']['peak_score'], b[b.group=='KO']['peak_score'])
print(f"Peak score WT vs KO: p={p2:.5f}")

print("\n=== SEIZURE-LIKE EVENTS ===")
s = pd.read_csv('data/video_seizures_4m.csv')
for g in ['WT','KO']:
    sub = s[s.group==g]
    print(f"{g}: n={len(sub)} events | mean_dur={sub.duration_s.mean():.2f}s | mean_peak={sub.peak_score.mean():.4f}")
_, p3 = mannwhitneyu(s[s.group=='WT']['duration_s'], s[s.group=='KO']['duration_s'])
print(f"Seizure duration WT vs KO: p={p3:.5f}")
