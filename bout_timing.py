import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

b = pd.read_csv('data/video_bouts_4m.csv')

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for ax, g, color in zip(axes, ['WT', 'KO'], ['#378ADD', '#D85A30']):
    sub = b[b.group == g]
    ax.hist(sub['onset_s'] / 3600, bins=48, color=color, alpha=0.7)
    ax.set_ylabel(f'{g} bout count', fontsize=10)
    ax.set_title(f'{g}: {len(sub)} total bouts', fontsize=10)

axes[1].set_xlabel('Time (hours)', fontsize=10)
fig.suptitle('Movement bout timing — 4m KA day', fontsize=11)
plt.tight_layout()
fig.savefig('figures/video_bout_timing_4m.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved: figures/video_bout_timing_4m.png')

# Also print hourly breakdown
print('\nHourly bout counts:')
print(f'{"Hour":<6} {"WT":>8} {"KO":>8} {"WT/KO ratio":>12}')
for h in range(24):
    wt = ((b.group=='WT') & (b.onset_s >= h*3600) & (b.onset_s < (h+1)*3600)).sum()
    ko = ((b.group=='KO') & (b.onset_s >= h*3600) & (b.onset_s < (h+1)*3600)).sum()
    ratio = wt/ko if ko > 0 else float('inf')
    print(f'{h:<6} {wt:>8} {ko:>8} {ratio:>12.1f}')
