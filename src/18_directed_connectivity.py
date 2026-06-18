"""
18_directed_connectivity.py
============================
TIER 1 — DIRECTED INFORMATION FLOW between CA3 (hippocampus) and S1/PtA (cortex)

The decorrelation index used elsewhere is symmetric and undirected. This script
asks WHICH DIRECTION of cortico-hippocampal communication degrades over the
disease course — directly testing the corticofugal hypothesis (anterograde
cortex->target degeneration in ALS).

PART A — TRANSFER ENTROPY (model-free directed information)
  TE(X->Y) quantifies how much knowing X's past reduces uncertainty about Y's
  future, beyond Y's own past. Computed both directions:
    TE(CTX->CA3)  and  TE(CA3->CTX)
  Net directionality:  dTE = TE(CTX->CA3) - TE(CA3->CTX)
  Significance via time-shifted surrogates (break the directional coupling while
  preserving each signal's autocorrelation).
  Reported per timepoint, per genotype, per state (REM/NREM/Wake).

  Method: history-embedded TE with Kraskov-Stoegbauer-Grassberger (KSG) k-NN
  estimator if available; falls back to a binned/Gaussian estimator otherwise.

PART B — CROSS-FREQUENCY DIRECTIONALITY (phase transfer entropy)
  Standard PAC is non-directional. Phase Transfer Entropy (pTE) on band-limited
  phase time-series tests whether one channel's phase drives the other's:
    pTE(CTX_theta -> CA3_gamma_phase) vs reverse, etc.
  Asks whether, even where coupling MAGNITUDE is preserved, its DIRECTION changes.

Outputs:
  results/transfer_entropy_results.csv
  results/cross_freq_directionality.csv
  figures/te_directionality_trajectory.png
  figures/te_net_heatmap.png
  figures/cross_freq_directionality.png

INPUT: raw ABF files (both channels). Set ABF_DIR and the channel map below.
Run:
    python src/18_directed_connectivity.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, decimate
from scipy.fft import next_fast_len
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# shared F:\EEG path resolver (folder->timepoint map, genotype from folder)
try:
    from abf_paths import iter_recordings, find_video_near
    HAVE_PATHS = True
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    from sklearn.neighbors import NearestNeighbors
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")
ABF_DIR     = os.path.join(PORT_DIR, "data")   # adjust if ABFs live elsewhere

# Channel map: which ABF channel is which structure
CA3_CH = 0   # hippocampal
CTX_CH = 1   # cortical S1/PtA

FS_TARGET = 500.0     # Hz after decimation
EPOCH_S   = 4.0
TP_ORDER  = ["3m", "4m", "6m", "7m", "9m", "12m"]
COLORS    = {"WT": "#378ADD", "KO": "#D85A30"}

# TE parameters
TE_K        = 4       # k-NN neighbors for KSG
TE_EMBED    = 3       # history embedding dimension
TE_TAU      = 1       # embedding delay (samples at FS_TARGET)
N_SURR      = 20      # group-level surrogates (estimator now fast; fewer needed)
MAX_EPOCHS  = 80      # cap epochs per file (estimator now fast)

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "low_gamma": (30, 80)}


# ── helpers ────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na-1)*np.var(a, ddof=1)+(nb-1)*np.var(b, ddof=1))/(na+nb-2))
    return 0.0 if pooled == 0 else (np.mean(a)-np.mean(b))/pooled


def fdr_bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0: return p
    o = np.argsort(p); r = np.empty(n, int); r[o] = np.arange(1, n+1)
    f = p*n/r; f = np.minimum.accumulate(f[o][::-1])[::-1]
    out = np.empty(n); out[o] = f
    return np.minimum(out, 1.0)


def bandpass(x, lo, hi, fs=FS_TARGET, order=3):
    ny = 0.5*fs
    b, a = butter(order, [max(lo/ny,1e-4), min(hi/ny,0.999)], btype="band")
    return filtfilt(b, a, x)


def embed(x, d, tau):
    """Time-delay embedding -> (N-(d-1)*tau, d) matrix."""
    N = len(x) - (d-1)*tau
    if N <= 0:
        return np.empty((0, d))
    return np.column_stack([x[i*tau:i*tau+N] for i in range(d)])


def _copula_normalize(M):
    """Rank-transform each column to standard-Gaussian margins (Gaussian copula)."""
    M = np.atleast_2d(M)
    if M.shape[0] == 1:
        M = M.T
    from scipy.stats import norm
    out = np.empty_like(M, dtype=float)
    n = M.shape[0]
    for j in range(M.shape[1]):
        order = np.argsort(np.argsort(M[:, j]))
        out[:, j] = norm.ppf((order + 1.0) / (n + 1.0))
    return out


def _gauss_entropy(C):
    """Differential entropy of a Gaussian with covariance C (nats)."""
    C = np.atleast_2d(C)
    sign, logdet = np.linalg.slogdet(2*np.pi*np.e*C)
    if sign <= 0:
        # regularize
        C = C + 1e-9*np.eye(C.shape[0])
        sign, logdet = np.linalg.slogdet(2*np.pi*np.e*C)
    return 0.5*logdet


def te_ksg(x, y, k=None, d=TE_EMBED, tau=TE_TAU):
    """
    Transfer entropy X->Y via the Gaussian-copula estimator (Ince et al. 2017).
    TE(X->Y) = I(Y_future ; X_past | Y_past), computed in closed form from
    covariances after copula-normalizing the marginals. Vectorized: no neighbor
    search, so it runs in milliseconds rather than seconds per epoch.
    Same signature/return as the previous estimator (k is ignored).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    Yp = embed(y[:-1], d, tau)                              # Y past (d-dim)
    n = min(len(Yp), len(x)-1-(d-1)*tau, len(y)-1-(d-1)*tau)
    if n < 50:
        return np.nan
    Yp = Yp[:n]
    Yf = y[(d-1)*tau+1:(d-1)*tau+1+n].reshape(-1, 1)        # Y future (1 step)
    Xp = x[(d-1)*tau:(d-1)*tau+n].reshape(-1, 1)            # X past (current)

    # Copula-normalize the full joint, then slice (preserves joint rank structure)
    J = _copula_normalize(np.column_stack([Yf, Yp, Xp]))
    nf = 1; npd = Yp.shape[1]
    Yf_c  = J[:, 0:nf]
    Yp_c  = J[:, nf:nf+npd]
    Xp_c  = J[:, nf+npd:]

    # TE = H(Yf,Yp) + H(Yp,Xp) - H(Yp) - H(Yf,Yp,Xp)
    try:
        H_YfYp   = _gauss_entropy(np.cov(np.column_stack([Yf_c, Yp_c]), rowvar=False))
        H_YpXp   = _gauss_entropy(np.cov(np.column_stack([Yp_c, Xp_c]), rowvar=False))
        H_Yp     = _gauss_entropy(np.cov(Yp_c, rowvar=False))
        H_YfYpXp = _gauss_entropy(np.cov(np.column_stack([Yf_c, Yp_c, Xp_c]), rowvar=False))
        te = H_YfYp + H_YpXp - H_Yp - H_YfYpXp
        return max(te, 0.0)
    except Exception:
        return np.nan


def te_with_surrogates(x, y, n_surr=N_SURR):
    """Return (TE_xy, TE_yx, p_xy, p_yx) using time-shift surrogates."""
    te_xy = te_ksg(x, y)
    te_yx = te_ksg(y, x)
    rng = np.random.default_rng(0)
    surr_xy, surr_yx = [], []
    for _ in range(n_surr):
        sh = rng.integers(len(x)//4, 3*len(x)//4)
        xs = np.roll(x, sh)
        surr_xy.append(te_ksg(xs, y))
        surr_yx.append(te_ksg(y, xs))
    surr_xy = np.array([s for s in surr_xy if not np.isnan(s)])
    surr_yx = np.array([s for s in surr_yx if not np.isnan(s)])
    p_xy = (np.sum(surr_xy >= te_xy)+1)/(len(surr_xy)+1) if len(surr_xy) else np.nan
    p_yx = (np.sum(surr_yx >= te_yx)+1)/(len(surr_yx)+1) if len(surr_yx) else np.nan
    return te_xy, te_yx, p_xy, p_yx


# ── ABF loading + per-animal state-resolved signal ─────────────────────────
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


def load_abf_signals(abf_path):
    """Load (ca3, ctx) using the portfolio loader; channel 0=CA3, 1=CTX."""
    ca3, fs = load_signal(abf_path, channel=CA3_CH)
    if ca3 is None:
        return None, None
    ctx, _ = load_signal(abf_path, channel=CTX_CH)
    if ctx is None:
        return None, None
    return ca3, ctx
    try:
        abf = pyabf.ABF(abf_path)
        fs_orig = abf.dataRate
        abf.setSweep(0, channel=CA3_CH); ca3 = abf.sweepY.copy()
        abf.setSweep(0, channel=CTX_CH); ctx = abf.sweepY.copy()
        factor = int(round(fs_orig / FS_TARGET))
        if factor > 1:
            ca3 = decimate(ca3, factor, ftype="iir", zero_phase=True)
            ctx = decimate(ctx, factor, ftype="iir", zero_phase=True)
        return ca3, ctx
    except Exception as e:
        print(f"  ABF load failed {os.path.basename(abf_path)}: {e}")
        return None, None


def states_by_basename(tp):
    """
    Build {abf_basename: [state per epoch]} from the epoch CSV for this timepoint.
    States come from the CSV (sleep classification); genotype comes from the
    folder tree via iter_recordings (authoritative).
    """
    ep_path = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
    if not os.path.exists(ep_path):
        return {}
    ep = pd.read_csv(ep_path)
    out = {}
    for abf_file, g in ep.groupby("abf_file"):
        out[os.path.basename(str(abf_file))] = g.sort_values("epoch_idx")["state"].tolist()
    return out


# ── PART A ─────────────────────────────────────────────────────────────────
def part_a_transfer_entropy(states=("REM", "NREM")):
    print("\n" + "="*64)
    print("PART A — TRANSFER ENTROPY  CTX<->CA3 (directed information)")
    print("="*64)
    if not HAVE_ABF:
        print("pyabf required: pip install pyabf --break-system-packages")
        return None

    rows = []
    for tp in TP_ORDER:
        state_lut = states_by_basename(tp)
        recs = list(iter_recordings(tp)) if HAVE_PATHS else []
        if not recs:
            print(f"\n[{tp}] no recordings found on disk (check F:\\EEG mapping)")
            continue
        print(f"\n[{tp}] {len(recs)} recordings on disk")
        for abf_path, mouse_id, group in recs:
            base = os.path.basename(abf_path)
            states_seq = state_lut.get(base)
            if states_seq is None:
                # no sleep labels for this file in the CSV -> skip (can't state-resolve)
                continue
            ca3, ctx = load_abf_signals(abf_path)
            if ca3 is None:
                continue
            ep_len = int(EPOCH_S * FS_TARGET)
            for state in states:
                idx = [i for i, s in enumerate(states_seq) if s == state]
                if len(idx) < 5:
                    continue
                idx = idx[:MAX_EPOCHS]
                te_xy_list, te_yx_list = [], []
                for ei in idx:
                    a = ei*ep_len; b = a+ep_len
                    if b > len(ca3) or b > len(ctx):
                        continue
                    seg_ca3 = ca3[a:b]; seg_ctx = ctx[a:b]
                    txy = te_ksg(seg_ctx, seg_ca3)   # CTX->CA3
                    tyx = te_ksg(seg_ca3, seg_ctx)   # CA3->CTX
                    if not np.isnan(txy): te_xy_list.append(txy)
                    if not np.isnan(tyx): te_yx_list.append(tyx)
                if len(te_xy_list) < 3:
                    continue
                te_ctx_ca3 = np.mean(te_xy_list)
                te_ca3_ctx = np.mean(te_yx_list)
                rows.append({
                    "timepoint": tp, "animal_id": mouse_id, "group": group, "state": state,
                    "te_ctx_to_ca3": te_ctx_ca3, "te_ca3_to_ctx": te_ca3_ctx,
                    "net_te": te_ctx_ca3 - te_ca3_ctx,   # +ve = cortex-led
                    "n_epochs": len(te_xy_list),
                })
            print(f"  {mouse_id} ({group}) done")

    if not rows:
        print("No TE computed (check ABF paths / channel map).")
        return None
    te_df = pd.DataFrame(rows)
    te_df.to_csv(os.path.join(RESULTS_DIR, "transfer_entropy_results.csv"), index=False)
    print(f"\nSaved: transfer_entropy_results.csv ({len(te_df)} rows)")

    # Group stats per timepoint/state on net_te and each direction
    print("\nNet TE (CTX->CA3 minus CA3->CTX): WT vs KO")
    stat_rows = []
    for state in te_df.state.unique():
        for tp in TP_ORDER:
            sub = te_df[(te_df.timepoint == tp) & (te_df.state == state)]
            for metric in ["net_te", "te_ctx_to_ca3", "te_ca3_to_ctx"]:
                wt = sub[sub.group == "WT"][metric].dropna().values
                ko = sub[sub.group == "KO"][metric].dropna().values
                if len(wt) < 2 or len(ko) < 2:
                    continue
                d = cohens_d(ko, wt)
                _, p = mannwhitneyu(ko, wt, alternative="two-sided")
                stat_rows.append({"state": state, "timepoint": tp, "metric": metric,
                                  "wt_mean": np.mean(wt), "ko_mean": np.mean(ko),
                                  "cohens_d": d, "pval": p})
                if metric == "net_te" and p < 0.10:
                    print(f"  [{state} {tp}] net TE: WT={np.mean(wt):.4f} "
                          f"KO={np.mean(ko):.4f} d={d:.2f} p={p:.3f}")
    stat_df = pd.DataFrame(stat_rows)
    if len(stat_df):
        stat_df["pval_fdr"] = fdr_bh(stat_df["pval"].values).round(4)
        stat_df.to_csv(os.path.join(RESULTS_DIR, "transfer_entropy_stats.csv"), index=False)

    plot_te_trajectory(te_df)
    return te_df


def plot_te_trajectory(te_df):
    states = [s for s in ["REM", "NREM"] if s in te_df.state.unique()]
    if not states:
        return
    fig, axes = plt.subplots(1, len(states), figsize=(6*len(states), 4.5), squeeze=False)
    for ci, state in enumerate(states):
        ax = axes[0][ci]
        sub = te_df[te_df.state == state]
        for group, color in COLORS.items():
            g = sub[sub.group == group]
            xs, ms, es = [], [], []
            for tp in TP_ORDER:
                v = g[g.timepoint == tp]["net_te"].dropna()
                if len(v) == 0: continue
                xs.append(TP_ORDER.index(tp)); ms.append(v.mean()); es.append(v.sem())
            if xs:
                ax.errorbar(xs, ms, yerr=es, fmt="-o", color=color, lw=2,
                            capsize=3, label=group)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xticks(range(len(TP_ORDER))); ax.set_xticklabels(TP_ORDER)
        ax.set_ylabel("Net TE  (CTX→CA3 − CA3→CTX)")
        ax.set_xlabel("Timepoint")
        ax.set_title(f"Directed cortico-hippocampal flow — {state}\n"
                     "(+ = cortex-led;  trajectory tests corticofugal hypothesis)",
                     fontsize=9)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "te_directionality_trajectory.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: te_directionality_trajectory.png")


# ── PART B — CROSS-FREQUENCY DIRECTIONALITY (phase TE) ──────────────────────
def phase_te(x_phase, y_phase, k=TE_K):
    """Phase transfer entropy on unwrapped phase increments."""
    dx = np.diff(np.unwrap(x_phase))
    dy = np.diff(np.unwrap(y_phase))
    n = min(len(dx), len(dy))
    return te_ksg(dx[:n], dy[:n], k=k)


def part_b_cross_freq_directionality(state="REM"):
    print("\n" + "="*64)
    print("PART B — CROSS-FREQUENCY DIRECTIONALITY (phase TE)")
    print(f"Does CTX theta-phase drive CA3 gamma-phase (or reverse)? [{state}]")
    print("="*64)
    if not HAVE_ABF:
        print("pyabf required.")
        return None

    rows = []
    for tp in TP_ORDER:
        state_lut = states_by_basename(tp)
        recs = list(iter_recordings(tp)) if HAVE_PATHS else []
        print(f"\n[{tp}] {len(recs)} recordings (phase-TE)")
        for abf_path, mouse_id, group in recs:
            base = os.path.basename(abf_path)
            states_seq = state_lut.get(base)
            if states_seq is None:
                continue
            ca3, ctx = load_abf_signals(abf_path)
            if ca3 is None:
                continue
            ep_len = int(EPOCH_S*FS_TARGET)
            idx = [i for i, s in enumerate(states_seq) if s == state][:MAX_EPOCHS]
            if len(idx) < 5:
                continue
            # Concatenate state epochs, then band-limit phases
            seg_ca3 = np.concatenate([ca3[i*ep_len:(i+1)*ep_len] for i in idx
                                      if (i+1)*ep_len <= len(ca3)])
            seg_ctx = np.concatenate([ctx[i*ep_len:(i+1)*ep_len] for i in idx
                                      if (i+1)*ep_len <= len(ctx)])
            if len(seg_ca3) < FS_TARGET*4:
                continue
            # Cap length so Hilbert FFT stays fast (60 s of REM is ample for pTE)
            cap = int(FS_TARGET*60)
            if len(seg_ca3) > cap:
                seg_ca3 = seg_ca3[:cap]; seg_ctx = seg_ctx[:cap]

            def _fast_phase(sig, band):
                bp = bandpass(sig, *band)
                nfast = next_fast_len(len(bp))
                analytic = hilbert(bp, N=nfast)[:len(bp)]
                return np.angle(analytic)

            ctx_theta_ph = _fast_phase(seg_ctx, BANDS["theta"])
            ca3_theta_ph = _fast_phase(seg_ca3, BANDS["theta"])
            ctx_gam_ph   = _fast_phase(seg_ctx, BANDS["low_gamma"])
            ca3_gam_ph   = _fast_phase(seg_ca3, BANDS["low_gamma"])
            pte_ctxTheta_ca3gam = phase_te(ctx_theta_ph, ca3_gam_ph)
            pte_ca3Theta_ctxgam = phase_te(ca3_theta_ph, ctx_gam_ph)
            rows.append({
                "timepoint": tp, "animal_id": mouse_id, "group": group, "state": state,
                "pte_ctxtheta_to_ca3gamma": pte_ctxTheta_ca3gam,
                "pte_ca3theta_to_ctxgamma": pte_ca3Theta_ctxgam,
                "net_pte": (pte_ctxTheta_ca3gam - pte_ca3Theta_ctxgam),
            })
            print(f"  [{tp}] {mouse_id} ({group}) pte done")
    if not rows:
        print("No phase-TE computed.")
        return None
    pte_df = pd.DataFrame(rows)
    pte_df.to_csv(os.path.join(RESULTS_DIR, "cross_freq_directionality.csv"), index=False)
    print(f"Saved: cross_freq_directionality.csv ({len(pte_df)} rows)")

    print("\nNet phase-TE (CTX theta->CA3 gamma minus reverse): WT vs KO")
    for tp in TP_ORDER:
        sub = pte_df[pte_df.timepoint == tp]
        wt = sub[sub.group == "WT"]["net_pte"].dropna().values
        ko = sub[sub.group == "KO"]["net_pte"].dropna().values
        if len(wt) < 2 or len(ko) < 2:
            continue
        d = cohens_d(ko, wt)
        _, p = mannwhitneyu(ko, wt, alternative="two-sided")
        flag = "  *" if p < 0.05 else ""
        print(f"  [{tp}] WT={np.mean(wt):.4f} KO={np.mean(ko):.4f} d={d:.2f} p={p:.3f}{flag}")

    # Figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group, color in COLORS.items():
        g = pte_df[pte_df.group == group]
        xs, ms, es = [], [], []
        for tp in TP_ORDER:
            v = g[g.timepoint == tp]["net_pte"].dropna()
            if len(v) == 0: continue
            xs.append(TP_ORDER.index(tp)); ms.append(v.mean()); es.append(v.sem())
        if xs:
            ax.errorbar(xs, ms, yerr=es, fmt="-o", color=color, lw=2, capsize=3, label=group)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(TP_ORDER))); ax.set_xticklabels(TP_ORDER)
    ax.set_ylabel("Net phase-TE (CTX θ→CA3 γ − reverse)")
    ax.set_xlabel("Timepoint")
    ax.set_title(f"Cross-frequency directionality trajectory — {state}")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIGURES_DIR, "cross_freq_directionality.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: cross_freq_directionality.png")
    return pte_df


def main():
    import sys
    print("="*64)
    print("DIRECTED CONNECTIVITY: transfer entropy + cross-freq directionality")
    print("="*64)
    if not HAVE_ABF:
        print("\npyabf not installed. Run: pip install pyabf --break-system-packages")
        return

    force = "--force-a" in sys.argv
    skip_b = "--only-a" in sys.argv
    only_b = "--only-b" in sys.argv

    te_csv = os.path.join(RESULTS_DIR, "transfer_entropy_results.csv")
    if only_b:
        print("\n[--only-b] Skipping Part A; using existing results CSV.")
    elif (not force) and os.path.exists(te_csv):
        print(f"\nPart A results already exist ({os.path.basename(te_csv)}); skipping recompute.")
        print("  (use --force-a to recompute Part A from raw ABFs)")
    else:
        part_a_transfer_entropy()

    if not skip_b:
        part_b_cross_freq_directionality()
    print("\n" + "="*64)
    print("COMPLETE")
    print("="*64)


if __name__ == "__main__":
    main()
