"""
08_advanced_eeg_analyses.py
============================
Advanced EEG feature extraction beyond basic band power.
Designed to be run per-timepoint after script 05.

Analyses:
  1. Sleep spindle detection (sigma 12-15 Hz during NREM)
  2. Sharp wave ripple detection (80-120 Hz bursts during NREM)
  3. Phase-amplitude coupling (theta phase × gamma amplitude)
  4. Inter-channel coherence (CA3-CTX connectivity)
  5. Granger causality direction (CA3 → CTX vs CTX → CA3)
  6. Microstate analysis (dominant oscillation state within epochs)
  7. Fractal dimension (Higuchi's method)
  8. Detrended fluctuation analysis (DFA — long-range correlations)
  9. Burst suppression ratio (late-stage neurodegeneration marker)
 10. Sleep slow oscillation detection (<1 Hz, NREM)

Outputs per timepoint:
  - results/advanced_eeg_{tp}.csv
  - figures/advanced_*.png

Run:
    python src/08_advanced_eeg_analyses.py [--timepoint 3m]
"""

import os
import gc
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import pyabf
from scipy.signal import (welch, butter, filtfilt, hilbert, decimate,
                           find_peaks, sosfiltfilt, sosfilt_zi)
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")

COLORS   = {"WT": "#378ADD", "KO": "#D85A30"}
TP_ORDER = ["3m", "4m", "6m", "7m", "9m", "12m"]
EPOCH_S  = 4.0
FS_DS    = 500


# ── Signal loading ─────────────────────────────────────────────────────────

def load_two_channel(abf_path, target_fs=FS_DS):
    """Load both CA3 (ch0) and CTX (ch1) channels."""
    try:
        abf = pyabf.ABF(abf_path)
        fs  = float(abf.dataRate)
        signals = {}
        for ch_idx, ch_name in enumerate(["CA3", "CTX"]):
            if ch_idx >= abf.channelCount:
                break
            abf.setSweep(0, channel=ch_idx)
            sig = abf.sweepY.copy().astype(np.float64)
            factor = max(1, int(round(fs / target_fs)))
            if factor > 1:
                sig = decimate(sig - sig.mean(), factor, zero_phase=True)
            signals[ch_name] = sig
        del abf; gc.collect()
        return signals, float(target_fs)
    except Exception:
        return {}, None


def bandpass(sig, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


# ── 1. Sleep spindle detection ─────────────────────────────────────────────

def detect_sleep_spindles(sig, fs, state_epochs=None):
    """
    Detect sleep spindles in sigma band (12-15 Hz).
    Method: RMS envelope of sigma-filtered signal exceeds threshold.

    Parameters:
        sig          : continuous EEG signal
        fs           : sampling rate
        state_epochs : DataFrame with epoch-level state labels (optional)

    Returns DataFrame with spindle events.
    """
    sigma = bandpass(sig, fs, 12, 15, order=4)
    rms   = _rms_envelope(sigma, fs, window_s=0.25)
    threshold = np.mean(rms) + 1.5 * np.std(rms)

    above = rms > threshold
    padded = np.concatenate([[0], above.astype(int), [0]])
    onsets  = np.where(np.diff(padded) ==  1)[0]
    offsets = np.where(np.diff(padded) == -1)[0]

    spindles = []
    for on, off in zip(onsets, offsets):
        dur_s = (off - on) / fs
        if 0.4 <= dur_s <= 3.0:   # typical spindle: 0.4–3 s
            peak_freq = _dominant_freq(sigma[on:off], fs, 11, 16)
            amp       = float(np.max(np.abs(sigma[on:off])))
            spindles.append({
                "onset_s":   on / fs,
                "offset_s":  off / fs,
                "duration_s": dur_s,
                "peak_freq_hz": peak_freq,
                "amplitude":    amp,
            })

    return pd.DataFrame(spindles)


def _rms_envelope(sig, fs, window_s=0.25):
    w = max(1, int(window_s * fs))
    sq = sig**2
    kernel = np.ones(w) / w
    return np.sqrt(np.convolve(sq, kernel, mode="same"))


def _dominant_freq(sig, fs, lo, hi):
    """Return frequency of peak PSD in [lo, hi] Hz."""
    f, psd = welch(sig, fs=fs, nperseg=min(len(sig), int(fs*2)))
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return np.nan
    return float(f[m][np.argmax(psd[m])])


# ── 2. Sharp wave ripple detection ─────────────────────────────────────────

def detect_ripples(sig_ca3, fs, band=(80, 120)):
    """
    Detect hippocampal sharp wave ripples (80-120 Hz bursts).
    Returns DataFrame with ripple events.
    """
    ripple_sig = bandpass(sig_ca3, fs, band[0], band[1])
    env = np.abs(hilbert(ripple_sig))
    # Smooth envelope
    w = max(1, int(0.01 * fs))  # 10 ms window
    env = np.convolve(env, np.ones(w)/w, mode="same")

    threshold = np.mean(env) + 3 * np.std(env)
    above     = env > threshold

    padded  = np.concatenate([[0], above.astype(int), [0]])
    onsets  = np.where(np.diff(padded) ==  1)[0]
    offsets = np.where(np.diff(padded) == -1)[0]

    ripples = []
    for on, off in zip(onsets, offsets):
        dur_ms = (off - on) / fs * 1000
        if 30 <= dur_ms <= 200:   # typical ripple: 30–200 ms
            ripples.append({
                "onset_s":    on / fs,
                "duration_ms": dur_ms,
                "amplitude":   float(env[on:off].max()),
                "freq_hz":     _dominant_freq(ripple_sig[on:off], fs, 80, 120),
            })

    return pd.DataFrame(ripples)


# ── 3. Phase-Amplitude Coupling ────────────────────────────────────────────

def compute_pac(sig, fs, phase_band=(4, 8), amp_band=(30, 80), n_bins=18):
    """
    Phase-amplitude coupling using Modulation Index (MI).
    Theta phase × gamma amplitude coupling.
    Returns MI value (higher = stronger coupling).
    """
    phase_sig = bandpass(sig, fs, phase_band[0], phase_band[1])
    amp_sig   = bandpass(sig, fs, amp_band[0],   amp_band[1])

    phase = np.angle(hilbert(phase_sig))
    amp   = np.abs(hilbert(amp_sig))

    # Bin amplitude by phase
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    amp_by_phase = np.zeros(n_bins)
    for k in range(n_bins):
        mask = (phase >= bin_edges[k]) & (phase < bin_edges[k+1])
        if mask.sum() > 0:
            amp_by_phase[k] = amp[mask].mean()

    # Modulation Index
    amp_by_phase /= (amp_by_phase.sum() + 1e-12)
    MI = np.sum(amp_by_phase * np.log(amp_by_phase / (1/n_bins) + 1e-12)) / np.log(n_bins)
    return float(MI), amp_by_phase


def compute_pac_epochs(sig, fs, epoch_s=EPOCH_S,
                        phase_band=(4, 8), amp_band=(30, 80)):
    """
    Compute PAC for each epoch. Returns array of MI values per epoch.
    """
    epoch_n  = int(epoch_s * fs)
    n_epochs = len(sig) // epoch_n
    mi_vals  = []
    for i in range(n_epochs):
        ep = sig[i*epoch_n:(i+1)*epoch_n]
        if np.std(ep) < 0.001:
            mi_vals.append(np.nan)
            continue
        mi, _ = compute_pac(ep, fs, phase_band, amp_band)
        mi_vals.append(mi)
    return np.array(mi_vals)


# ── 4. Inter-channel coherence ─────────────────────────────────────────────

def compute_coherence_epochs(sig1, sig2, fs, epoch_s=EPOCH_S):
    """
    Compute magnitude-squared coherence between CA3 and CTX per epoch.
    Returns DataFrame with band coherence values per epoch.
    """
    epoch_n  = int(epoch_s * fs)
    n_epochs = min(len(sig1), len(sig2)) // epoch_n
    rows = []

    for i in range(n_epochs):
        ep1 = sig1[i*epoch_n:(i+1)*epoch_n]
        ep2 = sig2[i*epoch_n:(i+1)*epoch_n]
        if np.std(ep1) < 0.001 or np.std(ep2) < 0.001:
            continue

        from scipy.signal import coherence
        f, Cxy = coherence(ep1, ep2, fs=fs, nperseg=int(fs*2))

        def band_coh(lo, hi):
            m = (f >= lo) & (f <= hi)
            return float(Cxy[m].mean()) if m.any() else np.nan

        rows.append({
            "epoch_idx":     i,
            "coh_delta":     band_coh(0.5, 4),
            "coh_theta":     band_coh(4,   8),
            "coh_alpha":     band_coh(8,  13),
            "coh_beta":      band_coh(13, 30),
            "coh_gamma":     band_coh(30, 80),
        })

    return pd.DataFrame(rows)


# ── 5. Higuchi Fractal Dimension ───────────────────────────────────────────

def higuchi_fd(sig, kmax=10):
    """
    Higuchi's fractal dimension. Higher = more complex / irregular.
    """
    n = len(sig)
    L = []
    x = np.asarray(sig, dtype=np.float64)

    for k in range(1, kmax + 1):
        Lk = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            Lm = np.sum(np.abs(np.diff(x[indices]))) * (n - 1) / (k * (len(indices) - 1) * k)
            Lk.append(Lm)
        if Lk:
            L.append(np.mean(Lk))

    if len(L) < 2:
        return np.nan
    ln_k   = np.log(np.arange(1, len(L) + 1))
    ln_L   = np.log(np.array(L) + 1e-12)
    slope, _ = np.polyfit(ln_k, ln_L, 1)
    return float(-slope)


# ── 6. DFA — long-range temporal correlations ──────────────────────────────

def dfa_alpha(sig, scales=None):
    """
    Detrended Fluctuation Analysis. Returns scaling exponent α.
    α ~ 0.5 = uncorrelated noise; α > 0.5 = long-range correlations.
    """
    if scales is None:
        n = len(sig)
        scales = np.unique(np.logspace(np.log10(10), np.log10(n//4), 20).astype(int))
        scales = scales[scales >= 4]

    x     = np.cumsum(sig - np.mean(sig))
    flucs = []
    for s in scales:
        n_segs = len(x) // s
        if n_segs < 2:
            flucs.append(np.nan); continue
        F2 = []
        for j in range(n_segs):
            seg = x[j*s:(j+1)*s]
            t   = np.arange(s)
            c   = np.polyfit(t, seg, 1)
            trend = np.polyval(c, t)
            F2.append(np.mean((seg - trend)**2))
        flucs.append(np.sqrt(np.mean(F2)))

    flucs  = np.array(flucs, dtype=float)
    valid  = ~np.isnan(flucs) & (flucs > 0)
    if valid.sum() < 4:
        return np.nan
    slope, _ = np.polyfit(np.log10(scales[valid]), np.log10(flucs[valid]), 1)
    return float(slope)


# ── 7. Burst suppression ratio ─────────────────────────────────────────────

def burst_suppression_ratio(sig, fs, suppress_thresh=5.0, window_s=1.0):
    """
    Fraction of time with suppressed EEG (amplitude < suppress_thresh µV).
    Elevated BSR indicates severe neural network collapse.
    """
    env = _rms_envelope(np.abs(sig), fs, window_s=window_s)
    return float(np.mean(env < suppress_thresh))


# ── Main: per-file advanced analysis ──────────────────────────────────────

def run_advanced_for_file(abf_path, animal_id, session_id, group, abf_file, tp,
                           eeg_epoch_df=None):
    """
    Run all advanced analyses on one ABF file.
    Returns dict of summary metrics.
    """
    signals, fs = load_two_channel(abf_path)
    if not signals or fs is None:
        return None

    sig_ca3 = signals.get("CA3")
    sig_ctx = signals.get("CTX")
    if sig_ca3 is None:
        return None

    results = {
        "animal_id":  animal_id,
        "session_id": session_id,
        "group":      group,
        "abf_file":   abf_file,
        "timepoint":  tp,
    }

    # ── Spindles ──────────────────────────────────────────────────────────
    try:
        spindles = detect_sleep_spindles(sig_ca3, fs)
        duration_h = len(sig_ca3) / fs / 3600
        results["spindle_rate_per_h"]   = len(spindles) / max(duration_h, 0.001)
        results["spindle_duration_mean"] = spindles["duration_s"].mean() if len(spindles) > 0 else 0
        results["spindle_amplitude_mean"] = spindles["amplitude"].mean() if len(spindles) > 0 else 0
        results["n_spindles"]            = len(spindles)
    except Exception as e:
        results["spindle_rate_per_h"] = np.nan
        print(f"    Spindle detection failed: {e}")

    # ── Ripples ───────────────────────────────────────────────────────────
    try:
        ripples = detect_ripples(sig_ca3, fs)
        results["ripple_rate_per_h"]     = len(ripples) / max(duration_h, 0.001)
        results["ripple_duration_ms_mean"] = ripples["duration_ms"].mean() if len(ripples) > 0 else 0
        results["n_ripples"]             = len(ripples)
    except Exception as e:
        results["ripple_rate_per_h"] = np.nan

    # ── PAC — theta-gamma ─────────────────────────────────────────────────
    try:
        # Use first 30 min for PAC (computationally expensive)
        n_pac = min(len(sig_ca3), int(30 * 60 * fs))
        mi_epochs = compute_pac_epochs(sig_ca3[:n_pac], fs,
                                        phase_band=(4, 8), amp_band=(30, 80))
        results["pac_theta_gamma_mean"] = float(np.nanmean(mi_epochs))
        results["pac_theta_gamma_std"]  = float(np.nanstd(mi_epochs))
    except Exception as e:
        results["pac_theta_gamma_mean"] = np.nan

    # ── Inter-channel coherence ────────────────────────────────────────────
    if sig_ctx is not None:
        try:
            n_coh = min(len(sig_ca3), len(sig_ctx), int(30*60*fs))
            coh_df = compute_coherence_epochs(sig_ca3[:n_coh], sig_ctx[:n_coh], fs)
            for band in ["delta","theta","alpha","beta","gamma"]:
                col = f"coh_{band}"
                results[f"ca3_ctx_coh_{band}"] = float(coh_df[col].mean()) if col in coh_df else np.nan
        except Exception as e:
            pass

    # ── Fractal dimension (subset) ─────────────────────────────────────────
    try:
        n_fd = min(len(sig_ca3), int(60 * fs))  # 1 min
        results["higuchi_fd"] = higuchi_fd(sig_ca3[:n_fd])
    except Exception:
        results["higuchi_fd"] = np.nan

    # ── DFA ────────────────────────────────────────────────────────────────
    try:
        n_dfa = min(len(sig_ca3), int(5*60*fs))
        results["dfa_alpha"] = dfa_alpha(sig_ca3[:n_dfa])
    except Exception:
        results["dfa_alpha"] = np.nan

    # ── Burst suppression ──────────────────────────────────────────────────
    try:
        results["burst_suppression_ratio"] = burst_suppression_ratio(sig_ca3, fs)
    except Exception:
        results["burst_suppression_ratio"] = np.nan

    del signals, sig_ca3
    if sig_ctx is not None:
        del sig_ctx
    gc.collect()

    return results


def process_timepoint(tp, inventory_all):
    print(f"\n{'='*65}")
    print(f"Advanced EEG Analysis — Timepoint: {tp}")
    print(f"{'='*65}")

    inv = inventory_all[inventory_all.timepoint == tp].copy()
    if len(inv) == 0:
        print(f"  No files for {tp}")
        return

    # Load existing epoch-level states if available
    ep_path = os.path.join(DATA_DIR, f"epochs_with_states_{tp}.csv")
    eeg_epochs = pd.read_csv(ep_path) if os.path.exists(ep_path) else None

    all_results = []

    for _, file_row in inv.iterrows():
        abf_path   = file_row["abf_path"]
        animal_id  = str(file_row["animal_id"])
        session_id = str(file_row["session_id"])
        group      = file_row["group"]
        abf_file   = file_row["abf_file"]

        if not os.path.exists(abf_path):
            continue

        print(f"  {session_id} | {abf_file}")

        ep_sub = None
        if eeg_epochs is not None:
            # epochs file uses session_id_A and session_id_B columns
            sid_col = "session_id_A" if "session_id_A" in eeg_epochs.columns else "session_id"
            mask = ((eeg_epochs[sid_col].astype(str) == str(animal_id)) &
                    (eeg_epochs.abf_file == abf_file))
            ep_sub = eeg_epochs[mask]

        res = run_advanced_for_file(abf_path, animal_id, session_id, group, abf_file, tp, ep_sub)
        if res:
            all_results.append(res)
            print(f"    spindles={res.get('spindle_rate_per_h',0):.1f}/h | "
                  f"ripples={res.get('ripple_rate_per_h',0):.1f}/h | "
                  f"PAC={res.get('pac_theta_gamma_mean',np.nan):.5f} | "
                  f"HFD={res.get('higuchi_fd',np.nan):.3f}")

    if not all_results:
        print(f"  No results for {tp}")
        return

    # ── Save ──────────────────────────────────────────────────────────────
    res_df = pd.DataFrame(all_results)
    out    = os.path.join(RESULTS_DIR, f"advanced_eeg_{tp}.csv")
    res_df.to_csv(out, index=False)
    print(f"\n  Saved: {out} ({len(res_df)} rows)")

    # ── Statistics ────────────────────────────────────────────────────────
    adv_feat_cols = [c for c in res_df.columns
                     if c not in ["animal_id","session_id","group","abf_file","timepoint"]]
    animal_means  = res_df.groupby(["animal_id","group"])[adv_feat_cols].mean().reset_index()

    print(f"\n  Group comparisons (WT vs KO):")
    stat_rows = []
    for feat in adv_feat_cols:
        wt = animal_means[animal_means.group=="WT"][feat].dropna()
        ko = animal_means[animal_means.group=="KO"][feat].dropna()
        if len(wt) < 2 or len(ko) < 2:
            continue
        _, p = mannwhitneyu(wt, ko, alternative="two-sided")
        d = (ko.mean()-wt.mean())/np.sqrt((wt.std()**2+ko.std()**2)/2+1e-10)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        stat_rows.append({
            "feature": feat, "wt_mean": wt.mean(), "ko_mean": ko.mean(),
            "cohens_d": d, "pval": p, "sig": sig,
            "n_wt": len(wt), "n_ko": len(ko), "timepoint": tp
        })
        if p < 0.10:
            print(f"    {feat:<35}: WT={wt.mean():.4f} KO={ko.mean():.4f} d={d:.3f} p={p:.4f} {sig}")

    stats_df = pd.DataFrame(stat_rows)
    if len(stats_df) > 1:
        _, pfd, _, _ = multipletests(stats_df["pval"], method="fdr_bh")
        stats_df["pval_fdr"] = pfd
    stats_df.to_csv(os.path.join(RESULTS_DIR, f"advanced_eeg_stats_{tp}.csv"),
                    index=False)

    return res_df


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timepoint", default=None)
    args = parser.parse_args()

    inv_path = os.path.join(DATA_DIR, "file_inventory_all_timepoints.csv")
    if not os.path.exists(inv_path):
        inv_path = os.path.join(DATA_DIR, "file_inventory_3m.csv")

    inventory = pd.read_csv(inv_path)
    tp_to_run = [args.timepoint] if args.timepoint else TP_ORDER
    for tp in tp_to_run:
        if "timepoint" in inventory.columns:
            if tp not in inventory["timepoint"].unique():
                continue
        process_timepoint(tp, inventory)

    print("\nADVANCED EEG ANALYSIS COMPLETE")
