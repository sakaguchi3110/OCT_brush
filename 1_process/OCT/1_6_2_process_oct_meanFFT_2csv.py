# -*- coding: utf-8 -*-
"""
OCT FFT/PSD summary (no changes to loading/splitting logic):
- Keep original loading, CSV structure, and split_dataframe behavior.
- Apply requested improvements ONLY on analysis side.

Improvements implemented:
(1) Use Welch PSD with proper normalization and dB representation.
(3) Boundary-safe depth/time indexing + tri-tap averaging (center-heavy).
(5) Detrend prior to Welch to mitigate low-frequency leakage.
(6) 2-Hz fixed-width band-power via *integration* of PSD (linear) -> dB.
(8) Keep original folder-name parser; no metadata CSV dependency.

Quicklook plotting (optional) can be toggled via PLOT_QUICKLOOK.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.signal import welch, detrend
from scipy.fft import fft, fftfreq  # (kept for compatibility; not used now)
import matplotlib.pyplot as plt     # for optional quicklook

# ---- Keep your project-specific path setup as-is ----
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager
from library_python.sensors.OCT.OCTMorph import OCTMorph


# === Settings (kept) ===
data_external_hdd = False
set_path_automatic = False
dataset = "OCT_BRUSH"
target_file = "skin_displacement_estimation_corrected.csv"
sampling_rate = 10000  # Fs (Hz) - DO NOT change loading logic
npyname = "phase_change_data.npy"

# --- Analysis parameters (only here we changed/added) ---
F_MAX = 1000.0          # Hz, report up to this frequency (kept same ceiling as original)
BIN_WIDTH = 2.0         # Hz, fixed-width band for integration (2 Hz as agreed)
WELCH_NPERSEG = 4096    # choose so that Δf = Fs/nperseg ≈ 2.44 Hz (OK for 2 Hz bands)
WELCH_NOVERLAP = WELCH_NPERSEG // 2
WELCH_NFFT = WELCH_NPERSEG
WELCH_WINDOW = "hann"
WELCH_DETREND = "constant"
MIN_SAMPLES = WELCH_NPERSEG  # require at least one full segment
# --- Detrend toggle ---
DETREND_ENABLE = False          # ← Falseで detrend 無効化（Trueで有効）
DETREND_TYPE_GLOBAL = "linear"  # compute_psd_welch_db の前処理 detrend 種類
DETREND_TYPE_WELCH  = "constant" # Welch内部の detrend 種類（有効時のみ）

# Depth neighborhood averaging (center-heavy tri-tap)
DEP_KERNEL = np.array([0.25, 0.50, 0.25], dtype=float)
DEP_MIN = 1      # safe bounds for tri-tap (so i-1,i,i+1 exist after clamping)
DEP_MAX = 1022

# --- Quicklook plotting (optional) ---
PLOT_QUICKLOOK = True   # Set True to show one sanity-check example
PLOT_MAX_EXAMPLES = 1


# === Loading (unchanged, except kept as functions in your script) ===
def parse_condition_name(name):
    # *** ORIGINAL FUNCTION (unchanged) ***
    parts = name.lower().split('_')
    return {
        'date': parts[0],
        'time': parts[1],
        'participant': parts[2],
        'location': parts[4],
        'texture': parts[6],
        'cover': parts[5],
        'frequency': parts[7]
    }

def split_dataframe(df):
    # *** ORIGINAL FUNCTION (unchanged) ***
    consecutive_zeros = 0
    start_index = -1
    end_index = -1

    for i in range(len(df)):
        if df.iloc[i, 0] == 0:
            consecutive_zeros += 1
            if consecutive_zeros == 10 and start_index == -1:
                start_index = i - 9
        else:
            if consecutive_zeros >= 10 and end_index == -1:
                end_index = i
            consecutive_zeros = 0

    before_df = pd.DataFrame()
    after_df = pd.DataFrame()

    if start_index != -1:
        before_index = max(0, start_index - 5001)
        before_df = df.iloc[before_index:start_index - 1]

    if end_index != -1:
        after_rows = []
        for i in range(end_index + 1, min(end_index + 5001, len(df))):
            if df.iloc[i, 0] == 0:
                break
            after_rows.append(df.iloc[i])
        after_df = pd.DataFrame(after_rows)

    return [("before_brushing", before_df), ("after_brushing", after_df)]


# === Analysis helpers (new/changed) ===
def safe_triplet_indices(i):
    """Return (i-1, i, i+1) safely clamped within [DEP_MIN, DEP_MAX]."""
    i = int(i)
    a = max(DEP_MIN, i - 1)
    b = min(DEP_MAX, i)
    c = min(DEP_MAX, i + 1)
    return a, b, c

def build_trace(phase_data, dep_indices, time_indices, offset):
    """
    Build 1D time trace by sampling phase_data at (depth+offset, time).
    - Depth tri-tap (i-1, i, i+1) with DEP_KERNEL weights, boundary-safe.
    - Keep original alignment style: time_indices come from df_part.index, used as (t_idx-1).
    """
    if phase_data is None or len(dep_indices) == 0 or len(time_indices) == 0:
        return np.array([])

    n = min(len(dep_indices), len(time_indices))
    dep_core = dep_indices[:n].astype(int) + int(offset)
    dep_core = np.clip(dep_core, DEP_MIN, DEP_MAX)

    # Original code used df_part.index then (t_idx-1); keep same pattern
    t_idx = time_indices[:n].astype(int) - 1
    T = phase_data.shape[2]
    t_idx = np.clip(t_idx, 0, T - 1)

    out = np.empty(n, dtype=float)
    for k in range(n):
        i0, i1, i2 = safe_triplet_indices(dep_core[k])
        v0 = phase_data[0, i0, t_idx[k]]
        v1 = phase_data[0, i1, t_idx[k]]
        v2 = phase_data[0, i2, t_idx[k]]

        # Handle possible duplicate neighbors near boundaries by renormalizing weights
        if (i0 == i1) and (i1 == i2):
            out[k] = float(v1)
        elif (i0 == i1) or (i1 == i2):
            vals = np.array([v0, v1, v2], float)
            w = DEP_KERNEL.copy()
            if i0 == i1:
                w[0] = 0.0
            if i1 == i2:
                w[2] = 0.0
            w = w / w.sum()
            out[k] = float((vals * w).sum())
        else:
            out[k] = float((np.array([v0, v1, v2]) * DEP_KERNEL).sum())

    return out

def compute_psd_welch_db(x, fs):
    """
    Optionally detrend then compute one-sided Welch PSD (density) and return (f, Pxx_dB).
    Detrend is controlled by DETREND_ENABLE, DETREND_TYPE_GLOBAL, DETREND_TYPE_WELCH.
    """
    if x.size < MIN_SAMPLES:
        return np.array([]), np.array([])
    # ← ここをトグル化
    x_proc = detrend(x, type=DETREND_TYPE_GLOBAL) if DETREND_ENABLE else x

    f, Pxx = welch(
        x_proc, fs=fs,
        window=WELCH_WINDOW,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        nfft=WELCH_NFFT,
        detrend=(DETREND_TYPE_WELCH if DETREND_ENABLE else False),  # ← ここもトグル
        return_onesided=True,
        scaling="density"
    )
    keep = (f >= 0.0) & (f <= F_MAX)
    f = f[keep]; Pxx = Pxx[keep]
    eps = np.finfo(float).tiny
    Pxx_db = 10.0 * np.log10(Pxx + eps)
    return f, Pxx_db


def integrate_band_power(f: np.ndarray, Pxx_db: np.ndarray, bin_width: float) -> pd.DataFrame:
    """
    Interpolated band integration:
    - Convert dB -> linear PSD
    - For each fixed-width band [L, R], linearly interpolate PSD at L/R,
      concatenate interior sample points, and integrate (trapezoid).
    - Guarantees a value for every band even if no native Welch points fall inside.
    """
    if f.size == 0:
        return pd.DataFrame(columns=["freq_bin_left_hz","freq_bin_right_hz","freq_bin_center_hz",
                                     "band_power","points"])

    Pxx_lin = 10.0 ** (Pxx_db / 10.0)

    edges = np.arange(0.0, F_MAX + bin_width, bin_width)  # e.g., 0,2,4,...,1000
    centers = (edges[:-1] + edges[1:]) / 2.0              # e.g., 1,3,5,...,999

    rows = []
    for L, R, C in zip(edges[:-1], edges[1:], centers):
        # strictly inside points; endpoints via interpolation
        inside = (f > L) & (f < R)
        pL = float(np.interp(L, f, Pxx_lin))
        pR = float(np.interp(R, f, Pxx_lin))

        if np.any(inside):
            fi = f[inside]
            Pi = Pxx_lin[inside]
            fi_full = np.concatenate(([L], fi, [R]))
            Pi_full = np.concatenate(([pL], Pi, [pR]))
            points = int(inside.sum())
        else:
            fi_full = np.array([L, R], dtype=float)
            Pi_full = np.array([pL, pR], dtype=float)
            points = 0

        band_power = float(np.trapz(Pi_full, fi_full))  # linear total power in the band (signal^2)
        rows.append((L, R, C, band_power, points))

    return pd.DataFrame(rows, columns=[
        "freq_bin_left_hz","freq_bin_right_hz","freq_bin_center_hz","band_power","points"
    ])



def compute_psd_welch(trace: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper returning both PSD linear (density) and dB.
    """
    f, Pxx_db = compute_psd_welch_db(trace, fs)
    if f.size == 0:
        return f, np.array([]), np.array([])
    Pxx_lin = 10.0 ** (Pxx_db / 10.0)
    return f, Pxx_lin, Pxx_db


def quicklook_plot(trace, fs, f, Pxx_db, band_df, title=""):
    """Minimal sanity-check plot (time snippet + PSD and band-power)."""
    if trace.size == 0 or f.size == 0 or band_df.empty:
        return
    max_samples = int(min(len(trace), fs * 1.0))
    t = np.arange(max_samples) / fs
    x = trace[:max_samples]

    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, x)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amplitude (arb.)")
    ax1.set_title(f"Time-domain (first {max_samples} samples)")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(f, Pxx_db)
    ax2.scatter(band_df["freq_bin_center_hz"], band_df["band_power_db"], s=12)
    ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("PSD / Band power (dB)")
    ax2.set_title("Welch PSD (dB) with 2-Hz band-power (dB)")
    ax2.grid(True, which="both", axis="x", linestyle=":")
    if title:
        fig.suptitle(title)
    plt.tight_layout(); plt.show()


def compute_fft_amp_power(trace: np.ndarray, fs: float, fmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute one-sided amplitude and power spectra from a single trace.
    - Detrend (linear), apply Hann window
    - Coherent-gain correction for Hann
    - One-sided scaling (×2) except DC/Nyquist
    Returns: (f, amplitude, power) up to fmax
    """
    if trace.size == 0:
        return np.array([]), np.array([]), np.array([])

    x = detrend(trace.astype(float), type="linear")
    N = x.size
    w = np.hanning(N)
    cg = w.sum() / N  # coherent gain of the window (Hann → 0.5)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(N, d=1.0/fs)

    # Amplitude spectrum with coherent-gain & one-sided scaling
    amp = (np.abs(X) / (N * max(cg, np.finfo(float).tiny)))
    if f.size > 0:
        # double all bins except DC and Nyquist (if present)
        if N % 2 == 0 and f[-1] == fs/2:
            amp[1:-1] *= 2.0
        else:
            amp[1:] *= 2.0

    power = amp ** 2

    keep = (f >= 0.0) & (f <= fmax)
    return f[keep], amp[keep], power[keep]


# === Main analysis (loading section unchanged) ===
def main():
    # --- Paths (unchanged) ---
    db_path = path_tools.define_OCT_database_path(data_external_hdd)
    db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")
    input_foldernames, input_foldernames_abs, _ = path_tools.get_folders_with_file(
        db_path_input, target_file, automatic=set_path_automatic, select_multiple=False, verbose=True
    )

    filepaths = [os.path.join(folder, npyname) for folder in input_foldernames_abs]
    corrpaths = [os.path.join(folder, target_file) for folder in input_foldernames_abs]

    phase_change_data_list = []
    for f in filepaths:
        try:
            data = np.load(f)
            phase_change_data_list.append(data)
        except Exception as e:
            print(f"⚠️ Failed to load {f}: {e}")
            phase_change_data_list.append(None)
    csv_data_list = [pd.read_csv(f) for f in corrpaths]

    # --- Output rows (wide format like original, but store band_power_db) ---
    output_rows = []

    # Optional plotting counter
    plotted = 0

    # --- Main loop (structure kept) ---
    for idx, cond in enumerate(input_foldernames):
        parsed = parse_condition_name(cond)
        date_time = f"{parsed['date']}_{parsed['time']}"
        participant = parsed['participant']
        location = parsed['location']
        texture = parsed['texture']
        cover = parsed['cover']
        frequency = parsed['frequency']

        if cover not in ['bare', 'tegaderm']:
            print(f"⚠️ Skip folder without valid cover ('bare'/'tegaderm'): {cond}")
            continue

        phase_data = phase_change_data_list[idx]
        if phase_data is None:
            print(f"⚠️ phase_change_data is None for {input_foldernames[idx]}. Skipping.")
            continue

        csv_data = csv_data_list[idx]
        dfs = split_dataframe(csv_data)

        offsets = [2, 20, 40, 160] if cover == 'bare' else [22, 40, 60, 180]

        for label, df_part in dfs:
            if df_part.empty:
                continue

            # Keep original index usage for time alignment
            dep_indices = df_part.iloc[:, 0].astype(int).values
            dep_indices = np.clip(dep_indices, 0, 1023)  # original clipping
            time_indices = df_part.index.values
            time_indices = time_indices[time_indices > 0]  # original behavior

            for offset in offsets:
                # Boundary-safe trace with tri-tap averaging (improvement #3)
                trace = build_trace(phase_data, dep_indices, time_indices, offset)

                # Require minimum samples for Welch (improvement #5)
                if trace.size < MIN_SAMPLES:
                    continue

                # Welch PSD in dB (improvement #1 + #5)
                f, Pxx_db = compute_psd_welch_db(trace, sampling_rate)
                if f.size == 0:
                    continue

                # --- Build rows in a wide format similar to original ---
                base_info = {
                    'datetime': date_time,
                    'participant': participant,
                    'location': location,
                    'texture': texture,
                    'cover': cover,
                    'frequency': frequency,
                    'condition_phase': label,
                    'depth_offset': offset,
                }

                # Compute PSD (Welch) in dB for integration
                f_psd, Pxx_db = compute_psd_welch_db(trace, sampling_rate)
                if f_psd.size == 0:
                    continue

                # Interpolated 2-Hz band-power (linear), 0–1000 Hz
                band_df = integrate_band_power(f_psd, Pxx_db, BIN_WIDTH)

                # Keep only 0–1000 Hz centers (should already be 1..999)
                mask = (band_df["freq_bin_center_hz"] >= 1.0) & (band_df["freq_bin_center_hz"] <= 999.0)
                band_df = band_df.loc[mask]

                # Build a single wide row with band_power_lin only
                row_bp_lin = base_info.copy()
                row_bp_lin['data_type'] = 'band_power_lin'
                for c, vlin in zip(band_df['freq_bin_center_hz'].values,
                                band_df['band_power'].values):
                    row_bp_lin[f'{c:.1f}Hz'] = vlin
                output_rows.append(row_bp_lin)

                # band_power_db
                row_bp_db = base_info.copy()
                row_bp_db['data_type'] = 'band_power_db'
                eps = np.finfo(float).tiny
                for c, vlin in zip(band_df['freq_bin_center_hz'].values,
                                band_df['band_power'].values):
                    row_bp_db[f'{c:.1f}Hz'] = 10.0 * np.log10(vlin + eps)
                output_rows.append(row_bp_db)

    # === Save (kept similar to original) ===
    df_out = pd.DataFrame(output_rows)

    if df_out.empty:
        print("❌ No rows to save (all segments were empty or too short).")
        return

    save_root = Path(input_foldernames_abs[0]).parent
    participants = "_".join(sorted(set(df_out["participant"].astype(str).unique())))
    save_path = save_root / f"bandpower_summary_{participants}.csv"  # <-- rename here
    df_out.to_csv(save_path, index=False, encoding="utf-8")
    print(f"✅ Band-power summary (linear) saved to:\n{save_path}")
    print("--- Summary ---")
    print(f"Rows: {len(df_out)} | Unique data_types: {df_out['data_type'].unique().tolist()}")
    print(f"Freq columns example: {[c for c in df_out.columns if c.endswith('Hz')][:5]}")


if __name__ == "__main__":
    main()
