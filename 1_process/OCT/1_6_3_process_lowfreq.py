# -*- coding: utf-8 -*-
"""
3段図（Tracking-depth Unwrap Phase + CWT）
- Bare/Tega に応じて複数 offset を自動選択
- 各 offset ごとに 3段図を横に並べて 1枚にまとめる
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from typing import cast
from matplotlib.gridspec import GridSpec

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager
from library_python.sensors.OCT.OCTMorph import OCTMorph

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# ===============================================
# Settings
# ===============================================
dataset = "OCT_BRUSH"
target_file = "skin_displacement_estimation_corrected.csv"
sampling_rate = 10000

WIN_SAMPLES = 3000      # プロットに使うサンプル数（中央 3000）
FMIN, FMAX = 1.0, 200.0
VOICES_PER_OCT = 32
CMOR_B, CMOR_C = 2.5, 1.0

DEP_MIN, DEP_MAX = 1, 1022

# ==== smoothing settings ====
USE_SMOOTH = True          # スムージング ON/OFF
SMOOTH_MODE = "sg"         # "sg" or "gaussian"
SG_WINDOW = 101            # Savitzky–Golay window (奇数)
SG_POLY = 3                # polynomial order
GAUSS_SIGMA = 8            # Gaussian sigma（点）

# ==== CWT 表示（線形 or 対数）====
USE_LOG_POWER = False      # True なら log10(power) で表示
LOG_EPS = 1e-12            # log のときの ε


# ===============================================
# Helpers
# ===============================================
def smooth_phase(x: np.ndarray) -> np.ndarray:
    """unwrap 位相の 1次元列を平滑化（任意）"""
    if not USE_SMOOTH:
        return x

    if SMOOTH_MODE == "sg":
        # window は 「信号長以下の最大の奇数」かつ SG_WINDOW 以下
        n = len(x)
        if n < 7:
            return x
        win = min(SG_WINDOW, n if n % 2 == 1 else n - 1)
        if win < 5:
            return x
        return savgol_filter(x, window_length=win, polyorder=min(SG_POLY, win - 1))

    elif SMOOTH_MODE == "gaussian":
        return gaussian_filter1d(x, sigma=GAUSS_SIGMA)

    else:
        return x


def split_dataframe(df: pd.DataFrame):
    """0 が10連続する区間を before / after に分ける（元コードを踏襲）"""
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
        before_df = df.iloc[max(0, start_index - 5001): start_index - 1]

    if end_index != -1:
        rows = []
        for i in range(end_index + 1, min(end_index + 5001, len(df))):
            if df.iloc[i, 0] == 0:
                break
            rows.append(df.iloc[i])
        after_df = pd.DataFrame(rows)

    return [("before_brushing", before_df), ("after_brushing", after_df)]

def make_offset_colors(n):
    cmap = plt.cm.get_cmap('autumn')
    return [cmap(x) for x in np.linspace(0.15, 0.95, n)]

def parse_condition_name(name: str):
    """フォルダ名から cover（bare/tega）などを抽出（以前のコードと同じ仕様）"""
    parts = name.lower().split('_')
    return {
        "date": parts[0],
        "time": parts[1],
        "participant": parts[2],
        "location": parts[4],
        "texture": parts[6],
        "cover": parts[5],          # "bare" or "tega" など
        "frequency": parts[7],
    }


def offsets_for_cover(cover: str):
    """Bare/Tega に応じて offset リストを返す（以前のフローを踏襲）"""
    cover = cover.lower()
    if cover == "bare":
        return [2, 20, 40, 160, 320]
    else:
        # Tegaderm 側（名称は実データに合わせて調整してね）
        return [22, 40, 60, 180, 340]


def cwt_morlet(x: np.ndarray, fs: float):
    """Morlet CWT（cmor）を使って power を返す"""
    n_freqs = int(np.log2(FMAX / FMIN) * VOICES_PER_OCT)
    n_freqs = max(8, n_freqs)
    freqs = np.geomspace(FMIN, FMAX, n_freqs)
    wave = f"cmor{CMOR_B}-{CMOR_C}"
    fc = pywt.central_frequency(wave)
    scales = (fc * fs) / freqs
    coeffs, _ = pywt.cwt(x, scales=scales, wavelet=wave, sampling_period=1.0 / fs)
    power = (np.abs(coeffs) ** 2).T  # [time, freq]
    return freqs, power


# ===============================================
# Plot: multi-offset 3段図
# ===============================================
def plot_multi_offset_three_panel(
    amp_img: np.ndarray,
    t_samples_plot: np.ndarray,
    offsets: list[int],
    depth_tracks_plot: list[np.ndarray],
    unwrap_plots: list[np.ndarray],
    freqs: np.ndarray,
    power_plots: list[np.ndarray],
    save_path: str,
):
    """
    amp_img: 全時間の amplitude（depth × time）
    t_samples_plot: プロットに使う時間 index（長さ WIN_SAMPLES 以下）
    offsets: 使用した offset のリスト（列数と一致）
    depth_tracks_plot: 各 offset 用 tracking depth（list of 1D arrays）
    unwrap_plots: 各 offset 用 unwrap phase（list of 1D arrays）
    power_plots: 各 offset 用 CWT power（time_cut × freq）
    """

    n_offsets = len(offsets)
    amp_win = amp_img[:, t_samples_plot]
    t_rel = t_samples_plot - t_samples_plot[0]

    depth_min, depth_max = 0, amp_win.shape[0] - 1

    fig = plt.figure(figsize=(4 * n_offsets, 9), constrained_layout=True)
    gs = GridSpec(
        3,
        n_offsets,
        height_ratios=[1.0, 0.7, 1.4],
        wspace=0.08,
        hspace=0.08,
        figure=fig,
    )

    top_axes = []
    mid_axes = []
    bottom_axes = []

    # CWT の vmax を全 offset で共通にする
    if USE_LOG_POWER:
        all_max = max(np.nanmax(p) for p in power_plots)
    else:
        all_max = max(np.nanmax(p) for p in power_plots)

    im0_ref = None
    im2_ref = None
    offset_colors = make_offset_colors(len(offsets))

    for j in range(n_offsets):
        offset = offsets[j]
        depth_tr = depth_tracks_plot[j]
        unwrap_pl = unwrap_plots[j]
        power_pl = power_plots[j]
        line_color = offset_colors[j]  

        # --- Axes 作成（列 j） ---
        if j == 0:
            ax0 = fig.add_subplot(gs[0, j])
            ax1 = fig.add_subplot(gs[1, j], sharex=ax0)
            ax2 = fig.add_subplot(gs[2, j], sharex=ax0)
        else:
            ax0 = fig.add_subplot(gs[0, j], sharex=top_axes[0])
            ax1 = fig.add_subplot(gs[1, j], sharex=top_axes[0])
            ax2 = fig.add_subplot(gs[2, j], sharex=top_axes[0])

        top_axes.append(ax0)
        mid_axes.append(ax1)
        bottom_axes.append(ax2)

        # -----------------------
        # 上段：Amplitude + tracking line
        # -----------------------
        extent0 = [0, t_rel[-1], depth_max, depth_min]
        im0 = ax0.imshow(
            amp_win,
            aspect="auto",
            cmap="gray",
            origin="upper",
            extent=extent0,
        )
        ax0.plot(t_rel, depth_tr, color=line_color, lw=1.3)

        if j == 0:
            ax0.set_ylabel("Depth (px)")
        else:
            ax0.set_yticklabels([])

        ax0.set_title(f"offset = {offset}")

        # -----------------------
        # 中段：unwrap phase
        # -----------------------
        ax1.plot(t_rel, unwrap_pl, lw=1.2, color=line_color)
        if j == 0:
            ax1.set_ylabel("unwrap phase [rad]")
        else:
            ax1.set_yticklabels([])

        # -----------------------
        # 下段：CWT
        # -----------------------
        extent2 = [0, t_rel[-1], freqs[0], freqs[-1]]
        im2 = ax2.imshow(
            power_pl.T,
            aspect="auto",
            origin="lower",
            extent=extent2,
            vmin=0,
            vmax=all_max,
        )
        if j == 0:
            ax2.set_ylabel("Frequency (Hz)")
        else:
            ax2.set_yticklabels([])

        ax2.set_xlabel("Time (samples)")

        # 参照用
        if im0_ref is None:
            im0_ref = im0
        if im2_ref is None:
            im2_ref = im2

        # 上2段の x tick は消す
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels(), visible=False)

    # カラーバー：上段と下段で 1個ずつ
    fig.colorbar(
        im0_ref,
        ax=top_axes,
        fraction=0.046,
        pad=0.02,
        label="Amplitude",
    )
    fig.colorbar(
        im2_ref,
        ax=bottom_axes,
        fraction=0.046,
        pad=0.02,
        label="Power" + (" (log10)" if USE_LOG_POWER else ""),
    )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===============================================
# MAIN
# ===============================================
def main():

    db_path = path_tools.define_OCT_database_path(False)
    db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")

    input_names, input_abs, _ = path_tools.get_folders_with_file(
        db_path_input,
        target_file,
        automatic=False,
        select_multiple=False,
        verbose=True,
    )
    if not input_abs:
        print("❌ No folders.")
        return

    for idx, cond in enumerate(input_names):

        print(f"\n=== {cond} ===")
        folder = input_abs[idx]

        # --- Bare/Tega 判定 ---
        parsed = parse_condition_name(cond)
        cover = parsed["cover"]
        offsets = offsets_for_cover(cover)
        print(f"  cover = {cover}, offsets = {offsets}")

        # --- morph 読み込み ---
        import pickle
        octr = OCTRecordingManager(folder, folder, autosave=False)
        with open(os.path.join(folder, "metadata.pkl"), "rb") as f:
            octr.metadata = pickle.load(f)
        with open(os.path.join(folder, "morph.pkl"), "rb") as f:
            octr.morph = cast(OCTMorph, pickle.load(f))
        octr.morph.get_morph_video()

        amp_img = np.array(octr.morph.morph_dB_video[0])   # depth × time
        morph0 = np.array(octr.morph.morph)[0]             # depth × time

        # unwrap 全期間
        unwrap_full = np.unwrap(np.angle(morph0), axis=1)  # depth × time

        # --- CSV 読み込み・after 抽出 ---
        df = pd.read_csv(os.path.join(folder, target_file))
        dfs = split_dataframe(df)
        after_df = next((d for (lbl, d) in dfs if lbl == "after_brushing"), None)

        if after_df is None or after_df.empty:
            print("⚠ No after data")
            continue

        # tracking depth の元データ（オフセットはまだ足さない）
        csv_depth = after_df.iloc[:, 0].astype(int).values
        time_idx = after_df.index.values.astype(int)   # After 区間の index（連続とは限らない）

        # CWT は After 全期間で計算する
        n_after = len(time_idx)

        # プロット用の中央 3000 サンプルの index（全 offset 共通）
        if n_after <= WIN_SAMPLES:
            use = np.arange(n_after)
        else:
            use = np.arange(0, WIN_SAMPLES)

        t_samples_plot = time_idx[use]

        # 各 offset ごとの結果を格納するリスト
        depth_tracks_plot = []
        unwrap_plots = []
        power_plots = []

        freqs_ref = None

        for offset in offsets:
            # ---- tracking depth（offset 加算）----
            depth_track = np.clip(csv_depth + offset, DEP_MIN, DEP_MAX)

            # unwrap_trace（After 全期間）
            # unwrap_full: [depth, time] に対し、
            # depth_track, time_idx を 1対1で参照
            unwrap_trace = unwrap_full[depth_track, time_idx]

            # スムージング
            unwrap_trace = smooth_phase(unwrap_trace)

            # CWT（After 全期間）
            freqs, power = cwt_morlet(unwrap_trace, sampling_rate)

            if USE_LOG_POWER:
                power = np.log10(power + LOG_EPS)

            if freqs_ref is None:
                freqs_ref = freqs

            # プロット用に中央 3000 サンプルへ切り出し
            unwrap_pl = unwrap_trace[use]
            depth_tr_pl = depth_track[use]
            power_pl = power[use, :]

            depth_tracks_plot.append(depth_tr_pl)
            unwrap_plots.append(unwrap_pl)
            power_plots.append(power_pl)

        # --- 図を保存 ---
        out_dir = folder.replace("2_processed", "3_analysed")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig_path = Path(out_dir) / f"{cond}_trackingDepth_unwrap_CWT_multiOffset.png"

        plot_multi_offset_three_panel(
            amp_img,
            t_samples_plot,
            offsets,
            depth_tracks_plot,
            unwrap_plots,
            freqs_ref,
            power_plots,
            save_path=str(fig_path),
        )

        print(f"✓ Saved → {fig_path}")


if __name__ == "__main__":
    main()
