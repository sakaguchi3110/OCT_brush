# -*- coding: utf-8 -*-
"""
OCT CWT quicklook (3段図)
- x軸はサンプル番号（1000刻み）、表示は3000ポイントに制限
- CWTカラーバー固定 [0, 17]
- 位相縦軸を ±π に固定
- 深さオフセットごとに色分け
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager
from library_python.sensors.OCT.OCTMorph import OCTMorph
from typing import cast
from matplotlib.gridspec import GridSpec

# ===== Settings =====
data_external_hdd = False
set_path_automatic = False
dataset = "OCT_BRUSH"
target_file = "skin_displacement_estimation_corrected.csv"
sampling_rate = 10000
npyname = "phase_change_data.npy"

# CWT params
FMIN = 20.0
FMAX = 500.0
VOICES_PER_OCT = 32
CMOR_B, CMOR_C = 2.5, 1.0

# Depth safe bounds
DEP_MIN = 1
DEP_MAX = 1022

# 固定レンジ
CWT_VMIN = 0.0
CWT_VMAX = None #40.0
WIN_SAMPLES = 3000  # 3000ポイント表示

# ---------- helpers ----------
def make_offset_colors(n):
    cmap = plt.cm.get_cmap('autumn')
    return [cmap(x) for x in np.linspace(0.15, 0.95, n)]

def load_acquisition(input_fn_abs, output_folder_abs, morphFilename):
    import pickle
    octr = OCTRecordingManager(input_fn_abs, output_folder_abs, autosave=False)
    meta_fp = os.path.join(input_fn_abs, "metadata.pkl")
    with open(meta_fp, 'rb') as f:
        octr.metadata = pickle.load(f)
    if not getattr(octr.metadata, "isVibration", False):
        print("Current recording is not a vibration dataset.")
        return False, None
    morph_fp = os.path.join(input_fn_abs, morphFilename)
    if not Path(morph_fp).is_file():
        print("File does not exist:", morph_fp)
        return False, None
    with open(morph_fp, 'rb') as f:
        octr.morph = cast(OCTMorph, pickle.load(f))
    # これらは安全に呼べる想定（内部でlazy生成）
    if hasattr(octr.morph, "get_morph_img"):
        octr.morph.get_morph_img()
    if hasattr(octr.morph, "get_morph_video"):
        octr.morph.get_morph_video()
    return True, octr
def build_trace_fixed_depth(phase_data, depth, time_indices):
    """
    depth : 指定深さ（int）
    time_indices : CSV の index をそのまま使用
    """
    depth = int(depth)
    depth = np.clip(depth, DEP_MIN, DEP_MAX)
    t_idx = np.clip(time_indices.astype(int), 0, phase_data.shape[2]-1)
    return phase_data[0, depth, t_idx].astype(float)

def build_trace(phase_data, dep_indices, time_indices, offset):
    if phase_data is None or len(dep_indices) == 0 or len(time_indices) == 0:
        return np.array([])
    n = min(len(dep_indices), len(time_indices))
    dep_core = np.clip(dep_indices[:n].astype(int) + int(offset), DEP_MIN, DEP_MAX)
    t_idx = np.clip(time_indices[:n].astype(int), 0, phase_data.shape[2]-1)
    return phase_data[0, dep_core, t_idx].astype(float)

def cwt_morlet_pywt(x, fs, fmin=FMIN, fmax=FMAX, voices_per_oct=VOICES_PER_OCT, cmor_B=CMOR_B, cmor_C=CMOR_C):
    n_freqs = int(np.log2(fmax/fmin) * voices_per_oct)
    n_freqs = max(n_freqs, 8)
    freqs = np.geomspace(fmin, fmax, n_freqs)
    wavelet = f"cmor{cmor_B}-{cmor_C}"
    fc = pywt.central_frequency(wavelet)
    scales = (fc * fs) / freqs
    coeffs, _ = pywt.cwt(x, scales=scales, wavelet=wavelet, sampling_period=1.0/fs)
    power = (np.abs(coeffs)**2).T  # [time x freq]
    return freqs, power

def parse_condition_name(name):
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
# def plot_fixed_depth_panel(octr, fixed_depth, time_indices, t_samples,
#                            phase_trace, freqs, power, save_path,
#                            *, show=False, line_color='blue'):

#     # 振幅画像
#     amp_img = np.array(octr.morph.morph_dB_video[0])   # depth × time

#     # 振幅の使用区間
#     amp_win = amp_img[:, t_samples]
#     t_rel = t_samples - t_samples[0]

#     depth_min, depth_max = 0, amp_win.shape[0]-1

#     # -----------------------
#     # Figure
#     # -----------------------
#     fig = plt.figure(figsize=(12, 8.5), constrained_layout=True)
#     gs = GridSpec(3, 1, height_ratios=[1.0, 0.7, 1.2], hspace=0.05, figure=fig)

#     ax0 = fig.add_subplot(gs[0])
#     ax1 = fig.add_subplot(gs[1], sharex=ax0)
#     ax2 = fig.add_subplot(gs[2], sharex=ax0)

#     # ===== 上段：Amplitude =====
#     extent0 = [0, t_rel[-1], depth_max, depth_min]
#     im0 = ax0.imshow(amp_win, aspect='auto', cmap='gray', origin='upper', extent=extent0)
#     ax0.set_title(f"Amplitude (dB)  |  fixed depth = {fixed_depth}")
#     ax0.set_ylabel("Depth (px)")

#     # ★ fixed depth の水平線
#     ax0.axhline(y=fixed_depth, color=line_color, lw=1.5)

#     # ===== 中段：Δφ =====
#     ax1.plot(t_rel, phase_trace, color=line_color, lw=1.3)
#     ax1.set_ylabel("Δφ [rad]")
#     ax1.set_ylim(-np.pi, np.pi)

#     # ===== 下段：CWT =====
#     extent2 = [0, t_rel[-1], freqs[0], freqs[-1]]
#     im2 = ax2.imshow(power.T, aspect='auto', origin='lower', extent=extent2,
#                      vmin=0, vmax=np.nanmax(power))
#     ax2.set_ylabel("Frequency (Hz)")
#     ax2.set_xlabel("Time (samples)")
#     ax2.set_title("CWT Power (Morlet)")

#     # カラーバー
#     fig.colorbar(im0, ax=[ax0], fraction=0.046, pad=0.02)
#     fig.colorbar(im2, ax=[ax2], fraction=0.046, pad=0.02)

#     plt.setp(ax0.get_xticklabels(), visible=False)
#     plt.setp(ax1.get_xticklabels(), visible=False)

#     fig.savefig(save_path, dpi=200, bbox_inches='tight')
#     if show:
#         plt.show(block=True)
#     plt.close(fig)
    
def plot_three_panel(octr_or_none, amp_img, depth_track_times, depth_track_vals,
                     t_samples, phase_trace, freqs, power, save_path, *, cwt_vmin=0.0, cwt_vmax=40.0, line_color=(1,0,0,1), show=False):
    # --- Amplitude map source ---
    if amp_img is not None:
        amp_full = amp_img
    elif (octr_or_none is not None and
          hasattr(octr_or_none, "morph") and
          hasattr(octr_or_none.morph, "morph_dB_video") and
          len(octr_or_none.morph.morph_dB_video) > 0):
        amp_full = np.array(octr_or_none.morph.morph_dB_video[0])
    else:
        print("⚠️ No amplitude image available; skip plot.")
        return

    # 相対サンプル座標（0開始）＆ 3000ポイント制限
    t_points = t_samples.copy()
    t_points_rel = t_points - t_points[0]
    if t_points_rel.size > WIN_SAMPLES:
        keep = WIN_SAMPLES
        t_points_rel = t_points_rel[:keep]
        amp_win = amp_full[:, t_samples[:keep]]
        phase_trace = phase_trace[:keep]
        power = power[:keep, :]
    else:
        amp_win = amp_full[:, t_samples]

    depth_min, depth_max = 0, amp_win.shape[0]-1

    # トラッキング線（同じ時間窓に限定・xはサンプル番号）
    if depth_track_times is None or len(depth_track_times) == 0:
        tt_samp = np.array([])
        yy_dep = np.array([])
    else:
        m = (depth_track_times >= t_samples[0]) & (depth_track_times <= t_samples[-1])
        tt_samp = depth_track_times[m] - t_samples[0]
        yy_dep = depth_track_vals[m]

    fig = plt.figure(figsize=(12, 8.5), constrained_layout=True)
    gs  = GridSpec(3, 1, height_ratios=[1.0, 0.55, 1.0], hspace=0.05, figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    # 上：Amplitude（グレースケール）
    extent0 = [0, t_points_rel[-1], depth_max, depth_min]
    im0 = ax0.imshow(amp_win, aspect='auto', cmap='gray', origin='upper', extent=extent0)
    ax0.set_ylabel("Depth (px)")
    # ax0.set_title("Amplitude (dB or magnitude) + tracking")
    ax0.set_title("Slice phase change + tracking")
    if tt_samp.size > 0:
        ax0.plot(tt_samp, yy_dep, color=line_color, lw=1.8)

    # 中：位相（±πに固定）
    ax1.plot(t_points_rel, phase_trace, lw=1.2, color=line_color)
    ax1.set_ylabel("Δφ [rad]")
    ax1.set_ylim(-np.pi, np.pi)
    # ax1.set_title("Δφ [rad]")

    # 下：CWT（vmin/vmax固定、x=サンプル番号）
    extent2 = [0, t_points_rel[-1], freqs[0], freqs[-1]]
    im2 = ax2.imshow(power.T, aspect='auto', origin='lower', extent=extent2,
                     vmin=cwt_vmin, vmax=cwt_vmax)
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (samples)")
    ax2.set_title("CWT Power (Morlet)")

    # x目盛を1000刻みに
    tick_step = 1000
    ticks = np.arange(0, t_points_rel[-1] + 1, tick_step)
    ax2.set_xticks(ticks)

    # カラーバー
    cbar0 = fig.colorbar(im0, ax=[ax0], location='right', fraction=0.046, pad=0.02)
    cbar0.set_label("Amplitude")
    cbar2 = fig.colorbar(im2, ax=[ax2], location='right', fraction=0.046, pad=0.02)
    cbar2.set_label("Power")

    # 共有xなので上2段のxラベル消す
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show(block=True)
    
# ---------- main ----------
def main():
    # 入力ディレクトリ
    db_path = path_tools.define_OCT_database_path(data_external_hdd)
    db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")
    input_foldernames, input_foldernames_abs, _ = path_tools.get_folders_with_file(
        db_path_input, target_file, automatic=set_path_automatic, select_multiple=False, verbose=True
    )
    if not input_foldernames_abs:
        print("❌ No folders found that contain", target_file, "under:", db_path_input)
        return

    # データロード
    filepaths = [os.path.join(folder, npyname) for folder in input_foldernames_abs]
    corrpaths = [os.path.join(folder, target_file) for folder in input_foldernames_abs]
    phase_change_data_list, csv_data_list = [], []
    for f in filepaths:
        try:
            phase_change_data_list.append(np.load(f))
        except Exception as e:
            print(f"⚠️ Failed to load {f}: {e}"); phase_change_data_list.append(None)
    for f in corrpaths:
        try:
            csv_data_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"⚠️ Failed to read CSV {f}: {e}"); csv_data_list.append(pd.DataFrame())

    ONLY_AFTER = True
    def offsets_for_cover(cover):
        # 必要に応じて増減させてOK
        return [2, 20, 40, 160, 320] if cover == 'bare' else [22, 40, 60, 180, 340]

    work_items = []
    group_max = {}
    for idx, cond in enumerate(input_foldernames):
        parsed = parse_condition_name(cond)
        cover = parsed['cover']
        phase_data = phase_change_data_list[idx]
        csv_data = csv_data_list[idx]
        if phase_data is None or csv_data.empty:
            print(f"⚠️ Missing data for: {cond}")
            continue

        # morph.pkl 読み込み
        input_fn_abs = input_foldernames_abs[idx]
        output_folder_abs = input_fn_abs.replace("2_processed", "3_analysed")
        success, octr = load_acquisition(input_fn_abs, output_folder_abs, "morph.pkl")
        success = False
        if not success or octr is None:
            print(f"⚠️ morph.pkl not loaded for: {cond}")
            amp_img = phase_data[0, :, :]  # fallback
        else:
            amp_img = np.array(octr.morph.morph_dB_video[0])
            print(f"[INFO] amp_img shape: {amp_img.shape}")
            
        morph0 = np.array(octr.morph.morph)[0]   # shape = [depth, time]
        phase_unwrapped = np.unwrap(np.angle(morph0), axis=1)  # depth × time


        # 保存先（サンプル直下）
        fig_dir = Path(input_foldernames_abs[idx])

        # before/after 分割
        dfs = split_dataframe(csv_data)
        offsets = offsets_for_cover(cover)
        offset_colors = make_offset_colors(len(offsets))

        for label, df_part in dfs:
            if ONLY_AFTER and label != "after_brushing":
                continue
            if df_part.empty:
                continue

            # 時間窓（3000ポイント制限）
            t0 = int(df_part.index.values[0])
            t1 = int(df_part.index.values[-1])
            t1_lim = min(t1, t0 + WIN_SAMPLES - 1)
            time_win = np.arange(t0, t1_lim + 1)

            dep_indices = np.clip(df_part.iloc[:, 0].astype(int).values, 0, 1023)
            time_indices = df_part.index.values

            for oi, offset in enumerate(offsets):
                line_color = offset_colors[oi]

                # トレース（当該オフセットで再構成）
                trace_sparse = build_trace(phase_data, dep_indices, time_indices, offset)
                trace = np.full(time_win.shape, np.nan, dtype=float)
                cols = np.searchsorted(time_win, time_indices)
                cols = cols[(cols >= 0) & (cols < trace.size)]
                trace[cols] = trace_sparse[:len(cols)]
                from scipy.signal import savgol_filter

                if np.any(np.isnan(trace)):
                    valid = np.where(~np.isnan(trace))[0]
                    if valid.size >= 2:
                        trace = np.interp(np.arange(trace.size), valid, trace[valid])
                    else:
                        print(f"⚠ trace too sparse: {cond} | {label} | off{offset}")
                        continue

                # ★ ここで軽く平滑化（window_length は奇数）
                SMOOTH = True
                if SMOOTH:
                    trace = savgol_filter(trace, window_length= 9, polyorder=2)

                # CWT
                freqs, power = cwt_morlet_pywt(trace, sampling_rate)

                # ★追加：グループ最大を更新
                key = (cond, label)
                pmax = np.nanmax(power)
                if np.isfinite(pmax):
                    group_max[key] = max(group_max.get(key, 0.0), pmax)

                # ★追加：描画ジョブとして貯める（ここでは保存しない）
                work_items.append({
                    "key": key,
                    "fig_dir": fig_dir,
                    "base": f"{parsed['participant']}_{parsed['location']}_{parsed['texture']}_{parsed['cover']}_{parsed['frequency']}_{label}_off{offset}",
                    "octr": octr,
                    "amp_img": amp_img,
                    "time_indices": time_indices,
                    "dep_indices": dep_indices,
                    "time_win": time_win,
                    "trace": trace,
                    "freqs": freqs,
                    "power": power,
                    "line_color": line_color,
                    "offset": offset,
                })
                print(f"… queued: {cond} | {label} | off{offset}")
                
                # # 深さトラック（オフセット加算したものを渡す）
                # depth_track_vals_with_offset = dep_indices + offset

                # # 保存名
                # base = f"{parsed['participant']}_{parsed['location']}_{parsed['texture']}_{parsed['cover']}_{parsed['frequency']}_{label}_off{offset}"
                # out_3panel = fig_dir / f"{base}_3panel.png"

                # plot_three_panel(
                #     octr_or_none=octr,
                #     amp_img=amp_img,
                #     depth_track_times=time_indices,
                #     depth_track_vals=depth_track_vals_with_offset,
                #     t_samples=time_win,
                #     phase_trace=trace,
                #     freqs=freqs,
                #     power=power,
                #     line_color=line_color,
                #     save_path=str(out_3panel),
                #     show=False
                # )

                # print(f"✅ Saved: {out_3panel}")

        # ★追加：2パス目（ここで描画）
    for it in work_items:
        key = it["key"]
        cwt_vmin = CWT_VMIN
        # CWT_VMAX が None ならグループ最大を使う／数値ならその固定値
        cwt_vmax = CWT_VMAX if CWT_VMAX is not None else group_max.get(key, CWT_VMIN + 1.0)

        out_3panel = it["fig_dir"] / f"{it['base']}_3panel.png"
        depth_track_vals_with_offset = it["dep_indices"] + it["offset"]

        plot_three_panel(
            octr_or_none=it["octr"],
            amp_img=it["amp_img"],
            depth_track_times=it["time_indices"],
            depth_track_vals=depth_track_vals_with_offset,
            t_samples=it["time_win"],
            phase_trace=it["trace"],
            freqs=it["freqs"],
            power=it["power"],
            line_color=it["line_color"],
            cwt_vmin=cwt_vmin,
            cwt_vmax=cwt_vmax,  
            save_path=str(out_3panel),
            show=False
        )
        print(f"✅ Saved: {out_3panel}")

    print("🎉 Done. 3-panel figures saved per sample folder.")
    

if __name__ == "__main__":
    main()
