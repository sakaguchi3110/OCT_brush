import numpy as np
import h5py
import os
from scipy.interpolate import interp1d
# from scipy.fftpack import fft
from scipy.signal import detrend, butter, filtfilt
from scipy.signal.windows import hann
from scipy.interpolate import CubicSpline
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import matplotlib
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter1d

# Check if running on SSH server
if 'SSH_CONNECTION' in os.environ:
    matplotlib.use('TkAgg')  # Use TkAgg or any other X11 back-end
else:
    pass
    #matplotlib.use('Agg')  # Use a non-interactive back-end for local use or environments without X11

# ==== ONE-PLACE SETTINGS (解析/可視化を完全同期) ====
CFG = dict(
    # 可視化・解析の時間範囲（x軸）
    T_RANGE=(10000, 20000),
    # 浅部カット（解析・描画の下限深さ）
    DEPTH_MIN=0,
    # 深さ→色マップ範囲（上=浅い, 下=深い）
    DEPTH_RANGE=(0, 1023),
    # 下段 y軸
    Y_LIM=(0, 1.25),
    # カラーマップ名
    CMAP='viridis',

    # 相関パラメータ（#2 複素相関）
    WZ=16, HOP=8, SEARCH=8, NEIGHBORS=2,
    RHO_MIN=0.4,      # estimate_shift_complex_corr の受理閾値
    RHO_PEAK=0.6,     # 後段ゲート（信頼度）

    # 表示用スムージング/間引き
    VIS_WIN=21, VIS_STEP=2,
)

def _parabolic_subpixel(y, i):
    denom = (2 * (y[i-1] - 2*y[i] + y[i+1]))
    if denom == 0:
        return 0.0
    return 0.5 * (y[i-1] - y[i+1]) / denom

def _ncc_xcorr(x, y, max_shift):
    """
    Normalized complex cross-correlation between complex depth-window vectors x(z), y(z).
    Search shifts tau in [-max_shift, +max_shift]. Returns (best_shift, peak_value).
    """
    # Ensure complex64/128
    x = np.asarray(x, dtype=np.complex128).ravel()
    y = np.asarray(y, dtype=np.complex128).ravel()
    N = x.size
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return np.nan, 0.0

    # Circular cross-correlation via complex FFT
    X = fft(x)
    Y = fft(y)
    r = ifft(Y * np.conj(X))  # length-N complex sequence, shift=0 at index 0

    # Center shifts around 0
    r = np.roll(r, N // 2)

    # Limit to allowed shifts
    center = N // 2
    lo = center - max_shift
    hi = center + max_shift
    r_seg = r[lo:hi+1]
    # NCC magnitude in [0,1] ideally
    r_mag = np.abs(r_seg) / (nx * ny)

    # Integer peak
    i_max = int(np.argmax(r_mag))
    tau_int = (lo + i_max) - center

    # Subpixel (parabolic) if neighbors exist
    if 0 < i_max < (r_mag.size - 1):
        delta = _parabolic_subpixel(r_mag, i_max)
    else:
        delta = 0.0

    tau = tau_int + float(delta)
    peak = float(r_mag[i_max])
    return tau, peak

def estimate_shift_complex_corr(C, Wz=16, hop=8, search=8, rho_min=0.4, neighbors=2):
    """
    C: complex array (Z, T) = self.morph[0, :, :].
    Returns:
      shifts (Nw, T-1), centers (Nw,), peaks (Nw, T-1)
    """
    # Ensure correct dtype/shape
    C = np.asarray(C)
    if C.ndim != 2:
        raise ValueError("C must be 2D (depth x time).")
    Z, T = C.shape

    # Window centers
    centers = []
    z0 = 0
    while z0 + Wz <= Z:
        centers.append(z0 + Wz//2)
        z0 += hop
    centers = np.array(centers, dtype=int)
    Nw = centers.size

    shifts = np.full((Nw, T-1), np.nan, dtype=float)
    peaks  = np.zeros((Nw, T-1), dtype=float)

    for wi, cz in enumerate(centers):
        z_start = cz - Wz//2
        z_end   = z_start + Wz
        if z_start < 0 or z_end > Z:
            continue
        # Fixed depth window slice
        win = slice(z_start, z_end)
        for t in range(1, T):
            taus, rhos = [], []
            # Compare to t-1, t-2, ... up to neighbors
            for k in range(1, neighbors+1):
                t_prev = t - k
                if t_prev < 0:
                    continue
                x = C[win, t_prev]   # previous
                y = C[win, t]        # current
                tau, rho = _ncc_xcorr(x, y, max_shift=search)
                if np.isfinite(tau):
                    taus.append(tau)
                    rhos.append(rho)
            if not taus:
                continue
            tau_med = float(np.median(taus))
            rho_max = float(np.max(rhos))
            if rho_max >= rho_min:
                shifts[wi, t-1] = tau_med
                peaks[wi,  t-1] = rho_max
            # else leave as NaN/0
    return shifts, centers, peaks



def build_plot_payload(C, centers, peaks, shifts,
                       Wz=16,
                       depth_min=None, t_min=None, t_max=None, rho_peak=0.6,
                       vis_win=21, vis_step=2,
                       z_origin=0, t_origin=0):
    Z, T = C.shape
    Nw = centers.size

    # --- gates (amplitude & correlation) ---
    amp = np.abs(C)
    amp_win = np.empty((Nw, T), dtype=np.float32)
    for i, cz_g in enumerate(centers):
        cz = cz_g - z_origin                         # ★ グローバル→ローカル
        z0 = max(cz - Wz//2, 0); z1 = min(cz + Wz//2, Z)
        amp_win[i] = amp[z0:z1].mean(axis=0)
    gate_corr = (peaks >= rho_peak)                       # (Nw,T-1)
    gate = gate_corr                     # (Nw,T-1)

    # --- displacement & |Δ| ---
    sh = np.array(shifts, copy=True)
    sh[~np.isfinite(sh)] = 0.0
    sh = np.where(gate, sh, 0.0)
    disp = np.cumsum(sh, axis=1)
    disp = np.concatenate([np.zeros((Nw,1)), disp], axis=1)
    
    v_frame = np.abs(np.diff(disp, axis=1))
    v_frame = np.concatenate([v_frame[:, :1], v_frame], axis=1)  # (Nw, T_local)

    T_local = disp.shape[1]

    # --- ★ Moving-Average 差分（符号付き）★ ---
    W = 10          # 窓幅（例：1-10, 6-15 ...）
    STEP = 5        # ずらし幅
    starts = np.arange(0, T_local - W + 1, STEP)          # 各窓の開始インデックス
    n_win = starts.size
    # 窓平均（Nw, n_win）
    M = np.empty((Nw, n_win), dtype=np.float32)
    for i, s in enumerate(starts):
        M[:, i] = np.nanmean(disp[:, s:s+W], axis=1)

    # 連続2窓の平均差（符号付き）：(Nw, n_win-1)
    v = np.abs(M[:, 1:] - M[:, :-1])   # ★平均差の絶対値★

    # この系列に対応する時刻（ローカル→グローバル）
    t_centers_local = starts + W/2.0                      # 各窓の中心
    t_pairs_local   = 0.5*(t_centers_local[1:] + t_centers_local[:-1])  # 2窓の中点
    t_vis = (t_origin + t_pairs_local).astype(np.int32)    # グローバル時刻

    # 深さの抽出
    keep_depth = centers > depth_min
    vs = v[keep_depth]                              # (K, T′)
    depth_vis = centers[keep_depth].astype(np.int16)

    # 表示用なめらかし（軽い移動平均）と間引き
    base = np.nanmedian(vs)
    if not np.isfinite(base):
        base = 0.0
    vs_f = uniform_filter1d(np.nan_to_num(vs, nan=base),
                            size=vis_win, axis=1, mode="nearest")
    vs_vis = vs_f[:, ::vis_step].astype(np.float16)

    # 時刻ベクトル（グローバル時刻に戻す）
    # t_vis_full = (t_origin + idx).astype(np.int32)
    t_vis = t_vis[::vis_step].astype(np.int32)

    return dict(
        t_vis=t_vis, vs_vis=vs_vis, depth_vis=depth_vis,
        a=np.int32(0),
        t_min=np.int32(t_min) if t_min is not None else np.int32(0),
        t_max=np.int32(t_max) if t_max is not None else np.int32(0),
        depth_min=np.int32(depth_min),
        centers_global=centers.astype(np.int32),
        t_origin=np.int32(t_origin),
        v_frame=v_frame.astype(np.float32)
    )
    
    # # disp_smooth = uniform_filter1d(disp, size=11, axis=1, mode="nearest")
    # # # 5フレームごとにサンプリング（重複なし）
    # # step = 5
    # # v_ma = disp_smooth[:, ::step]  # Tを約1/5に間引き
    # # v = np.abs(np.diff(v_ma, axis=1))                     # 変位変化量
    # # v = np.concatenate([v[:, :1], v], axis=1)
    # disp_smooth = gaussian_filter1d(disp, sigma=2, axis=1, mode="nearest")
    # # RMS計算（5フレーム単位）
    # win = 5
    # Nw, T = disp_smooth.shape
    # n_blocks = T // win
    # v_rms = np.zeros((Nw, n_blocks), dtype=np.float32)
    # for i in range(n_blocks):
    #     seg = disp_smooth[:, i*win:(i+1)*win]
    #     v_rms[:, i] = np.sqrt(np.nanmean(np.diff(seg, axis=1)**2, axis=1))

    # v = v_rms  # 以降の処理は v を使う
    
    # # v = np.abs(np.diff(disp, axis=1))
    # # v = np.where(gate, v, np.nan)
    # # v = np.concatenate([v[:, :1], v], axis=1)   # v: (Nw, T_local)
    
    # # # === 5フレーム非重複総和（ここが肝）===
    # # win = 5
    # # Nw, T_local = v.shape
    # # n_blocks = T_local // win
    # # if n_blocks == 0:
    # #     raise ValueError("総和窓(win)が大きすぎます。")

    # # # 末尾の端数は切り捨て（必要ならpadでも可）
    # # v5 = np.zeros((Nw, n_blocks))
    # # for i in range(n_blocks):
    # #     start = i * win
    # #     end   = start + win
    # #     v5[:, i] = np.nansum(v[:, start:end], axis=1)

    # # ブロック中心の時刻（ローカル）→ グローバルへ
    # t_centers_local = (np.arange(n_blocks) * win + (win - 1) / 2.0)
    # t_vis = (t_origin + t_centers_local).astype(np.int32)

    # # v = v5                                 # v: (Nw, n_blocks)
    # T_blk = v.shape[1]
    # # --- depth/time selection & smoothing for display ---
    # keep_depth = centers > depth_min                      # centers はグローバルでOK
    # sel_t = np.ones(T_blk, dtype=bool)      # 全ブロック選択（C_subで既に時間は切ってある）

    # vs = v[keep_depth][:, sel_t]
    # depth_vis = centers[keep_depth].astype(np.int16)

    # # 欠損マスクは先に取っておく
    # mask_nan = np.isnan(v[keep_depth][:, sel_t])

    # # extra smoothing + downsample for plotting only
    # base = np.nanmedian(vs)
    # if not np.isfinite(base):
    #     base = 0.0
    # vs_f = uniform_filter1d(np.nan_to_num(vs, nan=base), size=vis_win, axis=1, mode="nearest")
    # vs_f[mask_nan] = np.nan

    # vs_vis = vs_f[:, ::vis_step].astype(np.float16)
    # t_vis  = t_vis[::vis_step].astype(np.int32)

    # return dict(
    #     t_vis=t_vis, vs_vis=vs_vis, depth_vis=depth_vis,
    #     a=np.int32(0),
    #     t_min=np.int32(t_min), t_max=np.int32(t_max),
    #     depth_min=np.int32(depth_min)
    # )


# def build_plot_payload(C, centers, peaks, shifts,
#                        Wz=16,
#                        depth_min=None, t_min=None, t_max=None,
#                        rho_amp_db=6.0, rho_peak=0.6,
#                        vis_win=21, vis_step=2,
#                        z_origin=0, t_origin=0):

#     Z, T = C.shape
#     Nw = centers.size

#     def _compute_rms(disp_smooth, win, t_origin, mode="disp"):
#         """共通RMS計算関数（変位/差分をモードで切替）"""
#         Nw, T = disp_smooth.shape
#         n_blocks = T // win
#         rms = np.zeros((Nw, n_blocks), dtype=np.float32)
#         for i in range(n_blocks):
#             s, e = i * win, i * win + win
#             seg = disp_smooth[:, s:e]
#             if mode == "disp":  # ゼロ平均RMS
#                 seg -= np.nanmean(seg, axis=1, keepdims=True)
#             elif mode == "diff":  # 差分RMS
#                 seg = np.diff(seg, axis=1)
#             rms[:, i] = np.sqrt(np.nanmean(seg**2, axis=1))
#         t_local = np.arange(n_blocks) * win + (win - 1)/2
#         t_vis = (t_origin + t_local).astype(np.int32)
#         return rms, t_vis
    
#     def _smooth_for_plot(v, t_vis):
#         """共通スムージング・ダウンサンプリング処理"""
#         keep_depth = centers > depth_min
#         vs = v[keep_depth]
#         depth_vis = centers[keep_depth].astype(np.int16)

#         mask_nan = np.isnan(vs)
#         base = np.nanmedian(vs)
#         if not np.isfinite(base):
#             base = 0.0
#         vs_f = uniform_filter1d(np.nan_to_num(vs, nan=base),
#                                 size=vis_win, axis=1, mode="nearest")
#         vs_f[mask_nan] = np.nan

#         return vs_f[:, ::vis_step].astype(np.float16), t_vis[::vis_step].astype(np.int32), depth_vis

#     # --- gates (amplitude & correlation) ---
#     amp = np.abs(C)
#     amp_win = np.empty((Nw, T), dtype=np.float32)
#     for i, cz_g in enumerate(centers):
#         cz = cz_g - z_origin
#         z0 = max(cz - Wz//2, 0); z1 = min(cz + Wz//2, Z)
#         amp_win[i] = amp[z0:z1].mean(axis=0)
#     amp_db = 20*np.log10(np.maximum(amp_win, 1e-12))
#     noise_db = np.nanpercentile(amp_db, 20, axis=1, keepdims=True)
#     gate_amp = amp_db >= (noise_db + rho_amp_db)
#     gate_corr = (peaks >= rho_peak)
#     gate = gate_amp[:,1:] & gate_corr

#     # --- displacement ---
#     sh = np.array(shifts, copy=True)
#     sh[~np.isfinite(sh)] = 0.0
#     sh = np.where(gate, sh, 0.0)
#     disp = np.cumsum(sh, axis=1)
#     disp = np.concatenate([np.zeros((Nw,1)), disp], axis=1)
    
        
#     sigma_disp = 5   # 変位RMS用は強
#     sigma_diff = 2   # 差分RMS用は弱
#     disp_smooth_for_disp = gaussian_filter1d(disp, sigma=sigma_disp, axis=1, mode="nearest")
#     disp_smooth_for_diff = gaussian_filter1d(disp, sigma=sigma_diff, axis=1, mode="nearest")

#     win_disp = 100    # 長め
#     win_diff = 10     # 短め
    
#     rms_disp, t_vis_disp = _compute_rms(disp_smooth_for_disp, win_disp, t_origin, mode="disp")
#     rms_diff, t_vis_diff = _compute_rms(disp_smooth_for_diff, win_diff, t_origin, mode="diff")

#     # --- prepare for visualization ---
#     vs_disp, t_disp, depth_disp = _smooth_for_plot(rms_disp, t_vis_disp)
#     vs_diff, t_diff, depth_diff = _smooth_for_plot(rms_diff, t_vis_diff)

#     return dict(
#         # RMS(変位)
#         t_vis_disp=t_disp, vs_vis_disp=vs_disp, depth_vis_disp=depth_disp,
#         # RMS(変位変化)
#         t_vis_diff=t_diff, vs_vis_diff=vs_diff, depth_vis_diff=depth_diff,
#         a=np.int32(0),
#         depth_min=np.int32(depth_min),
#         t_min=np.int32(t_min) if t_min is not None else np.int32(0),
#         t_max=np.int32(t_max) if t_max is not None else np.int32(0),
#     )
    
class OCTMorph:
    def __init__(self, metadata=None, downsample_method=1):
        self.md = metadata
        self.downsample_method = downsample_method
        self.raw = []
        self.corrected = []
        self.morph = []
        self.morph_ampl = []
        self.morph_dB_img = []
        self.morph_dB_video = []

    def compute_morph(self):
        if self.md.isStructural:
            ndepths = int(self.md.OCTCCD_NPIXELS/2)
            f = ""
            if self.md.whichStructural == "2D":
                f = open(f"{self.md.folder}/{self.md.MEASFILE_STRUCT2D}", 'rb')
                morph_ = np.fromfile(f, dtype=np.float32)
                n_alines = morph_.size // ndepths
                self.morph = morph_.reshape((ndepths, n_alines))
            elif self.md.whichStructural == "3D":
                f = open(f"{self.md.folder}/{self.md.MEASFILE_STRUCT3D}", 'rb')
                ncols = morph_.size // ndepths
                n_blines = int(ncols / self.md.n_alines)
                expected_shape = (ndepths, self.md.n_alines, n_blines)
                
                if morph_.size == np.prod(expected_shape):
                    pass
                elif morph_.size > np.prod(expected_shape):
                    # Trim the array if it's too large
                    morph_ = morph_[:np.prod(expected_shape)]
                elif morph_.size < np.prod(expected_shape):
                    # Pad the array with zeros if it's too small
                    morph_ = np.pad(morph_, (0, np.prod(expected_shape) - morph_.size))
                self.morph = morph_.reshape(expected_shape)
            
            self.morph_ampl = np.abs(self.morph)
            # decibel adjustment has been already made on LabView:
            self.morph_dB_img = self.morph_ampl
            # self.save_morph_dB_img()
        else:
            self.load_raw_data()
            self.apply_hardware_correction()
            self.create_morph()
            # ... self.create_morph() の直後あたり
            if self.morph.ndim == 3 and np.iscomplexobj(self.morph):
                C = self.morph[0]  # (Z,T)

                t0, t1 = CFG['T_RANGE']          
                z0, z1 = CFG['DEPTH_MIN'], CFG['DEPTH_RANGE'][1]  
                C_sub = C[z0:z1+1, t0:t1+1]       # (Z', T') だけに削減
                
                shifts_sub, centers_sub, peaks_sub = estimate_shift_complex_corr(
                    C_sub, Wz=CFG['WZ'], hop=CFG['HOP'], search=CFG['SEARCH'],
                    rho_min=CFG['RHO_MIN'], neighbors=CFG['NEIGHBORS']
                )
                # 元の深さ座標系へ戻す
                centers = centers_sub + z0
                shifts, peaks = shifts_sub, peaks_sub

                self.plot_payload = build_plot_payload(
                    C_sub, centers, peaks, shifts,
                    Wz=CFG['WZ'],
                    depth_min=CFG['DEPTH_MIN'],
                    t_min=CFG['T_RANGE'][0],
                    t_max=CFG['T_RANGE'][1],
                    rho_peak=CFG['RHO_PEAK'],
                    vis_win=CFG['VIS_WIN'], vis_step=CFG['VIS_STEP'],
                    z_origin=CFG['DEPTH_MIN'], t_origin=CFG['T_RANGE'][0]
                )

                self.plot_payload.update({
                    "depth_range": (CFG['DEPTH_MIN'], CFG['DEPTH_RANGE'][1]),
                    "y_lim": CFG['Y_LIM'],
                    "cmap_name": CFG['CMAP'],
                })



            self.get_morph_img()
            self.get_morph_video()
        if self.md.isVibration and self.md.downsample and self.downsample_method == 3:
            self.apply_downsample(self.md.nsample)
        print("Morph is done.")


    def get_nsample(self):
        nsample = None
        if len(self.raw):
            nsample = self.raw.shape[-1]
        elif len(self.morph):
            nsample = self.morph.shape[-1]
        elif len(self.morph_ampl):
            nsample = self.morph_ampl.shape[-1]
        elif len(self.morph_dB_video):
            nsample = self.morph_dB_video.shape[-1]
        return nsample

    # if downsample is applied on already saved data, there is a chance that only .morph exists.
    # create verifications to match existing object characteristics
    def apply_downsample(self, nsample_target, average_mode):
        if len(self.morph) == 0:
            self.create_morph()
        if len(self.morph_ampl) == 0:
            self.morph_ampl = np.abs(self.morph)
        if len(self.morph_dB_video) == 0:
            self.get_morph_video()

        nLines, nDepths = self.morph.shape[:2]
        nCCDPixel = 2 * nDepths
        if len(self.raw):
            nsample_origin = self.raw.shape[-1]
        else:
            nsample_origin = self.morph.shape[-1]
        interval = np.round(np.linspace(1, nsample_origin, nsample_target + 1)).astype(int)

        raw_ = np.zeros((nLines, nCCDPixel, nsample_target), dtype=np.complex64)
        corrected_ = np.zeros((nLines, nCCDPixel, nsample_target), dtype=np.complex64)
        morph_ = np.zeros((nLines, nDepths, nsample_target), dtype=np.complex64)
        morph_ampl_ = np.zeros((nLines, nDepths, nsample_target), dtype=np.float32)
        morph_dB_video_ = np.zeros((nLines, nDepths, nsample_target))

        # for t in range(len(interval) - 1):
        #     R = slice(interval[t], interval[t + 1])
        #     if len(self.raw) > 0:
        #         raw_[:, :, t] = np.mean(self.raw[:, :, R], axis=2)
        #     if len(self.corrected) > 0:
        #         corrected_[:, :, t] = np.mean(self.corrected[:, :, R], axis=2)
        #     morph_[:, :, t] = np.mean(self.morph[:, :, R], axis=2)
        #     morph_ampl_[:, :, t] = np.mean(np.abs(self.morph_ampl[:, :, R]), axis=2)
        #     morph_dB_video_[:, :, t] = np.mean(np.abs(self.morph_dB_video[:, :, R]), axis=2)

        for t in range(len(interval) - 1):
            R = slice(interval[t], interval[t + 1])
            
            if average_mode == "complex":
                morph_[:, :, t] = np.mean(self.morph[:, :, R], axis=2)
                morph_ampl_[:, :, t] = np.abs(morph_[:, :, t])
            else:  # amplitude
                morph_ampl_[:, :, t] = np.mean(self.morph_ampl[:, :, R], axis=2)
                morph_[:, :, t] = morph_ampl_[:, :, t] * np.exp(1j * 0)  # ダミー位相で整形一致

            morph_dB_video_[:, :, t] = 20 * np.log10(np.maximum(morph_ampl_[:, :, t], 1e-12))

        self.raw = raw_
        self.corrected = corrected_
        self.morph = morph_
        self.morph_ampl = morph_ampl_
        self.morph_dB_video = morph_dB_video_

    def load_raw_data(self):
        nLines = self.md.n_alines
        nCCDPixel = self.md.OCTCCD_NPIXELS
        nSamples = self.md.nsample_original
        raw_ = np.zeros((nLines, nCCDPixel, nSamples), dtype=np.float32)
        fname = f"{self.md.folder}/{self.md.MEASFILE_DATA}"
        with h5py.File(fname, 'r') as f:
            for aline_id in range(nLines):
                print(f"Morph extracting ({aline_id + 1}/{nLines})")
                if aline_id < f['RawSpectra'].shape[0]:
                    data = np.transpose(f['RawSpectra'][aline_id, :, :])  
                else:
                    data = np.zeros((nCCDPixel, nSamples))
                raw_[aline_id, :, :] = data

        if self.md.downsample and self.downsample_method == 1:
            self.raw = np.zeros((nLines, nCCDPixel, self.md.nsample), dtype=np.float32)
            interval = np.round(np.linspace(0, self.md.nsample_original-1, self.md.nsample)).astype(int)
            for t in range(len(interval)):
                self.raw[:, :, t] = raw_[:, :, interval[t]]
        elif self.md.downsample and self.downsample_method == 2:
            self.raw = np.zeros((nLines, nCCDPixel, self.md.nsample), dtype=np.float32)
            interval = np.round(np.linspace(0, self.md.nsample_original-1, self.md.nsample)).astype(int)
            for t in range(len(interval) - 1):
                start, stop = interval[t], interval[t + 1]
                self.raw[:, :, t] = np.mean(raw_[:, :, start:stop], axis=2)
        else:
            self.raw = raw_

    def apply_hardware_correction(self):
        nLines = self.md.n_alines
        nCCDPixel = self.md.OCTCCD_NPIXELS
        nSamples = self.raw.shape[-1]
        corrected_ = np.zeros((nLines, nCCDPixel, nSamples), dtype=np.float32)
        for aline_id in range(nLines):
            print(f"Morph hardware_correction ({aline_id + 1}/{nLines})")
            corrected_[aline_id, :, :] = hardware_correction(self.raw[aline_id, :, :], self.md)
        self.corrected = corrected_

    def create_morph(self):
        nLines = self.md.n_alines
        nDepths = self.md.OCTCCD_NPIXELS // 2
        nSamples = self.corrected.shape[-1]
        morph_ = np.zeros((nLines, nDepths, nSamples), dtype='complex128')
        for aline_id in range(nLines):
            print(f"Morph fft_slice ({aline_id + 1}/{nLines})")
            morph_[aline_id, :, :] = fft_slice(self.corrected[aline_id, :, :])
        self.morph = morph_
                
        self.morph_ampl = np.abs(self.morph)

    def get_morph_video(self, adjust_inf=True, verbose=False):
        self.morph_dB_video = []
        
        if len(self.morph_ampl) == 0:

            self.morph_ampl = np.abs(self.morph)
        morph_ampl = self.morph_ampl
        if adjust_inf == True:
            # Modify values below 1 to 1
            morph_ampl[morph_ampl<1] = 1

        nLines = morph_ampl.shape[0]
        for aline_id in range(nLines):
            if verbose:
                print(f"Morph video processing ({aline_id + 1}/{nLines})")
            self.morph_dB_video.append(get_morph_video(morph_ampl[aline_id, :, :]))
        self.morph_dB_video = np.array(self.morph_dB_video)

    def get_morph_img(self, verbose=False):
        if len(self.morph_ampl) == 0:
            if len(self.morph) == 0:
                self.create_morph()
            
            self.morph_ampl = np.abs(self.morph)
        nLines = self.morph_ampl.shape[0]
        for aline_id in range(nLines):
            if verbose:
                print(f"Morph structural image processing ({aline_id + 1}/{nLines})")
            if self.md.isStructural:
                self.morph_dB_img.append(get_morph_img(self.morph_ampl[aline_id, :]))
            else:
                self.morph_dB_img.append(get_morph_img(self.morph_ampl[aline_id, :, :]))
        self.morph_dB_img = np.array(self.morph_dB_img)
        
def hardware_correction(dataSlice, metadata):
    # Ensure dataSlice is a 2D array (squeeze if necessary)
    dataSlice = np.squeeze(dataSlice)
    nsample = dataSlice.shape[-1]
    # If the data type is STIMTYPE, apply the scaling factor
    if metadata.dataType == metadata.STIMTYPE:
        dataSlice *= 540
    
    # Sort indices based on the metadata.K values
    sorted_indices = np.argsort(metadata.K)
    # Initialize the corrected array with the same shape as dataSlice
    corrected = np.zeros_like(dataSlice)
    # Perform the correction for each sample
    for i in range(nsample):
        # Subtract the Apo value from each column (sample) of the data
        dataSlice[:, i] -= metadata.Apo
        # Create the cubic spline interpolation using sorted K values
        cs = CubicSpline(metadata.K[sorted_indices], dataSlice[sorted_indices, i], axis=0)
        # Interpolate at the desired points (KES) and store in the corrected array
        corrected[[sorted_indices], i] = cs(metadata.KES[sorted_indices])
    
    # sum(sum(abs(corrected-dataSlice))) / sum(sum(abs(corrected))) can shows 40% difference
    # similar results on Matlab side
    return corrected

def fft_slice(correctedSlice, ignoreLowFreqs=False, highPassFreqID=10):
    correctedSlice = np.squeeze(correctedSlice)
    L, n_sample = correctedSlice.shape
    wind = np.tile(hann(L), (n_sample, 1)).T
    P2 = fft(correctedSlice * wind, axis=0) / L
    P2 *= 2
    if ignoreLowFreqs:
        P2[0:highPassFreqID, :] = 0
    P1 = 2 * P2[:L // 2, :]
    return P1

def get_morph_img(morph, depth_range=None):
    if len(morph.shape) == 2:
        morph = np.mean(np.abs(morph), axis=1)

    if depth_range:
        morph_dB_img = morph
        morph_dB_img[depth_range] = 20 * np.log10(morph[depth_range])
    else:
        morph_dB_img = 20 * np.log10(morph)

    return morph_dB_img

def get_morph_video(morph, depth_range=None):
    morph = np.abs(morph)
    if depth_range:
        morph_dB_video = morph
        if len(morph.shape) == 2:
            morph_dB_video[depth_range, :] = 20 * np.log10(morph[depth_range, :])
        else:
            morph_dB_video[depth_range] = 20 * np.log10(morph[depth_range])
    else:
        morph_dB_video = 20 * np.log10(morph)
    return morph_dB_video