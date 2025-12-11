# run_quality_decimate10.py
import os, pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import tifffile as tiff
from scipy.signal import butter, filtfilt, resample_poly
from scipy.ndimage import gaussian_filter
import sys

# ---- プロジェクト依存の import ----
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager
from library_python.sensors.OCT.OCTMorph import OCTMorph

def load_acquisition(input_fn_abs, morph_fn):
    octr = OCTRecordingManager(input_fn_abs, input_fn_abs, autosave=False)
    with open(os.path.join(input_fn_abs, "metadata.pkl"), "rb") as f:
        octr.metadata = pickle.load(f)
    if not octr.metadata.isVibration:
        return None
    pkl = os.path.join(input_fn_abs, morph_fn)
    if not Path(pkl).is_file():
        return None
    with open(pkl, "rb") as f:
        octr.morph = pickle.load(f)
    return octr

def temporal_downsample_snr(A_linear, fs_hz, factor):
    """
    A_linear: 線形振幅 (lines, depths, time)
    1) パワー化 → 2) アンチエイリアス（ゼロ位相LPF）→ 3) 多相デシメート → 4) RMS振幅へ戻す
    """
    # パワー（SNR平均の基本）
    P = A_linear**2

    # アンチエイリアスLPF：新ナイキストの ~40% で安全
    fs_new = fs_hz / factor
    fc = 0.4 * (fs_new/2.0)
    wn = fc / (fs_hz/2.0)
    b, a = butter(N=4, Wn=wn, btype="low")
    Pf = filtfilt(b, a, P, axis=-1, padtype='odd', padlen=3*max(len(a), len(b)))

    # 多相デシメート（時間軸のみ）
    Pd = resample_poly(Pf, up=1, down=factor, axis=-1)

    # RMS振幅へ戻す
    A_ds = np.sqrt(np.maximum(Pd, 0.0))
    return A_ds, fs_new

if __name__ == "__main__":
    # ---- 設定 ----
    dataset         = "OCT_BRUSH"
    morph_pkl_name  = "morph.pkl"
    save_dir_suffix = "3_analysed"
    factor          = 10   # 時間分解能を 1/10 に
    filename_tpl    = "morph_quality_decimate{f}x_{fs_khz:.3f}kHz.tiff"

    # （任意）空間側の軽いS/Nアップ
    USE_DEPTH_SMOOTH= True
    DEPTH_SIGMA_PX  = 1.0

    # ---- 対象フォルダ列挙 ----
    db_root = path_tools.define_OCT_database_path(False)
    base_in = os.path.join(db_root, dataset, "2_processed", "oct")
    names, abspaths, _ = path_tools.get_folders_with_file(base_in, morph_pkl_name, automatic=False, select_multiple=False)

    print(datetime.now())
    for idx, in_name in enumerate(names, 1):
        in_abs  = abspaths[idx-1]
        out_abs = in_abs.replace("2_processed", save_dir_suffix)
        os.makedirs(out_abs, exist_ok=True)

        print(f"[{idx}/{len(names)}] {in_name}")
        octr = load_acquisition(in_abs, morph_pkl_name)
        if octr is None:
            print("  skip (not vibration or missing morph.pkl)")
            continue

        Fs = float(octr.metadata.Fs_OCT)  # [Hz]

        # 線形振幅で取得
        A = np.abs(octr.morph.morph).astype(np.float32)   # (lines, depths, time)

        # 時間1/10：アンチエイリアス＋デシメート（SNR最大化の正攻法）
        A_ds, Fs_new = temporal_downsample_snr(A, fs_hz=Fs, factor=factor)

        # （任意）深さ方向の軽い平滑
        if USE_DEPTH_SMOOTH:
            A_ds = gaussian_filter(A_ds, sigma=(0.0, DEPTH_SIGMA_PX, 0.0))

        # dB化（保存用）
        A_ds = np.nan_to_num(A_ds, nan=0.0, posinf=0.0, neginf=0.0)
        A_ds[A_ds < 1e-12] = 1e-12
        dB = 20.0 * np.log10(A_ds)

        # TIFF 保存：A-line ごとに 1 スライス（frames, H=depth, W=time_new）
        stack = np.transpose(dB, (0,1,2))
        out_tiff = os.path.join(out_abs, filename_tpl.format(f=factor, fs_khz=Fs_new/1000.0))
        tiff.imwrite(out_tiff, stack.astype(np.float32))
        print("  saved:", out_tiff)
    print(datetime.now(), "done.")
