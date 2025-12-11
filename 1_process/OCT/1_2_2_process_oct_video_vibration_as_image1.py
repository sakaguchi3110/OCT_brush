# run_quality_fullres.py
import os, pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import tifffile as tiff
from scipy.signal import butter, filtfilt
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

def lpf_zero_phase_linear(A, fs_hz, fc_hz):
    """線形域振幅 A (lines, depths, time) にゼロ位相ローパスを適用（サンプル数はそのまま）"""
    wn = fc_hz / (fs_hz / 2.0)
    b, a = butter(N=4, Wn=wn, btype="low")
    # 時間軸は最後と仮定
    return filtfilt(b, a, A, axis=-1, padtype='odd', padlen=3*max(len(a), len(b)))

if __name__ == "__main__":
    # ---- 設定 ----
    dataset         = "OCT_BRUSH"
    morph_pkl_name  = "morph.pkl"
    save_dir_suffix = "3_analysed"  # 出力先を 2_processed → 3_analysed に自動切替
    save_as         = "morph_quality_fullres.tiff"

    # S/N 改善パラメータ（時間分解能は保持）
    USE_TIME_LPF    = True
    TIME_LPF_CUTOFF = 200.0  # [Hz] 主要帯域上限に合わせて調整（例：200Hz）
    USE_DEPTH_SMOOTH= True
    DEPTH_SIGMA_PX  = 1.0    # 深さ方向の軽いガウシアン（空間分解能への影響が小さい範囲で）

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
        # 線形域で処理：複素→振幅
        A = np.abs(octr.morph.morph).astype(np.float32)   # (lines, depths, time)

        # 時間ローパス（ゼロ位相）
        if USE_TIME_LPF:
            fc = min(TIME_LPF_CUTOFF, 0.45*Fs/2)  # ナイキストに対して安全
            A = lpf_zero_phase_linear(A, fs_hz=Fs, fc_hz=fc)

        # 深さ方向の軽い平滑（空間S/Nを少し上げる：時間・A-lineはそのまま）
        if USE_DEPTH_SMOOTH:
            # (lines, depths, time) → depth 軸=1 にのみガウシアン
            A = gaussian_filter(A, sigma=(0.0, DEPTH_SIGMA_PX, 0.0))

        # dB化（可視化用・保存用）
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A[A < 1e-12] = 1e-12
        dB = 20.0 * np.log10(A)  # 同じ形状

        # TIFF 保存：A-line ごとに 1 スライス（frames, H=depth, W=time）
        stack = np.transpose(dB, (0,1,2))  # (lines, depths, time)
        out_tiff = os.path.join(out_abs, save_as)
        tiff.imwrite(out_tiff, stack.astype(np.float32))
        print("  saved:", out_tiff)
    print(datetime.now(), "done.")
