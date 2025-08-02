import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy.fft import fft, fftfreq

# ライブラリのパス追加
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager
from library_python.sensors.OCT.OCTMorph import OCTMorph

# === 設定 ===
data_external_hdd = False
set_path_automatic = False
dataset = "OCT_BRUSH"
target_file = "skin_displacement_estimation_corrected.csv"
sampling_rate = 10000
npyname = "phase_change_data.npy"

# === 入力データ読み込み ===
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

# === 条件解析ヘルパー ===
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

def analyze_fft(depth_data, sampling_rate):
    N = len(depth_data)
    window = np.hanning(N)
    depth_data_win = depth_data * window
    yf = fft(depth_data_win)
    xf = fftfreq(N, 1 / sampling_rate)
    posit_freq = xf >= 0
    return xf[posit_freq], np.abs(yf[posit_freq])

def calculate_power(amplitude):
    return amplitude ** 2

# === 分析本体 ===
output_rows = []

for idx, cond in enumerate(input_foldernames):
    parsed = parse_condition_name(cond)
    date_time = f"{parsed['date']}_{parsed['time']}"
    participant = parsed['participant']
    location = parsed['location']
    texture = parsed['texture']
    cover = parsed['cover']
    frequency = parsed['frequency']

    if cover not in ['bare', 'tegaderm']:
        continue

    phase_data = phase_change_data_list[idx]
    if phase_data is None:
        print(f"⚠️ phase_change_data is None for {input_foldernames[idx]}. Skipping.")
        continue

    csv_data = csv_data_list[idx]
    dfs = split_dataframe(csv_data)

    offsets = [2, 20, 40, 160] if cover == 'bare' else [10, 28, 48, 168]

    for label, df_part in dfs:
        if df_part.empty:
            continue

        dep_indices = df_part.iloc[:, 0].astype(int).values
        dep_indices = np.clip(dep_indices, 0, 1023)
        time_indices = df_part.index.values
        time_indices = time_indices[time_indices > 0]

        for offset in offsets:
            adjusted_dep_indices = dep_indices + offset
            adjusted_dep_indices = np.clip(adjusted_dep_indices, 0, 1023)

            depth_data = []
            for d_idx, t_idx in zip(adjusted_dep_indices, time_indices - 1):
                try:    
                    avg_val = (
                        phase_data[0, d_idx - 1, t_idx] +
                        phase_data[0, d_idx,     t_idx] +
                        phase_data[0, d_idx + 1, t_idx]
                    ) / 3
                    depth_data.append(avg_val)
            # depth_data.append(phase_data[0, d_idx, t_idx])
                except IndexError:
                    continue

            depth_data = np.array(depth_data)
            if len(depth_data) < 3000:
                continue

            freqs, amps = analyze_fft(depth_data, sampling_rate)
            mask = (freqs >= 0) & (freqs <= 1000)
            freqs_cut = freqs[mask]
            amps_cut = amps[mask]
            power_cut = calculate_power(amps_cut)

            # === 2Hz刻みで四捨五入し、同じBinは平均 ===
            bin_freqs = np.round(freqs_cut / 2) * 2
            df_fft = pd.DataFrame({'freq': bin_freqs, 'amp': amps_cut, 'power': power_cut})
            df_fft_grouped = df_fft.groupby('freq').mean().sort_index()

            # 行生成
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

            row_amp = base_info.copy()
            row_amp['data_type'] = 'amplitude'
            for f, a in zip(df_fft_grouped.index, df_fft_grouped['amp']):
                row_amp[f'{f:.1f}Hz'] = a

            row_power = base_info.copy()
            row_power['data_type'] = 'power'
            for f, p in zip(df_fft_grouped.index, df_fft_grouped['power']):
                row_power[f'{f:.1f}Hz'] = p

            output_rows.append(row_amp)
            output_rows.append(row_power)

# === DataFrame作成＆保存 ===
df_out = pd.DataFrame(output_rows)

# 保存パス（1つ上の階層へ）
save_root = Path(input_foldernames_abs[0]).parent
participant_all = "_".join(sorted(set(df_out["participant"].unique())))
save_path = save_root / f"fft_summary_{participant_all}.csv"
df_out.to_csv(save_path, index=False)

print(f"✅ FFT summary CSV saved to:\n{save_path}")
