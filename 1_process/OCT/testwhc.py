import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import multiprocessing

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
        print(f"\u26a0\ufe0f Failed to load {f}: {e}")
        phase_change_data_list.append(None)
csv_data_list = [pd.read_csv(f) for f in corrpaths]

# === ヘルパー関数 ===
def parse_condition_name(name):
    parts = name.lower().split('_')
    return {
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

    after_df = pd.DataFrame()
    if end_index != -1:
        after_rows = []
        for i in range(end_index + 1, min(end_index + 5001, len(df))):
            if df.iloc[i, 0] == 0:
                break
            after_rows.append(df.iloc[i])
        after_df = pd.DataFrame(after_rows)

    return [("after_brushing", after_df)]

def analyze_fft(depth_data, sampling_rate):
    N = len(depth_data)
    window = np.hanning(N)
    depth_data_win = depth_data * window
    yf = fft(depth_data_win)
    xf = fftfreq(N, 1 / sampling_rate)
    posit_freq = xf >= 0
    return xf[posit_freq], np.abs(yf[posit_freq])

def process_key(args):
    key, entries = args
    print(f"\n\ud83d\udd0d Processing key: {key}")
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    default_colors = ['C0', 'C1', 'C2']
    valid_plot = False
    
    for row_idx, cover in enumerate(['bare', 'tegaderm']):
        if cover not in entries:
            continue
        offsets = [2, 20, 160] if cover == 'bare' else [22, 40, 180]
        combined_fft_results = {offset: [] for offset in offsets}

        for entry in entries[cover]:
            phase_data = entry['phase_change_data']
            if phase_data is None:
                continue
            dfs = split_dataframe(entry['csv_data'])
            for label, df_part in dfs:
                if df_part.empty:
                    continue
                dep_indices = df_part.iloc[:, 0].astype(int).values
                dep_indices = np.clip(dep_indices, 0, 1023)
                time_indices = df_part.index.values
                time_indices = time_indices[time_indices > 0]
                for i, offset in enumerate(offsets):
                    adj_dep = np.clip(dep_indices + offset, 0, 1023)
                    depth_data = []
                    for d_idx, t_idx in zip(adj_dep, time_indices - 1):
                        try:
                            avg_val = (phase_data[0, d_idx - 1, t_idx] + phase_data[0, d_idx, t_idx] + phase_data[0, d_idx + 1, t_idx]) / 3
                            depth_data.append(avg_val)
                        except IndexError:
                            continue
                    depth_data = np.array(depth_data)
                    if len(depth_data) < 3000:
                        continue
                    freqs, amps = analyze_fft(depth_data, sampling_rate)
                    mask = (freqs >= 0) & (freqs <= 1000)
                    freqs_cut = freqs[mask]
                    amps_cut = amps[mask]
                    combined_fft_results[offset].append((freqs_cut, amps_cut))

        for i, offset in enumerate(offsets):
            spectra = combined_fft_results[offset]
            if not spectra:
                continue
            try:
                min_len = min(len(s[1]) for s in spectra)
                all_amp = np.array([s[1][:min_len] for s in spectra])
                mean_amp = np.mean(all_amp, axis=0)
                freqs = spectra[0][0][:min_len]
                axs[row_idx].plot(freqs, mean_amp, label=f'Depth {offset}px', color=default_colors[i])
                valid_plot = True
            except Exception as e:
                print(f"\u26a0\ufe0f Averaging failed at offset {offset}px: {e}")
                continue

        axs[row_idx].set_title(f"{cover.capitalize()}")
        axs[row_idx].set_xlabel("Frequency (Hz)")
        axs[row_idx].set_xlim(0, 300)
        axs[row_idx].set_ylim(5, 300)
        axs[row_idx].grid(True, color='lightgray')
        axs[row_idx].legend(fontsize=9, facecolor='white')
        axs[row_idx].set_facecolor('white')

    axs[0].set_ylabel("Amplitude (Bare)")
    axs[1].set_ylabel("Amplitude (Tegaderm)")
    fig.suptitle(f"AMP Spectrum: {key[3]} - {key[0]} - {key[1]} - {key[2]}", fontsize=16)
    plt.tight_layout()
    if valid_plot:
        save_dir = Path.home() / "Desktop" / "AMP_spectra"
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / f"whcAMP_spectrum_{key[0]}_{key[1]}_{key[2]}.tiff"
        fig.savefig(save_path, dpi=300, format='tiff', facecolor='white')
        print(f"\u2705 Figure saved to: {save_path}")
        plt.show()
    else:
        print(f"\u26a0\ufe0f No valid plot for {key}")
    plt.close(fig)

# === グルーピング ===
grouped_data = {}
for idx, cond in enumerate(input_foldernames):
    parsed = parse_condition_name(cond)
    key = (parsed['location'], parsed['texture'], parsed['frequency'], parsed['participant'])
    if key not in grouped_data:
        grouped_data[key] = {}
    cover = parsed['cover']
    if cover not in grouped_data[key]:
        grouped_data[key][cover] = []
    grouped_data[key][cover].append({
        'phase_change_data': phase_change_data_list[idx],
        'csv_data': csv_data_list[idx],
        'condname': cond
    })

# === 並列実行 ===
if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        pool.map(process_key, grouped_data.items())
