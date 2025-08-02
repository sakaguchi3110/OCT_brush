import os
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import matplotlib.colors as mcolors

# Add the path to the library_python module
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager # noqa: E402
from library_python.sensors.OCT.OCTMorph import OCTMorph # noqa: E402


data_external_hdd = False
set_path_automatic = False
dataset = "OCT_BRUSH"
target_file = "skin_displacement_estimation_corrected.csv"

# force_processing = False
save_results = True
show = True

# Initialize paths and setup folders
db_path = path_tools.define_OCT_database_path(data_external_hdd)  # Assumes this is a custom function
db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")
input_foldernames, input_foldernames_abs, _ = path_tools.get_folders_with_file(
    db_path_input, target_file, automatic=set_path_automatic, select_multiple=False, verbose=True
)


def load_phase_change_data(filepaths):
    phase_change_data_list = []
    for filepath in filepaths:
        try:
            if os.path.getsize(filepath) == 0:
                print(f" ⚠️ Skipping empty file: {filepath}")
                continue                
            phase_change_data = np.load(filepath)
            phase_change_data_list.append(phase_change_data)
        except Exception as e:
            print(f" ❌ Failed to load {filepath}: {e}")
            continue            
    return phase_change_data_list

npyname = "phase_change_data.npy"
# print("\n📂 input_foldernames_abs の中身一覧:")
# for folder in input_foldernames_abs:
    # print(f" - {folder}")

filepaths = [os.path.join(folder, npyname) for folder in input_foldernames_abs]
phase_change_data_list = load_phase_change_data(filepaths)


# Define the sampling rate (in Hz)
sampling_rate = 10000

def analyze_frequency(phase_change_data, sampling_rate):
    N = phase_change_data.shape[-1]
    yf = fft(phase_change_data, axis=-1)
    xf = fftfreq(N, 1 / sampling_rate)
    return xf, np.abs(yf)

def calculate_displacement(phase_change_data, wavelength=1300):
    wavelength_m = wavelength * 1e-3     # Convert wavelength from nm to um
    displacement = (phase_change_data * wavelength_m) / (4 * np.pi * 1.38)     # Calculate displacement using the formula: Δd = (Δϕ * λ) / (4πn)
    return displacement

# phase FFT
def plot_displacement_vs_frequency_at_depths(phase_change_data, sampling_rate, dfs, save_path, condname, cover, wavelength=1300):
    titles = ['after_brushing']
    # titles = ['before_brushing', 'after_brushing']
    if cover == 'bare':
        offsets = [2, 20, 160] #[2, 20, 40, 160]
    elif cover == 'tegaderm':
        offsets = [22, 40, 180] #[22, 40, 60, 180]
    else:
        raise ValueError(f"Unknown cover type: {cover}")

    fft_results = {title: {offset: [] for offset in offsets} for title in titles}

    for label, dff in dfs:
        if dff.empty:
            print(f" ⚠️ Skipping {label}: empty DataFrame")
            continue

        if label not in titles:
            print(f" ⚠️ Unknown label: {label}")
            continue

        for offset in offsets:
            dep_indices = dff.iloc[:, 0].astype(int).values + offset
            dep_indices = np.clip(dep_indices, 0, 1023)
            time_indices = dff.index.values
            time_indices = time_indices[time_indices > 0]

            if len(time_indices) == 0:
                continue

            try:
                # depth_data = phase_change_data[0, dep_indices, time_indices - 1]
                depth_data = (
                        phase_change_data[0, dep_indices - 1, time_indices - 1] +
                        phase_change_data[0, dep_indices,     time_indices - 1] +
                        phase_change_data[0, dep_indices + 1, time_indices - 1]
                    ) / 3
            except IndexError as e:
                print(f" ⚠️ Skipping due to indexing error: {e}")
                continue

            if len(depth_data) < 3000:
                print(f" ⚠️ Skipping {label} offset={offset} due to insufficient data length ({len(depth_data)} < 3000)")
                continue
            
            n_samples = depth_data.shape[0]
            window = np.hanning(n_samples)
            window_size = 10
            depth_data = pd.Series(depth_data).rolling(window=window_size, min_periods=1).mean().to_numpy()

            depth_window = depth_data * window
            frequencies = np.fft.fftfreq(n_samples, d=1 / sampling_rate)
            posit_freq = frequencies >= 0
            original_fft = np.abs(np.fft.fft(depth_window, axis=-1))

            fft_results[label][offset].append({
                'frequencies': frequencies[posit_freq],
                'amplitudes': original_fft[posit_freq]
            })
            
            if depth_data.size == 0:
                print(f" ⚠️ No data for offset={offset}, label={label}")
                fft_results[label][offset].append({
                    'frequencies': np.array([]),
                    'amplitudes': np.array([])
                })
                continue

    return fft_results


# _estimation_corrected.csv
def load_csv_data(corrpaths):
    csv_data_list = []
    for corrpath in corrpaths:
        df = pd.read_csv(corrpath)
        csv_data_list.append(df)
    return csv_data_list

corrpaths = [os.path.join(folder, target_file) for folder in input_foldernames_abs]
csv_data_list = load_csv_data(corrpaths)






def process_dataframe(df):
    print("📄 [DEBUG] process_dataframe called")
    print(f" - Type of input: {type(df)}")

    # I DataFrame
    if isinstance(df, pd.DataFrame):
        print(" - Received a DataFrame directly")
        result = split_dataframe(df)
        for label, part_df in result:
            print(f"   ✅ Split: {label} - Length: {len(part_df)}")
        return split_dataframe(df)

    # II list
    elif isinstance(df, list):
        dfs = []
        for i, item in enumerate(df):
            if isinstance(item, pd.DataFrame):
                print(f"   - Item {i} is a DataFrame")
                dfs.extend(split_dataframe(item))
            elif isinstance(item, str):
                print(f"   - Item {i} is a file path: {item}")
                try:
                    loaded_df = pd.read_csv(item)
                    dfs.extend(split_dataframe(loaded_df))
                except Exception as e:
                    print(f"   ⚠️ Failed to read CSV file {item}: {e}")
            else:
                print(f"   ⚠️ Unknown item type in list: {type(item)}")
        return dfs

    # III filepath str
    elif isinstance(df, str):
        print(f" - csv_data is a file path: {df}")
        try:
            loaded_df = pd.read_csv(df)
            return split_dataframe(loaded_df)
        except Exception as e:
            print(f"   ⚠️ Failed to read CSV file: {e}")
            return []

    else:
        print(f" ⚠️ Unknown csv_data format: {df}")
        return []

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

    # before_df = pd.DataFrame()
    after_df = pd.DataFrame()

    # if start_index != -1:
    #     before_index = max(0, start_index - 5001)
    #     before_df = df.iloc[before_index:start_index - 1]

    if end_index != -1:
        after_rows = []
        for i in range(end_index + 1, min(end_index + 5001, len(df))):
            if df.iloc[i, 0] == 0:
                break
            after_rows.append(df.iloc[i])
        after_df = pd.DataFrame(after_rows)

    result = []
    # result.append(("before_brushing", before_df))
    result.append(("after_brushing", after_df))

    return result






def parse_condition_name(name):
    parts = name.lower().split('_')
    return {
        'participant': parts[2],
        'location': parts[4],        # thumb or web
        'texture': parts[6],         # soft or rough
        'cover': parts[5],           # bare or tegaderm
        'frequency': parts[7]        # 0.3, 3.0, or 30
    }

# grouping
# grouped_data = {}
# for idx, cond in enumerate(input_foldernames):
#     parsed = parse_condition_name(cond)
#     key = (parsed['location'], parsed['texture'], parsed['frequency'], parsed['participant'])
#     if key not in grouped_data:
#         grouped_data[key] = {}
#     grouped_data[key][parsed['cover']] = {
#         'phase_change_data': phase_change_data_list[idx],
#         'csv_data': csv_data_list[idx],
#         'condname': cond
#     }
grouped_data = {}

for idx, cond in enumerate(input_foldernames):
    parsed = parse_condition_name(cond)
    key = (parsed['location'], parsed['texture'], parsed['frequency'], parsed['participant'])

    if key not in grouped_data:
        grouped_data[key] = {'bare': [], 'tegaderm': []}

    cover_type = parsed['cover']
    if cover_type in ['bare', 'tegaderm']:
        grouped_data[key][cover_type].append({
            'phase_change_data': phase_change_data_list[idx],
            'csv_data': csv_data_list[idx],
            'condname': cond
        })

# plot
for key, entries in grouped_data.items():
    print(f"\n🔍 Processing key: {key}")

    if 'bare' in entries and 'tegaderm' in entries:
        fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
        # titles = ['before_brushing', 'after_brushing']
        titles = ['after_brushing']
        valid_plot = False

        for row_idx, cover in enumerate(['bare', 'tegaderm']):
            offsets = [2, 20, 160] if cover == 'bare' else [22, 40, 180] #[2, 20, 40, 160][22, 40, 60, 180]
            combined_fft_results = {
                title: {offset: [] for offset in offsets} for title in titles
            }

            for entry in entries[cover]:
                if len(entry['csv_data']) < 3000:
                    print(f" ⚠️ Skipping {cover} {entry['condname']} due to insufficient data length (<3000)")
                    continue

                print(f" ▶ Cover: {cover}, Condition: {entry['condname']}")
                dfs = process_dataframe(entry['csv_data'])

                if all(df.empty for _, df in dfs):
                    print(f" ⚠️ Skipped empty dataframes in: {entry['condname']}")
                    continue

                fft_results = plot_displacement_vs_frequency_at_depths(
                    entry['phase_change_data'],
                    sampling_rate,
                    dfs,
                    save_path=None,
                    condname=entry['condname'],
                    cover=cover
                )

                for condition in titles:
                    for offset in combined_fft_results[condition]:
                        combined_fft_results[condition][offset].extend(fft_results[condition][offset])

            for col_idx, condition in enumerate(titles):
                depths = combined_fft_results[condition].keys()
                norm = mcolors.Normalize(vmin=min(depths), vmax=max(depths))
                cmap = plt.get_cmap('YlOrRd', len(depths))
                colors = [cmap(i) for i in range(len(depths))]
                depth_to_color = {depth: colors[i] for i, depth in enumerate(depths)}

                for offset in depths:
                    spectra = combined_fft_results[condition][offset]
                    spectra = [s for s in spectra if len(s['amplitudes']) > 0]  
                    if not spectra:
                        print(f" ⚠️ No spectra for {condition} at offset {offset}px")
                        continue
                    
                    try:
                        min_len = min(len(s['amplitudes']) for s in spectra)
                        # all_power = np.array([s['amplitudes'][:min_len]**2 for s in spectra])
                        # mean_power = np.mean(all_power, axis=0)
                        all_amp = np.array([s['amplitudes'][:min_len] for s in spectra])
                        mean_amp = np.mean(all_amp, axis=0)
                        frequencies = spectra[0]['frequencies'][:min_len]
                    except Exception as e:
                        print(f" ⚠️ Averaging failed at {condition}-{offset}px: {e}")
                        continue

                    axs[row_idx, col_idx].plot(
                        frequencies,
                        # mean_power,
                        mean_amp,
                        label=f'Depth {offset}px',
                        color=depth_to_color[offset]
                    )
                    valid_plot = True

                axs[row_idx, col_idx].set_title(f"{cover.capitalize()} - {condition}")
                axs[row_idx, col_idx].set_xlabel("Frequency (Hz)")
                axs[row_idx, col_idx].set_xlim(0, 500)
                # axs[row_idx, col_idx].set_ylim(1, 100000)
                # axs[row_idx, col_idx].set_yscale('log')
                axs[row_idx, col_idx].set_ylim(10, 300)
                axs[row_idx, col_idx].grid(True, color='lightgray')
                axs[row_idx, col_idx].legend(fontsize=9, facecolor='dimgray')
                axs[row_idx, col_idx].set_facecolor('dimgray')

        axs[0, 0].set_ylabel("Amplitude (Bare)")
        axs[1, 0].set_ylabel("Amplitude (Tegaderm)")
        fig.suptitle(f"Averaged AMP Spectrum vs Frequency: {key[3]} - {key[0]} - {key[1]} - {key[2]}", fontsize=16)
        plt.tight_layout()
        # plt.show()

        if valid_plot:
            tiffname = f"AMP_spectrum_{key[0]}_{key[1]}_{key[2]}.tiff"
            save_dir = _  
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, tiffname)
            fig.savefig(save_path, dpi=100, format='tiff', facecolor='dimgray')
            print(f" ✅ Figure saved to: {save_path}")
        else:
            print(f" ⚠️ No valid plots generated for key: {key}")
        plt.close(fig)
