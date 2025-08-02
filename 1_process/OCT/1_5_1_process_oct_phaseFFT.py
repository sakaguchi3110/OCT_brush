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
        phase_change_data = np.load(filepath)
        phase_change_data_list.append(phase_change_data)
    return phase_change_data_list

npyname = "phase_change_data.npy"
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
def plot_displacement_vs_frequency_at_depths(phase_change_data, sampling_rate, dfs, save_path, condname, wavelength=1300): #1300 nm (central wavelength)
    titles = ['before_brushing', 'after_brushing']  #   titles = ['before_brushing', 'right_after_brushing', 'after_brushing']
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for i, dff in enumerate(dfs):
        if dff.empty:
            print("Skip")
            continue
        
        # plt.figure()
        for offset in [0, 40, 160]: # 2.5um  #0, 38 2.63 um/pixel: 0, 100, 400(152), 1000(380p) um from surface   approximately 2.7 mm (Max imaging depth for air(3.5 mm) & skin(2.54 mm))
            depth_data = np.zeros(dff.shape[0])
            dep_indices = dff.iloc[:, 0].astype(int).values + offset
            dep_indices = np.clip(dep_indices, 0, 1023) # Ensure values do not exceed 1024
            time_indices = dff.index.values
            # plt.figure()
            # plt.plot(phase_change_data[0,:,:])
            # plt.plot(time_indices, dep_indices, 'o')
            # plt.xlim(0, phase_change_data.shape[2])
            # plt.ylim(0, 1024)
            # plt.show()
            # plt.close()
            
            depth_data = phase_change_data[0, dep_indices, time_indices-1]
            # displacement = calculate_displacement(depth_data, wavelength)
            n_samples = depth_data.shape[-1]
            window = np.hanning(n_samples)
            # disp_window = displacement * window

        
        
            window_size = 10             # Calculate the moving average while ignoring NaNs
            depth_data = pd.Series(depth_data).rolling(window=window_size, min_periods=1).mean().to_numpy()            
            # df_out = df_out.astype(int)
                   
            # plt.plot(depth_data)

            frequencies = np.fft.fftfreq(n_samples, d=1/sampling_rate)
            posit_freq = frequencies >= 0
            # displacement_spectrum = np.abs(np.fft.fft(disp_window, axis=-1))
            # axs[i].plot(frequencies[posit_freq], displacement_spectrum[posit_freq], label=f'Depth {offset} pixel')
            # # axs[i].plot(frequencies[:n_samples // 2], displacement_spectrum[:n_samples // 2], label=f'Depth {offset} pixel')

            depth_window = depth_data * window
            original_fft = np.abs(np.fft.fft(depth_window, axis=-1))
            axs[i].plot(frequencies[posit_freq], original_fft[posit_freq], label=f'Depth {offset} pixel')
        # plt.ylim(-3.2, 3.2)
        # plt.show()
        # plt.close()
        
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_title(f'{titles[i]} _( {n_samples} points)')
        axs[i].legend(fontsize=10)
        axs[i].grid(True)
        # axs[i].set_xscale('log')
        axs[i].set_xlim(0, 1000)
        # axs[i].set_yscale('log')
        axs[i].set_ylim(0, 500)

    # axs[0].set_ylabel('Displacement (um)')
    axs[0].set_ylabel('Amplitude')
    fig.suptitle(f'Amplitude vs Frequency ({condname})', fontsize=16)
    # fig.suptitle(f'Displacement vs Frequency ({condname})', fontsize=16)
    plt.tight_layout()
    # plt.savefig(save_path, dpi=50)
    plt.show()

# morph FFT
def plot_displacement_vs_frequency_at_surface(sampling_rate, dfs, save_path_surf, condname):
    titles = ['before_brushing', 'after_brushing']  #   titles = ['before_brushing', 'right_after_brushing', 'after_brushing']
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for i, dff in enumerate(dfs):
        if dff.empty:
            print("Skip")
            continue
        
        signal = dff.iloc[:, 0]
        length = len(signal)
        wind = np.hanning(length)
        sig_wind = signal * wind
        ffe_res = np.fft.fft(sig_wind)
        morpfreq = np.fft.fftfreq(len(signal), d=1/sampling_rate)
        positive_freq = morpfreq >= 0
        axs[i].plot(morpfreq[positive_freq], np.abs(ffe_res)[positive_freq]) 
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_title(f'{titles[i]} _( {length} points)')
        axs[i].legend()
        axs[i].grid(True)
        # axs[i].set_xscale('log')
        axs[i].set_xlim(0, 1000)
        axs[i].set_yscale('log')
        axs[i].set_ylim(0, 30000)
    
    axs[0].set_ylabel('Amplitude')
    fig.suptitle(f'Surface Displacement vs Frequency ({condname})', fontsize=16)
    plt.tight_layout()
    # plt.savefig(save_path_surf)
    plt.show()


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
    # Initialize variables
    consecutive_zeros = 0
    start_index = -1
    end_index = -1
    
    # # Find the first occurrence of 10 consecutive zeros and the first non-zero value after that
    # # divide into 3 parts
    # for i in range(len(df)):
    #     if df.iloc[i, 0] == 0:
    #         consecutive_zeros += 1
    #         if consecutive_zeros == 10 and start_index == -1:
    #             start_index = i - 9
    #     else:
    #         if consecutive_zeros >= 10:
    #             end_index = i
    #             break
    #         consecutive_zeros = 0

    # # Ensure before_index is not less than 0
    # before_index = max(0, start_index - 3001)
    # before_df = df.iloc[before_index:start_index - 1]

    # # Extract 3000 rows after the first value that follows the 10 consecutive zeros
    # right_after_df = df.iloc[end_index + 1:end_index + 3001]

    # # Extract the next 3000 rows after 'Right_after'
    # after_index = end_index + 3001
    # after_df = df.iloc[after_index + 1:after_index + 3001]

    # dfs = [before_df, right_after_df, after_df]

    # return dfs


    # divide into 2 parts
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

    return [before_df, after_df]



# Process each dataframe and plot the results
for phase_change_data, csv_data, folder, cond in zip(phase_change_data_list, csv_data_list, input_foldernames_abs, input_foldernames):
    dfs = process_dataframe(csv_data)
    if all(df.empty for df in dfs):
        print("Skip")
        continue
    tiffname = "disp_vs_freq_500hz_amp_nonl.tiff"
    tiffname_surf = "surfdisp_vs_freq_500hz_amp_nonl.tiff"
    # save_path = [os.path.join(folder, tiffname) for folder in input_foldernames_abs]
    save_path = os.path.join(folder, tiffname) # save_path = os.path.join(folder, "disp_vs_freq.tiff")
    save_path_surf = os.path.join(folder, tiffname_surf) # save_path = os.path.join(folder, "disp_vs_freq.tiff")
    condname = cond
    plot_displacement_vs_frequency_at_depths(phase_change_data, sampling_rate, dfs, save_path, condname)
    # plot_displacement_vs_frequency_at_surface(sampling_rate, dfs, save_path_surf, condname)
