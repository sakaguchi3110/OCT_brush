import os
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end # matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt

import pickle
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools  # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager  # noqa: E402
from library_python.sensors.OCT.OCTMorph import OCTMorph  # noqa: E402

    

def plot_displacement_vs_frequency_at_depths(phase_change_data, sampling_rate, dfs, save_path, condname, wavelength=1300):  # 1300 nm (central wavelength)
    titles = ['before_brushing', 'after_brushing']  #   titles = ['before_brushing', 'right_after_brushing', 'after_brushing']
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    # peak_values = []

    for i, dff in enumerate(dfs):
        for offset in [0, 38, 152]:  # 0, 38 2.63 um/pixel: 0, 100, 400(152), 1000(380p) um from surface approximately 2.7 mm (Max imaging depth for air(3.5 mm) & skin(2.54 mm))
            depth_data = np.zeros(dff.shape[0])
            dep_indices = dff.iloc[:, 0].astype(int).values + offset
            dep_indices = np.clip(dep_indices, 0, 1023)  # Ensure values do not exceed 1024
            time_indices = dff.index.values

            depth_data = phase_change_data[0, dep_indices, time_indices - 1]
            n_samples = depth_data.shape[-1]
            window = np.hanning(n_samples)

            window_size = 10  # Calculate the moving average while ignoring NaNs
            depth_data = pd.Series(depth_data).rolling(window=window_size, min_periods=1).mean().to_numpy()
            frequencies = np.fft.fftfreq(n_samples, d=1 / sampling_rate)
            posit_freq = frequencies >= 0
            depth_window = depth_data * window
            original_fft = np.abs(np.fft.fft(depth_window, axis=-1))
            axs[i].plot(frequencies[posit_freq], original_fft[posit_freq], label=f'Depth {offset} pixel')
            # find peak
            peak_index = np.argmax((original_fft[posit_freq]))
            peak_value = original_fft[posit_freq][peak_index]
            peak_freq = frequencies[posit_freq][peak_index]

        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_title(f'{titles[i]} _( {n_samples} points)')
        axs[i].legend(fontsize=10)
        axs[i].grid(True)
        # axs[i].set_xscale('log')
        axs[i].set_xlim(10, 300)
        # axs[i].set_yscale('log')
        axs[i].set_ylim(10, 500)
    
    axs[i].set_ylabel('Amplitude')
    fig.suptitle(f'Amplitude vs Frequency ({condname})', fontsize=16)
    plt.tight_layout()
    plt.show()


# morph FFT
def plot_displacement_vs_frequency_at_surface(sampling_rate, dfs, save_path_surf, condname):
    titles = ['before_brushing', 'after_brushing']  #   titles = ['before_brushing', 'right_after_brushing', 'after_brushing']
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for i, dff in enumerate(dfs):
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
        axs[i].set_xlim(0, 100)
        # axs[i].set_yscale('log')
        axs[i].set_ylim(0, 10000)
    
    axs[0].set_ylabel('Amplitude')
    fig.suptitle(f'Surface Displacement vs Frequency ({condname})', fontsize=16)
    plt.tight_layout()
    # plt.savefig(save_path_surf)
    plt.show()



def load_csv_data(corrpaths):
    csv_data_list = []
    for corrpath in corrpaths:
        df = pd.read_csv(corrpath)
        csv_data_list.append(df)
    return csv_data_list


def process_dataframe(df):
    # Initialize variables
    consecutive_zeros = 0
    start_index = -1
    end_index = -1

    # divide into 2 parts
    for i in range(len(df)):
        if df.iloc[i, 0] == 0:
            consecutive_zeros += 1
            if consecutive_zeros == 10 and start_index == -1:
                start_index = i - 9
        else:
            if consecutive_zeros >= 10:
                end_index = i
                break
            consecutive_zeros = 0

    # Ensure start_index and end_index are valid
    if start_index == -1 or end_index == -1:
        raise ValueError("10 consecutive zeros not found in the DataFrame.")

    # Ensure before_index is not less than 0
    before_index = max(0, start_index - 5001)
    before_df = df.iloc[before_index:start_index - 1]

    # Extract 5000 rows after the end_index
    after_df = []
    for i in range(end_index + 1, min(end_index + 5001, len(df))):
        if df.iloc[i, 0] == 0:
            break
        after_df.append(df.iloc[i])
    after_df = pd.DataFrame(after_df)
    dfs = [before_df, after_df]

    return dfs


if __name__ == "__main__":
    # 0. Initialization of parameters
    data_external_hdd = False
    set_path_automatic = False
    dataset = "OCT_BRUSH"
    target_file = "skin_displacement_estimation_corrected.csv"

    force_processing = True
    save_results = False

    # Initialize paths and setup folders
    db_path = path_tools.define_OCT_database_path(data_external_hdd)  # Assumes this is a custom function
    db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")
    input_foldernames, input_foldernames_abs, _ = path_tools.get_folders_with_file(
        db_path_input, target_file, automatic=set_path_automatic, select_multiple=False, verbose=True
    )

    # 2. Extracting scans
    print(datetime.now())
    n_success = 0

    phase_change_data_list = []

    for acq_id, input_fn in enumerate(input_foldernames, start=1):
        t = f"Acquisition nº {acq_id}/{len(input_foldernames)}: {input_fn}"
        print(f"{datetime.now()}\t{t}")
        input_folder_abs = input_foldernames_abs[acq_id - 1]

        output_folder_abs = input_folder_abs

        octr = OCTRecordingManager(input_folder_abs, output_folder_abs, autosave=save_results)
        octr.load_metadata(force_processing=False, save_hdd=False, destdir=input_folder_abs)
        if not octr.metadata.isVibration:
            continue
        octr.compute_morph(force_processing=False, save_hdd=False, destdir=input_folder_abs, verbose=True)
        octr.compute_phaseChange(force_processing=force_processing, save_hdd=save_results)
        phase_change_data = octr.PChange.phase_change
        phase_change_data_list.append(phase_change_data)

        n_success += 1

    print(datetime.now())
    print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")

    # Define the sampling rate (in Hz)
    sampling_rate = 10000

    # Calculate surface pixel number from _estimation.csv
    corrpaths = [os.path.join(folder, target_file) for folder in input_foldernames_abs]
    csv_data_list = load_csv_data(corrpaths)

    # Process each dataframe and plot the results
    for phase_change_data, csv_data, folder, cond in zip(phase_change_data_list, csv_data_list, input_foldernames_abs, input_foldernames):
        tiffname = "disp_vs_freq_500hz_amp_nonl.tiff"
        tiffname_surf = "surfdisp_vs_freq_500hz_amp_nonl.tiff"
        dfs = process_dataframe(csv_data)
        save_path = os.path.join(folder, tiffname)
        save_path_surf = os.path.join(folder, tiffname_surf) # save_path = os.path.join(folder, "disp_vs_freq.tiff")
        condname = cond
        plot_displacement_vs_frequency_at_depths(phase_change_data, sampling_rate, dfs, save_path, condname)
        plot_displacement_vs_frequency_at_surface(sampling_rate, dfs, save_path_surf, condname)
