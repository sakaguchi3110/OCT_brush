import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import cv2
import tkinter as tk
from tkinter import filedialog
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end # matplotlib.use('gtk3agg')

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from library_python.sensors.OCT.OCTRecordingManager1 import OCTRecordingManager  # noqa: E402
from library_python.sensors.OCT.OCTMorph import OCTMorph  # noqa: E402

def set_up_folders(db_path, csv_path, ds):
    """Sets up the input and output folders and retrieves folder names with required files."""
    input_foldernames = []
    input_foldernames_abs = []

    parent_dir = csv_path.parent.parent
    data_oct_dir = parent_dir / "data_oct"        
    
    for index, row in ds.iterrows(): # define interval
        trial_number = row['trial_number']
        if len(str(trial_number)) == 1:
            trial_number = f"0{trial_number}"
        block_number = csv_path.stem.split('_')[4]
        target_interval = row['target_interval']            
        pattern = f"**/*{block_number}*trial-*{trial_number}*"     
        for trial_folder in data_oct_dir.glob(pattern):
            # print(f"Found trial folder: {trial_folder}")
            input_foldernames.append(trial_folder.name)         
            interval_folder = f"interval{target_interval}"       
            full_path = os.path.join(db_path, "data_oct", trial_folder, interval_folder)
            input_foldernames_abs.append(full_path)
                  
    return input_foldernames, input_foldernames_abs


def set_up_folders_csv(db_path):
    """Sets up the input and output folders and retrieves folder names with required files."""
    
    csv_paths = []
    
    dss = []
    data_dir = Path(db_path) / "data"
    
    for csv_path in data_dir.glob("**/*_results.csv"): #"**/*block-01*_results.csv"
        ds = pd.read_csv(csv_path)
        offset = [0, 38, 152]
        for offset in offset:
            ds[f'Offset_{offset}_Peak_Value'] = pd.NA
            ds[f'Offset_{offset}_Peak_Freq'] = pd.NA
        csv_paths.append(csv_path)
        dss.append(ds)
                  
    return csv_paths, dss


def plot_displacement_vs_frequency_at_depths(phase_change_data, sampling_rate, dfs, input_fn, wavelength=1300):  # 1300 nm (central wavelength)
    plt.figure(figsize=(18, 6))
    peak_values = []

    for dff in dfs:
        for offset in [0, 38, 152]:  # 0, 38 2.63 um/pixel: 0, 100, 400(152), 1000(380p) um from surface approximately 2.7 mm (Max imaging depth for air(3.5 mm) & skin(2.54 mm))
            depth_data = np.zeros(dff.shape[0])
            dep_indices = dff.iloc[:, 0].astype(int).values - offset
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
            # plt.plot(frequencies[posit_freq], original_fft[posit_freq], label=f'Depth {offset} pixel')

            peak_index = np.argmax((original_fft[posit_freq]))            # find peak
            peak_value = original_fft[posit_freq][peak_index]
            peak_freq = frequencies[posit_freq][peak_index]

            peak_values.append((offset, peak_value, peak_freq))

    # plt.xlabel('Frequency (Hz)')
    # plt.legend(fontsize=10)
    # plt.grid(True)
    # plt.xlim(0, 1000)
    # plt.ylabel('Amplitude')
    # plt.title(f'Amplitude vs Frequency ({input_fn})', fontsize=16)
    # plt.tight_layout()
    # plt.show()
    return peak_values

def select_directories():
    root = tk.Tk()
    root.withdraw()
    db_paths = []

    for _ in range(2): #######################CHANGE################
        dir_path = filedialog.askdirectory(title="Select Folder")
        if dir_path:
            db_paths.append(dir_path)
            print(f"Selected folder: {dir_path}")

    print("All selected directories:", db_paths)
    return db_paths

if __name__ == "__main__":
    # 0. Initialization of parameters
    force_processing = False
    save_results = False
    show = False
    save_figure = False

    root = tk.Tk()
    root.withdraw()
    # db_path = filedialog.askdirectory(title="Select Folder")
    # print(f"Selected folder: {db_path}")
    db_paths = select_directories()


    for db_path in db_paths:
        # Initialize paths and setup folders
        csv_paths, dss = set_up_folders_csv(db_path)

        # 2. Extracting scans
        print(datetime.now())
        n_success = 0


        for layer_id, (csv_path, ds) in enumerate(zip(csv_paths, dss), start=1):

            input_foldernames, input_foldernames_abs = set_up_folders(db_path, csv_path, ds)
            all_peak_values = []

            for acq_id, (input_fn, input_fn_abs) in enumerate(zip(input_foldernames, input_foldernames_abs), start=1):
                t = f"Acquisition nº {acq_id}/{len(input_foldernames)}: {input_fn}"
                print(f"{datetime.now()}\t{t}")

                output_folder_abs = input_fn_abs.replace("1_primary", "2_processed")

                octr = OCTRecordingManager(input_fn_abs, output_folder_abs, autosave=save_results)
                octr.load_metadata()
                if octr.metadata.isStructural:
                    continue
                if octr.exist("morph.pkl")[0] and not(force_processing):
                    print("Morph has been previously processed and processing is not forced.")
                    continue

                octr.compute_morph(force_processing=force_processing, save_hdd=False)  # save_results

                # Process the morph data without saving
                depth_offset = 15
                [nalines, ndepths, nsamples] = octr.morph.morph_dB_video.shape

                # Initialize an empty DataFrame to hold all the data
                df = pd.DataFrame()
                
                for a in range(nalines):
                    d = octr.morph.morph_dB_video[a, depth_offset:, :]
                    # Calculate the mean and standard deviation along depth
                    mean = np.mean(d, axis=0, keepdims=True)
                    std = np.std(d, axis=0, keepdims=True)
                    # Define a threshold for noise, for example, mean ± 2*std
                    threshold_low = mean + 1 * std
                    # Set pixels that are considered noise to 0
                    d[(d < threshold_low)] = 0
                    # Normalize the image to the range 0-255
                    d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
                    # Convert the normalized image to uint8 type
                    d = d.astype(np.uint8)
                    # Apply a median blur to remove small particles
                    d = cv2.medianBlur(d, 5)
                    # Create a binary image and convert boolean to integer (0 or 1)
                    d = (d > np.mean(d)).astype(np.uint8)

                    d_flipped = np.flipud(d)
                    expected_skin_locations = np.argmax(d_flipped == 1, axis=0) + depth_offset
                    expected_skin_locations = 1024 - expected_skin_locations + depth_offset

                    # Use the directory name as the column name
                    column_name = f"aline_id{a}"
                    # Add this column to the large DataFrame
                    df[column_name] = expected_skin_locations

                    if show or save_figure:
                        fig, axs = plt.subplots(2, 1, figsize=(16, 9))
                        im = axs[0].imshow(octr.morph.morph_dB_video[a, :, :], cmap='gray', aspect='auto')
                        axs[0].set_title('Initial')
                        axs[0].set_ylabel('Depth (pxl)')
                        fig.colorbar(im, ax=axs[0])

                        im = axs[1].imshow(d, cmap='gray', aspect='auto')
                        axs[1].set_title('Processed')
                        axs[1].set_ylabel('Depth (pxl)')
                        fig.colorbar(im, ax=axs[1])
                        # Plot the locations of the first value equal to 1 in each column as a red line
                        axs[1].plot(expected_skin_locations - depth_offset, color='red')
                        fig.suptitle(f"{input_fn_abs}: a-line {a}/{nalines}")
                        if save_figure:
                            output_img = Path(output_folder_abs) / f"_skin-displacement-estimation_figure_a-line-{a}.png"
                            output_img.parent.mkdir(parents=True, exist_ok=True)
                            fig.savefig(output_img_abs, dpi=300, bbox_inches='tight')  # Use dpi=300 for high-quality images
                        if show:
                            plt.show(block=True)

                # if save_results:
                #     output_filename = "skin_displacement_estimation.csv"
                #     output_filename_abs = Path(output_folder_abs) / output_filename
                #     output_filename_abs.parent.mkdir(parents=True, exist_ok=True)
                #     df.to_csv(output_filename_abs, index=False)

                force_processing = True
                save_results = False

                # 2. Extracting scans
                n_success = 0
                phase_change_data_list = []
                # input_folder_abs = input_foldernames_abs[acq_id - 1]

                # output_folder_abs = input_folder_abs

                # octr = OCTRecordingManager(input_folder_abs, output_folder_abs, autosave=save_results)
                # octr.load_metadata(force_processing=False, save_hdd=False, destdir=input_folder_abs)
                # if not octr.metadata.isVibration:
                #     continue
                # octr.compute_morph(force_processing=False, save_hdd=False, destdir=input_folder_abs, verbose=True)
                octr.compute_phaseChange(force_processing=force_processing, save_hdd=save_results)
                phase_change_data = octr.PChange.phase_change
                phase_change_data_list.append(phase_change_data)

                # Define the sampling rate (in Hz)
                sampling_rate = 147000

                min_val = df.min().min()
                datasurface = {'value': [min_val] * len(df)}
                dff = pd.DataFrame(datasurface)
                dfs = [dff]

                # Process each dataframe and plot the results
                # for phase_change_data, dff, folder, cond in zip(phase_change_data_list, dfs, input_foldernames_abs, input_foldernames):
                    # tiffname = "disp_vs_freq_500hz_amp_nonl.tiff"
                # save_path = os.path.join(folder, tiffname)
                match = re.search(r'trial-(\d+)', input_fn_abs)
                if match:
                    trial_number1 = int(match.group(1))
                else:
                    raise ValueError(f"not found trial_number")
                
                peak_values = plot_displacement_vs_frequency_at_depths(phase_change_data, sampling_rate, dfs, input_fn)
                all_peak_values.append((trial_number1, peak_values))
                                    
                n_success += 1

            for trial_number1, peak_values in all_peak_values:
                row_index = ds.index[ds['trial_number'] == trial_number1].tolist()
                if not row_index:
                    raise ValueError(f"not found trial_number")                
                row_index = row_index[0]
            
                for i, (offset, peak_value, peak_freq) in enumerate(peak_values):
                    col_index = 10 + 2 * i
                    ds.iat[row_index, col_index] = peak_value
                    ds.iat[row_index, col_index + 1] = peak_freq

            output_folder_abs_ = ('\\').join(output_folder_abs.split('\\')[:5])
            output_csv_path = Path(output_folder_abs_) / f"{csv_path.stem}_freq.csv"
            ds.to_csv(output_csv_path, index=False)

            print(datetime.now())
            print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")

    # save peakdata in csv