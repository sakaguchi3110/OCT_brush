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
import csv

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



# Obtain Morph.npy for traced phase change analysis
def load_npy_files(filepaths):
    morph_data = []
    for npfilepath in filepaths:
        try:
            data = np.load(npfilepath)
            morph_data.append(data)
        except FileNotFoundError:
            print(f"File not found: {npfilepath}")
    return morph_data

npname = "morph_matrix.npy"
npfilepaths = [os.path.join(folder, npname) for folder in input_foldernames_abs]
morph_data = load_npy_files(npfilepaths)



# _estimation_corrected.csv
def load_csv_data(corrpaths):
    csv_data_list = []
    for corrpath in corrpaths:
        df = pd.read_csv(corrpath)
        csv_data_list.append(df)
    return csv_data_list

corrpaths = [os.path.join(folder, target_file) for folder in input_foldernames_abs]
csv_data_list = load_csv_data(corrpaths)


def save_phase_change_data(phase_change_data, input_foldernames_abs, filename="phase_change_data_traced.npy"):
    filename_abs = os.path.join(input_foldernames_abs, filename)
    np.save(filename_abs, phase_change_data)
    print(f"Phase change data saved to {filename_abs}")


for slice_morph, csv_data, folder in zip(morph_data, csv_data_list, input_foldernames_abs):
    slice_morph = np.squeeze(slice_morph)
    shift_values = csv_data.iloc[:, 0].values
    slice_morph = np.angle(slice_morph)
    
    # plt.figure()
    # # plt.plot(slice_morph[133, 0:3970]) # 270+ 0, 38pixel
    # # plt.plot(slice_morph[171, 0:3970]) # 270+ 0, 38pixel
    # # plt.plot(slice_morph[133, 6588:10000])
    # # plt.plot(slice_morph[171, 6588:10000])
    # # plt.plot(slice_morph[462, 12671:13671]) # 462+ 0, 38
    # # plt.plot(slice_morph[500, 12671:13671])
    # plt.legend()
    # plt.show()
    # plt.close()
    
    rows, cols = slice_morph.shape
    shifted_matrix = np.full((rows, cols), np.nan)     # Create a new matrix with the same shape
    
    for col in range(cols):
        shift = shift_values[col]
        if shift < rows:
            shifted_matrix[:rows-shift, col] = slice_morph[shift:, col]
    
    # plt.figure()
    # # plt.plot(shifted_matrix[0, 6990:7990], label='surface pixel')
    # # plt.plot(shifted_matrix[100, 6990:7990], label='100 pixel below surface')
    # plt.plot(shifted_matrix[0, 12671:13671], label='surface pixel')
    # plt.plot(shifted_matrix[100, 12671:13671], label='100 pixel below surface')
    # plt.legend()
    # plt.show()
    # plt.close()
    
    # plt.imshow(shifted_matrix[:100, :], cmap='viridis', aspect='auto')
    # plt.colorbar()
    # plt.show()
    # plt.close()

    phasechange = np.diff(shifted_matrix, axis=1)
    phasechange[phasechange > +np.pi] -= 2 * np.pi
    phasechange[phasechange < -np.pi] += 2 * np.pi
    
    plt.figure()
    # plt.plot(phasechange[0, 0:3970], label='surface pixel')
    # plt.plot(phasechange[38, 0:3970], label='38 pixel below surface')
    # plt.plot(phasechange[152, 0:3970], label='152 pixel below surface')
    plt.plot(phasechange[0, 6588:10000], label='surface pixel')
    plt.plot(phasechange[38, 6588:10000], label='38 pixel below surface')
    plt.plot(phasechange[152, 6588:10000], label='152 pixel below surface')
    plt.legend()
    plt.show()
    plt.close()
    
    save_phase_change_data(phasechange, folder)   # save _traced.npy data
