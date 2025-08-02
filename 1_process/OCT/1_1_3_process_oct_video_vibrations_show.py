import os
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
import pickle
import sys
from skimage.transform import resize
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.signal import detrend
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end # matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools  # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager  # noqa: E402
from library_python.sensors.OCT.OCTMorph import OCTMorph  # noqa: E402
from library_python.visualisation.visualise3D import visualise3D  # noqa: E402


def nandetrend(Y, axis=0):
    """
    Detrends the input matrix `Y` along the specified axis while ignoring NaN values.

    Parameters:
        Y (numpy.ndarray): Input data (2D array).
        axis (int): Axis along which to detrend (0 for rows, 1 for columns).

    Returns:
        numpy.ndarray: Detrended data with the same shape as `Y`.
    """
    # Ensure axis is valid
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")
    
    # Transpose data if necessary to always process columns
    if axis == 1:
        Y = Y.T

    # Create a time or sample index array
    x = np.arange(Y.shape[0])
    
    # Initialize the detrended matrix
    detrend_Y = np.empty_like(Y)
    
    # Loop over each column to detrend
    for i in range(Y.shape[1]):
        y = Y[:, i]
        not_nan_ind = ~np.isnan(y)
        
        if np.sum(not_nan_ind) > 1:  # Ensure there are enough valid points
            m, b, _, _, _ = stats.linregress(x[not_nan_ind], y[not_nan_ind])
            detrend_Y[:, i] = y - (m * x + b)
        else:
            detrend_Y[:, i] = np.nan  # If insufficient data, fill with NaN
    
    # Transpose back if axis=1
    if axis == 1:
        detrend_Y = detrend_Y.T
    
    return detrend_Y


def initialize_paths(data_external_hdd):
    """Initializes paths and sets up the database path."""
    db_path = path_tools.define_OCT_database_path(data_external_hdd)  # Assumes this is a custom function
    print(f"Path initialized:\ndb_path = '{db_path}'")
    return db_path



def set_up_folders(db_path, dataset="OCT_BRUSH", target_file="morph.pkl", automatic=True):
    """Sets up the input and output folders and retrieves folder names with required files."""
    db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")
    input_foldernames, input_foldernames_abs, _ = path_tools.get_folders_with_file(
        db_path_input, target_file, automatic=automatic, select_multiple=False
    )

    print("------------------")
    print("Input acquisitions of interest (absolute):")
    print(input_foldernames_abs)
    print("Acquisitions of interest:")
    print(input_foldernames)
    print("Total number of acquisitions:")
    print(len(input_foldernames))
    print("done.")
    print("------------------")
    
    return db_path_input, input_foldernames, input_foldernames_abs



def plot_surf(data, subtitle):
    nlines = data.shape[0]
    for a in range(nlines):
        d = data[a, :, :]
        x = np.arange(0, d.shape[0])
        y = np.arange(0, d.shape[1])
        X, Y = np.meshgrid(x, y)

        fig  = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot the surface
        surf = ax.plot_surface(X, Y, np.transpose(d), cmap='viridis')
        # Add color bar
        fig.colorbar(surf)
        plt.title(subtitle)


def plot_subplot(data, octr, params, subtitle, save_results=False, output_folder_abs=''):
    [nlines, ndepths, nsamples] = data.shape
    plt.figure(1)
    plt.clf()
    for a in range(nlines):
        plt.subplot(1, nlines+1, a+1)
        plt.imshow(data[a, :, :], aspect='auto')
        plt.xticks([])
        plt.yticks([])
        if type == 0:
            plt.title(f"A-line {a + 1}")
    plt.subplot(1, nlines + 1, nlines + 1)
    s = (f"processing steps:\n"
         f"1. Depth selected: {params['depth_cutoff_top']}:{params['depth_cutoff_bottom']}\n"
         f"2. Downsample ratio: {params['downsample_ratio']:.2f}\n"
         f"3. Gauss filter: {params['gauss_val']:.1f}\n"
         f"4. Use detrend: {params['use_detrend']}\n"
         f"5. Use fliplr: {params['use_fliplr']}\n"
         f"6. Use normalisation: {params['use_normalisation']}")
    plt.text(0.1, 0.5, s, fontsize=12)
    plt.axis('off')
    plt.suptitle(subtitle)
    if save_results:
        plt.savefig(f"{output_folder_abs}/morph_processed.png", dpi=600)
    else:
        plt.show()


def load_acquisition(input_fn_abs, output_folder_abs, morphFilename):
    """Processes a single acquisition and saves the morphological data as a JPEG."""
    octr = OCTRecordingManager(input_fn_abs, output_folder_abs, autosave=False)
    
    file_path = os.path.join(input_fn_abs, "metadata.pkl")
    with open(file_path, 'rb') as file:
        octr.metadata = pickle.load(file)
    if not octr.metadata.isVibration:
        return False, None

    file_path = os.path.join(input_fn_abs, morphFilename)
    if Path(file_path).is_file():
        with open(file_path, 'rb') as file:
            octr.morph = pickle.load(file)
        octr.morph.get_morph_img()
        octr.morph.get_morph_video()
    else:
        print("File does not exist.")

    return True, octr

def modify_morph(octr, params, show=False):
    A = octr.morph.morph_dB_video[:, params["depth_cutoff_top"]:-1-params["depth_cutoff_bottom"], :]
    A = octr.morph.morph_dB_video
    if show:
        plt.imshow(A[0,:,:])
        plt.show(block=False)
    nlines, ndepth, nsample = A.shape
    if params["use_resize"]:
        new_ndepth = round(params["downsample_ratio"] * ndepth)
        new_nsample = round(params["downsample_ratio"] * nsample)
        data_processed = np.zeros((nlines, new_ndepth, new_nsample))
    else:
        data_processed = np.zeros((nlines, ndepth, nsample))

    for a in range(nlines):
        d = A[a, :, :]
        if params["use_resize"]:
            d = resize(d, (new_ndepth, new_nsample))
        if params["use_smoothing"]:
            d = gaussian_filter(d, params["gauss_val"])
        if params["use_detrend"]:
            d = nandetrend(d, axis=0)

        if params["use_fliplr"]:
            d = np.fliplr(d)
        if params["use_normalisation"]:
            d = cv2.normalize(d, None, 0, 1, cv2.NORM_MINMAX)
        data_processed[a, :, :] = d
    
    return data_processed


if __name__ == "__main__":
    # 0. Initialization of parameters
    data_external_hdd = False
    set_path_automatic = False
    force_processing = False
    save_results = True

    dataset = "OCT_BRUSH" # OCT_VIB_NEUR, OCT_HAIR-DEFLECTION
    
    downsample = False
    downsample_method = 3
    ndownsample = 2940
    
    plot_method = 2
    res = '-r600'
    data_to_load = 'morph.pkl'
    
    process_params = {
        'use_resize': True,
        'downsample_ratio': 0.2,
        'depth_cutoff_top': 100,
        'depth_cutoff_bottom': 0,
        'use_smoothing': False,
        'gauss_val': 1.0,
        'use_detrend': True,
        'use_fliplr': True,
        'use_normalisation': True
    }
    
    # Initialize paths and setup folders
    db_path = initialize_paths(data_external_hdd)
    db_path_input, input_foldernames, input_foldernames_abs = set_up_folders(db_path, dataset=dataset, target_file=data_to_load, automatic=set_path_automatic)

    # 2. Extracting scans
    print(datetime.now())
    n_success = 0
    
    for acq_id, input_fn in enumerate(input_foldernames, start=1):
        t = f"Acquisition nº {acq_id + 1}/{len(input_foldernames)}: {input_fn}"
        print(f"{datetime.now()}\t{t}")
        input_fn_abs = input_foldernames_abs[acq_id - 1]
        output_folder_abs = input_fn_abs.replace("2_processed", "3_analysed")

        success, octr = load_acquisition(input_fn_abs, output_folder_abs, data_to_load)
        if not success:
            print('Something went wrong while loading the file.')
            continue
        n_success += 1

        morph_modified = modify_morph(octr, process_params, show=False)
        subtitle = f"{octr.metadata.session_name} [{data_to_load}]"
        if plot_method == 1:
            visualise3D(np.transpose(morph_modified, (2, 0, 1)), title_base=subtitle)
        elif plot_method == 2:
            print(f"plot_surf")
            plot_surf(morph_modified, subtitle)
        elif plot_method == 3:
            print(f"plot_subplot")
            plot_subplot(morph_modified, octr, process_params, subtitle)
        plt.show(block=False)
            

    print(datetime.now())
    print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")

