import os
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
import pickle
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.signal import detrend
from skimage.transform import resize
import sys
import tifffile as tiff
from typing import cast

import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end # matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools  # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager  # noqa: E402
from library_python.sensors.OCT.OCTMorph import OCTMorph  # noqa: E402



def load_acquisition(input_fn_abs, output_folder_abs, morphFilename):
    octr = OCTRecordingManager(input_fn_abs, output_folder_abs, autosave=False)
    
    file_path = os.path.join(input_fn_abs, "metadata.pkl")
    with open(file_path, 'rb') as file:
        octr.metadata = pickle.load(file)
    if not octr.metadata.isVibration:
        print("Current recording is not a vibration dataset.")
        return False, None

    file_path = os.path.join(input_fn_abs, morphFilename)
    if not Path(file_path).is_file():
        print("File does not exist.")
        return False, None

    with open(file_path, 'rb') as file:
        octr.morph = cast(OCTMorph, pickle.load(file))  # Explicitly tell the type checker 
    octr.morph.get_morph_img()
    octr.morph.get_morph_video()

    return True, octr



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
        finite_ind = np.isfinite(y)
        
        if np.sum(finite_ind) > 1:  # Ensure there are enough valid points
            m, b, _, _, _ = stats.linregress(x[finite_ind], y[finite_ind])
            detrend_Y[:, i] = y - (m * x + b)
        else:
            detrend_Y[:, i] = np.nan  # If insufficient data, fill with NaN
    
    # Transpose back if axis=1
    if axis == 1:
        detrend_Y = detrend_Y.T
    
    return detrend_Y


def remove_speckle_noise(image, aperture_size=5):
    if aperture_size > 5:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
    else:
        image = image.astype(np.float32)
    denoised_image = cv2.medianBlur(image, aperture_size)
    return denoised_image


def process_image(I, params, verbose=False, show=False):
    steps = {
        "use_depth_cutoff": lambda I: I[params["use_depth_cutoff"]["depth_cutoff_top"]:-1-params["use_depth_cutoff"]["depth_cutoff_bottom"], :],
        "use_resize":       lambda I: resize(I, (round(params["use_resize"]["resize_ratio"] * I.shape), round(params["use_resize"]["resize_ratio"] * I.shape))),
        "use_smoothing":    lambda I: gaussian_filter(I, params["use_smoothing"]["gauss_val"]),
        "use_detrend":      lambda I: nandetrend(I, axis=0),
        "use_median_depth": lambda I: np.nanmedian(I, axis=0),
        "use_median_time":  lambda I: np.nanmedian(I, axis=1),
        "use_fliplr":       lambda I: np.fliplr(I),
        "use_flipud":       lambda I: np.flipud(I),
        "use_normalisation": lambda I: cv2.normalize(np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0), None, 0, 1, cv2.NORM_MINMAX)
    }

    for key, value in params.items():
        if verbose:
            print(f"Key: {key}, Value: {value}")
        if value["enabled"]:
            func = steps.get(key, False)
            if func:
                I = func(I)
            else:
                print("process_image: lambda function name not found")

    if show:
        fig = plt.figure()
        plt.imshow(I)
        fig.show()

    return I


def display_processed_image(I, process_params):
    plt.figure()
    plt.imshow(I)
    plt.title("processed image")

    s = "Process image order:\n"
    cpt = 1
    for key, value in process_params.items():
        if not value['enabled']:
            continue
        if key == 'use_depth_cutoff':
            s += f"{cpt}. Use depth cutoff: {value['enabled']}\n"
            s += f"    Depth cutoff top: {value['depth_cutoff_top']}\n"
            s += f"    Depth cutoff bottom: {value['depth_cutoff_bottom']}\n"
        elif key == 'use_resize':
            s += f"{cpt}. Use resize: {value['enabled']}\n"
            s += f"    Downsample ratio: {value['resize_ratio']:.2f}\n"
        elif key == 'use_smoothing':
            s += f"{cpt}. Use smoothing: {value['enabled']}\n"
            s += f"    Gauss filter: {value['gauss_val']:.1f}\n"
        elif key == 'use_detrend':
            s += f"{cpt}. Detrend: {value['enabled']}\n"
        elif key == 'use_median_depth':
            s += f"{cpt}. Median depth: {value['enabled']}\n"
        elif key == 'use_median_time':
            s += f"{cpt}. Median time: {value['enabled']}\n"
        elif key == 'use_fliplr':
            s += f"{cpt}. Use fliplr: {value['enabled']}\n"
        elif key == 'use_flipud':
            s += f"{cpt}. Use flipud: {value['enabled']}\n"
        elif key == 'use_normalisation':
            s += f"{cpt}. Use normalisation: {value['enabled']}\n"
        cpt += 1
    s = s.replace('\t', '    ')
    print(s)
    plt.text(0.5, 0.5, s, fontsize=12, va='center', ha='left', wrap=True, transform=plt.gca().transAxes)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.xticks([])
    plt.yticks([])
    plt.show(block=True)



if __name__ == "__main__":
    # 0. Initialization of parameters
    data_external_hdd = False
    set_path_automatic = False
    dataset = "OCT_BRUSH"  # OCT_VIB_NEUR, OCT_HAIR-DEFLECTION

    show = False
    
    force_processing = False
    save_results = False
    
    data_to_load = 'morph.pkl'
    downsample = True
    # Fs_target = 9454
    Fs_target = 945.4
    process_params = {
        'use_depth_cutoff': {'enabled': False, 'depth_cutoff_top': 100, 'depth_cutoff_bottom': 0},
        'use_resize':       {'enabled': False, 'resize_ratio': 0.2},
        'use_smoothing':    {'enabled': False, 'gauss_val': 1.0},
        
        'use_detrend': {'enabled': False},
        'use_median_time': {'enabled': False},
        'use_median_depth': {'enabled': False},
        'use_normalisation': {'enabled': True},
        'use_flipud':{'enabled': False},
        'use_fliplr':{'enabled': False},
    }
    
    # Initialize paths and setup folders
    db_path = path_tools.define_OCT_database_path(data_external_hdd)  # Assumes this is a custom function
    print(f"Path initialized:\ndb_path = '{db_path}'")
    db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")
    input_foldernames, input_foldernames_abs, db_path_input = path_tools.get_folders_with_file(
        db_path_input, data_to_load, automatic=set_path_automatic, select_multiple=False
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
    
    # 2. Extracting scans
    print(datetime.now())
    n_success = 0
    
    raw_data_list = []
    
    for acq_id, input_fn in enumerate(input_foldernames, start=0):
        t = f"Acquisition nº {acq_id + 1}/{len(input_foldernames)}: {input_fn}"
        print(f"{datetime.now()}\t{t}")
        input_folder_abs = input_foldernames_abs[acq_id]

        output_folder_abs = input_folder_abs
        img_filename = f'morph_preprocessed_{Fs_target}kHz.tiff'
        if not os.path.exists(output_folder_abs) and save_results:
            os.makedirs(output_folder_abs)
        img_filename_abs = os.path.join(output_folder_abs, img_filename)
        
        if not(force_processing) and os.path.exists(img_filename_abs):
            print(f"Result already exists and not force_processing.")
            continue
        
        success, octr = load_acquisition(input_folder_abs, output_folder_abs, data_to_load)
        if success:
            n_success += 1
        else:
            print(f"Warning: Load_acquisition was not a success.")
            continue
        
        nsample_target = round(octr.morph.get_nsample() * (Fs_target/octr.metadata.Fs_OCT))
        octr.morph.apply_downsample(nsample_target)

        processed_images = []
        for a in np.arange(octr.metadata.n_alines):
            morph_updated = octr.morph.morph_dB_video[a, :, :]
            # raw_data_list.append(morph_updated)  # Store the raw data
        
            I = process_image(morph_updated, process_params)
            processed_images.append(I)  # Store the processed image
            
            if show:
                display_processed_image(morph_updated, process_params)

        if save_results and (force_processing or not(os.path.exists(img_filename_abs))):
            # Convert the list of images to a 3D numpy array
            I_stack = np.stack(processed_images, axis=0)
            # Save the stack of images as a TIFF file
            tiff.imwrite(img_filename_abs, I_stack, dtype=np.float64, compression=None)
    
    print(datetime.now())
    print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")
    
    # 元データを保存する
    # raw_data_filename = 'C:/Users/saisa68/OneDrive - Linköpings universitet/oct/brush/analyzed/2025-04-03_13-37-46_brush3.0_v1p1_/raw_data.pkl'
    # with open(raw_data_filename, 'wb') as file:
    #     pickle.dump(raw_data_list, file)


