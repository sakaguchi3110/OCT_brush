import os
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
import re
import sys

import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end # matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools  # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager  # noqa: E402




def set_up_folders(db_path, datatype="OCT_BRUSH", automatic=True):
    """Sets up the input and output folders and retrieves folder names with required files."""
    db_path_input = os.path.join(db_path, datatype, "1_primary", "oct")
    input_foldernames, input_foldernames_abs, input_folder_session_abs = path_tools.get_folders_with_file(
        db_path_input, "Measurement.srm", automatic=automatic, select_multiple=False
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



def process_acquisition(input_fn, input_fn_abs, use_fliplr=True, save_results=True, force_processing=True):
    """Processes a single acquisition and saves the morphological data as a JPEG."""
    output_folder_abs = Path(str(input_fn_abs).replace("1_primary", "2_processed"))
    print(f"{datetime.now()}\tAcquisition nº {input_fn}")
    
    octr = OCTRecordingManager(input_fn_abs, output_folder_abs, autosave=False)
    octr.load_metadata(force_processing=force_processing, save_hdd=False)
    if not octr.metadata.isStructural:
        print(f"{datetime.now()}\t not a structural image.")
        return False
    
    # Load and compute the morphological image/video
    octr.compute_morph(force_processing=force_processing, save_hdd=False)
    # data = 20 * np.log10(np.abs(octr.morph.morph[:, 100:950]))
    if use_fliplr:
        data = np.fliplr(octr.morph.morph_dB_img).T
    else:
        data = octr.morph.morph_dB_img

    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (65535 * (data - data_min) / (data_max - data_min)).astype(np.uint16)
    # Create a PIL image from the normalized array
    high_res_image = Image.fromarray(normalized_data, mode='I;16')
    
    # if it is part of a vibration recording part, 
    # add the frequency information for clarity to facilitate further processing
    splits = re.split(r'[\\/]', str(octr.scan_folder_processed))
    
    if splits[-2].isdigit():
        path_filename = "/".join(splits[:-2])
        filename = f"{splits[-1]}_{splits[-2]}Hz"
        filename_abs = os.path.join(path_filename, filename)
    else:
        filename_abs = str(octr.scan_folder_processed)

    dupl_id = 1
    while os.path.isfile(filename_abs):
        dupl_id += 1
        filename_abs = f"{filename_abs}_{dupl_id}"
    n = output_folder_abs.parent / output_folder_abs.name / output_folder_abs.name
    filename_abs = f"{filename_abs}.jpeg"
    output_folder_abs_str = f"{n}.jpeg"
    print(filename_abs)

    # Save the matrix as a JPEG
    if save_results:
        high_res_image.save(output_folder_abs_str, format='TIFF', dpi=(300, 300))
        # filename_abs_npy = f"{filename_abs}.npy"
        # np.save(filename_abs_npy, normalized_data)
        #cv2.imwrite(filename_abs, matrix_uint8)
    print(f"{datetime.now()}\tAcquisition finished.")
    print("-------------------\n")
    
    return True




if __name__ == "__main__":
    # 0. Initialization of parameters
    data_external_hdd = False
    set_path_automatic = False
    datatype = "OCT_BRUSH" # OCT_VIB_NEUR, OCT_HAIR-DEFLECTION

    force_processing = True
    save_results = True

    # processing parameters
    use_fliplr = False

    # Initialize paths and setup folders
    db_path = path_tools.define_OCT_database_path(data_external_hdd)
    print(f"Path initialized:\ndb_path = '{db_path}'")
    db_path_input, input_foldernames, input_foldernames_abs = set_up_folders(db_path, datatype=datatype, automatic=set_path_automatic)
    
    # 2. Extracting scans
    print(datetime.now())
    n_success = 0
    
    for acq_id, input_fn in enumerate(input_foldernames, start=1):
        input_fn_abs = input_foldernames_abs[acq_id - 1]
        success = process_acquisition(input_fn, input_fn_abs, use_fliplr, save_results, force_processing)
        if success:
            n_success += 1
    
    print(datetime.now())
    print(f"{n_success}/{len(input_foldernames)} acquisitions have been processed.")

