import os
from datetime import datetime
from pathlib import Path
import sys

import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end # matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools  # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager2 import OCTRecordingManager  # noqa: E402


def set_up_folders(db_path, dataset="OCT_BRUSH", automatic=True):
    """Sets up the input and output folders and retrieves folder names with required files."""
    db_path_input = os.path.join(db_path, dataset, "1_primary", "oct")
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



if __name__ == "__main__":
    # 0. Initialization of parameters
    data_external_hdd = False
    set_path_automatic = False
    dataset = "OCT_BRUSH" # OCT_VIB_NEUR, OCT_HAIR-DEFLECTION

    force_processing = True
    save_results = True
    
    # Initialize paths and setup folders
    db_path = path_tools.define_OCT_database_path(data_external_hdd)  # Assumes this is a custom function
    print(f"Path initialized:\ndb_path = '{db_path}'")
    db_path_input, input_foldernames, input_foldernames_abs = set_up_folders(db_path, dataset=dataset, automatic=set_path_automatic)

    # 2. Extracting scans
    print(datetime.now())
    n_success = 0
    
    for acq_id, input_fn in enumerate(input_foldernames, start=1):
        t = f"Acquisition nº {acq_id}/{len(input_foldernames)}: {input_fn}"
        print(f"{datetime.now()}\t{t}")
        input_fn_abs = input_foldernames_abs[acq_id - 1]

        output_folder_abs = input_fn_abs.replace("1_primary", "2_processed")

        octr = OCTRecordingManager(input_fn_abs, output_folder_abs, autosave=save_results)
        octr.load_metadata()
        if octr.metadata.isStructural:
            continue
        if octr.exist("morph.pkl")[0] and not(force_processing):
            print("Morph has been previously processed and processing is not forced.")
            continue
        
        octr.compute_morph(force_processing=force_processing, save_hdd=save_results)
        
        n_success += 1

    print(datetime.now())
    print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")

