import os
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end # matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools  # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager  # noqa: E402
from library_python.sensors.OCT.OCTMorph import OCTMorph  # noqa: E402

def save_phase_change_data(phase_change_data, output_folder_abs, filename="phase_change_data.npy"):
    filename_abs = os.path.join(output_folder_abs, filename)
    np.save(filename_abs, phase_change_data)
    print(f"Phase change data saved to {filename_abs}")

if __name__ == "__main__":
    # 0. Initialization of parameters
    data_external_hdd = False
    set_path_automatic = False
    dataset = "OCT_BRUSH" 
    target_file = "morph.pkl"

    force_processing = True
    save_results = True
    
    # Initialize paths and setup folders
    db_path = path_tools.define_OCT_database_path(data_external_hdd)  # Assumes this is a custom function
    db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")
    input_foldernames, input_foldernames_abs, _ = path_tools.get_folders_with_file(
        db_path_input, target_file, automatic=set_path_automatic, select_multiple=False, verbose=True
    )
    
    # 2. Extracting scans
    print(datetime.now())
    n_success = 0
    
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
        octr.compute_phaseChange(force_processing=True, save_hdd=save_results)
        phase_change_data = octr.PChange.phase_change
        save_phase_change_data(phase_change_data, output_folder_abs) # save .npy data
        
        n_success += 1


    print(datetime.now())
    print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")
    
    


    

    

