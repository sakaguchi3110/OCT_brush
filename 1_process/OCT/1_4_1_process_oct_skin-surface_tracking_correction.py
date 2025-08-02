import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end # matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools  # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager  # noqa: E402
from library_python.sensors.OCT.OCTMorph import OCTMorph  # noqa: E402
from library_python.signal_processing.InteractiveVectorEditor import InteractiveVectorEditor  # noqa: E402


def set_up_folders(db_path, input_filename, datatype="OCT_VIB_NEUR", automatic=True, verbose=True):
    """Sets up the input and output folders and retrieves folder names with required files."""
    db_path_input = os.path.join(db_path, datatype, "2_processed", "oct") #os.path.join(db_path, datatype, "2_processed", "oct", "2025-01-15_P01")
    input_foldernames, input_foldernames_abs, input_folder_session_abs = path_tools.get_folders_with_file( # 
        db_path_input, input_filename, automatic=automatic, select_multiple=False
    )
    if verbose:
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
    datatype = "OCT_HAIR-DEFLECTION" # OCT_VIB_NEUR, OCT_HAIR-DEFLECTION
    input_filename = "skin_displacement_estimation.csv"
    output_filename = "skin_displacement_estimation_corrected.csv"

    show = False
    force_processing = True
    save_results = True
    
    # Initialize paths and setup folders
    db_path = path_tools.define_OCT_database_path(data_external_hdd)  # Assumes this is a custom function
    print(f"Path initialized:\ndb_path = '{db_path}'")
    db_path_input, input_foldernames, input_foldernames_abs = set_up_folders(db_path, input_filename, datatype=datatype, automatic=set_path_automatic)
    
    # 2. Extracting scans
    print(datetime.now())
    n_success = 0
    # Initialize a list to hold all the DataFrames
    df_list = []
    for acq_id, input_fn in enumerate(input_foldernames, start=1):
        t = f"Acquisition nº {acq_id}/{len(input_foldernames)}: {input_fn}"
        print(f"{datetime.now()}\t{t}")
        input_folder_abs = input_foldernames_abs[acq_id - 1]
        input_filename_abs = os.path.join(input_folder_abs, input_filename)
        morph_filename_abs = os.path.join(input_folder_abs, "morph.pkl")
        output_filename_abs = os.path.join(input_folder_abs, output_filename)
        if not(force_processing) and os.path.exists(output_filename_abs):
            continue
        
        if os.path.exists(output_filename_abs):
            # df = pd.read_csv(input_filename_abs)
            df = pd.read_csv(output_filename_abs)  # Erase this To avoid multiple moving average
        else:
            df = pd.read_csv(input_filename_abs)
        
        morph = OCTMorph()
        with open(morph_filename_abs, 'rb') as f:
            morph = pickle.load(f)
        morph.get_morph_video()
        
        title = f"Acquisition nº {acq_id}/{len(input_foldernames)}:\n{input_fn}"
        df_out = df.copy()
        for a in range(df.shape[1]):
            vector = df.iloc[:, a]
            img = morph.morph_dB_video[a, :, :]
            
            interface = InteractiveVectorEditor(vector, img=img, title=title)
            plt.show(block=True)
            
            # img_save_path = os.path.join(input_folder_abs, f"OCT_frame_{a+1}.png")
            # # 図を作成してOCT画像のみ描画（軸付き）
            # fig, ax = plt.subplots(figsize=(10, 4))
            # ax.imshow(img, cmap='gray', aspect='auto')
            # # 軸ラベル（必要なら調整）
            # ax.set_xlabel("Index (horizontal pixel)")
            # ax.set_ylabel("Vertical pixel")
            # ax.set_title(f"OCT Frame - Channel {a+1}")
            # plt.tight_layout()
            # fig.savefig(img_save_path, dpi=300, bbox_inches='tight')
            # plt.close(fig)
            # print(f"✅ 画像のみ保存完了: {img_save_path}")
            
            
            cleaned_vector = np.nan_to_num(interface.modified_vector, nan=0.0, posinf=0.0, neginf=0.0)       # all numbers are inverted. 0 means erased line.              
            df_out.iloc[:, a] = cleaned_vector

            zero_cells = df_out == 0
            df_out[df_out == 0] = np.nan  # Replace zeros with NaN
            window_size = 50             # Calculate the moving average while ignoring NaNs
            df_out = df_out.apply(lambda x: x.rolling(window=window_size, min_periods=1).mean())            
            df_out = df_out.apply(lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
            df_out[zero_cells] = 0
            df_out = df_out.astype(int)
        n_success += 1
        
        if show:
            # Plot all columns on the same plot
            df.plot()
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.title(input_filename)
            plt.legend(title='Directories')
            plt.show(block=True)
                        
        if save_results:
            # Create the directory if it doesn't exist
            if not os.path.exists(os.path.dirname(output_filename_abs)):
                os.makedirs(os.path.dirname(output_filename_abs))
                
            ## Confirm surface
            # plt.figure()
            # plt.plot(df_out)
            # plt.legend()
            # plt.show()
            # plt.close()
            # Write to CSV file
            df_out.to_csv(output_filename_abs, index=False)
            print(f"Corrected skin displacement saved in: {output_filename_abs}.")



