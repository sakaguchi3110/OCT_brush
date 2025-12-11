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


import pandas as pd




import numpy as np
import matplotlib.pyplot as plt


def plot_phase_debug(octr, morph, dep_indices, time_indices, offset, a_line=0, depth=400):
    """
    morph: complex OCT morph data, shape (nLines, nDepth, nSamples)
    a_line: which A-line to inspect
    depth: which depth pixel to inspect
    """
    morph = np.array(octr.morph.morph)            # complex morph
    amp_img_full = np.array(octr.morph.morph_dB_video[a_line])  # ★正しい振幅画像

    # 1) raw phase (wrapped)
    raw_phase = np.angle(morph[a_line, depth, :])  # [-π, +π]

    # 2) unwrapped phase
    unwrapped_phase = np.unwrap(raw_phase)         # continuous

    # 3) phase change (difference)
    phase_change = np.diff(unwrapped_phase)        # Δφ(t)

    # ---------------------------
    nLines, nDepth, nSamples = morph.shape

    # 深さトラッキング（CSVのdepth列 + offset）
    moving_depth = dep_indices.astype(int) + int(offset)
    moving_depth = np.clip(moving_depth, 0, nDepth - 1)

    # 時間インデックス（CSVの index をそのまま時間として使う前提）
    moving_time = time_indices.astype(int)
    moving_time = np.clip(moving_time, 0, nSamples - 1)

    # 長さをそろえる
    n = min(len(moving_depth), len(moving_time))
    moving_depth = moving_depth[:n]
    moving_time  = moving_time[:n]
    wrapped_tracking = np.angle(morph[a_line, moving_depth, moving_time])
    unwrapped_tracking = np.unwrap(wrapped_tracking)        # shape = [n]

    # 5) tracking Δφ
    delta_tracking = np.diff(unwrapped_tracking)  
    moving_time_rel = moving_time - moving_time[0]
    # ---------------------------
    # 6段プロット
    # ---------------------------
    fig, axs = plt.subplots(6, 1, figsize=(11, 16), sharex=True)

    # 0) Amplitude map
    # amp = np.abs(morph[a_line])

    im0 = axs[0].imshow(amp_img_full, cmap='gray', aspect='auto')
    axs[0].set_title("Amplitude image")
    axs[0].set_ylabel("Depth(px)")
    # fig.colorbar(im0, ax=axs[0])

    # 固定深さの水平ライン（青）
    axs[0].axhline(y=depth, color='blue', lw=2)

    # tracking 深さのライン（オレンジ）
    axs[0].plot(moving_time_rel, moving_depth, color='orange', lw=2)

    # 1) fixed wrapped
    axs[1].plot(raw_phase, color='blue')
    axs[1].set_title(f"Wrapped Phase (fixed depth={depth})")
    axs[1].set_ylabel("φ [rad]")

    # 2) fixed unwrapped
    axs[2].plot(unwrapped_phase, color='blue')
    axs[2].set_title("Unwrapped Phase (fixed)")
    axs[2].set_ylabel("φ_unwrap [rad]")

    # 3) fixed Δφ
    axs[3].plot(phase_change, color='blue')
    axs[3].set_title("Phase Change Δφ (fixed)")
    axs[3].set_ylabel("Δφ [rad]")

    # 4) tracking unwrapped（オレンジ）
    axs[4].plot(unwrapped_tracking, color='orange')
    axs[4].set_title(f"Tracking Unwrapped Phase (offset={offset})")
    axs[4].set_ylabel("φ_unwrap_trk [rad]")

    # 5) tracking Δφ（オレンジ）
    axs[5].plot(delta_tracking, color='orange')
    axs[5].set_title("Tracking Δφ (from unwrapped tracking)")
    axs[5].set_ylabel("Δφ_trk [rad]")
    axs[5].set_xlabel("Time index")
    
    
    plt.tight_layout()
    plt.show(block=True)
    plt.savefig("debug_phase.png", dpi=200)








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
        octr.morph.get_morph_video()

        octr.compute_phaseChange(force_processing=True, save_hdd=save_results)
        phase_change_data = octr.PChange.phase_change
        phase_unwrap_data = octr.PChange.phase_unwrapped
        
        save_phase_change_data(phase_change_data, output_folder_abs) # save .npy data
        
        csv_path = input_folder_abs + "/skin_displacement_estimation_corrected.csv"
        df_csv = pd.read_csv(csv_path)

        # Depth tracking
        dep_indices = df_csv.iloc[:, 0].astype(int).values
        time_indices = df_csv.index.values
        offset = 160
        
        n_success += 1
        
        plot_phase_debug(
            octr=octr,
            morph=octr.morph.morph,
            dep_indices=dep_indices,
            time_indices=time_indices,
            offset=offset,
            a_line=0,
            depth=400
        )



    print(datetime.now())
    print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")
    
    


    

    

