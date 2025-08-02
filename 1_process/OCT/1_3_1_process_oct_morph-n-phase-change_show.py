import os
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end # matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools  # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager  # noqa: E402
from library_python.sensors.OCT.OCTMorph import OCTMorph  # noqa: E402



if __name__ == "__main__":
    # 0. Initialization of parameters
    data_external_hdd = False
    set_path_automatic = False
    dataset = "OCT_BRUSH"
    target_file = "phasechange.pkl"

    force_processing = False
    save_results = True
    pack_results = False
    pack_result_folder = "_all_morph_and_phase"

    show = True

    # Initialize paths and setup folders
    db_path = path_tools.define_OCT_database_path(data_external_hdd)  # Assumes this is a custom function
    db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")
    input_foldernames, input_foldernames_abs, _ = path_tools.get_folders_with_file(
        db_path_input, target_file, automatic=set_path_automatic, select_multiple=False, verbose=True
    )
    
    # 2. Extracting scans
    print(datetime.now())
    n_success = 0
    
    for acq_id, (input_folder_abs, input_folder) in enumerate(zip(input_foldernames_abs, input_foldernames), start=0):
        t = f"Acquisition nº {acq_id+1}/{len(input_foldernames)}: {input_folder}"
        print(f"{datetime.now()}\t{t}")
        output_folder_abs = input_folder_abs
        
        octr = OCTRecordingManager(output_folder_abs, output_folder_abs, autosave=save_results)
        octr.load_metadata(force_processing=False, save_hdd=False, destdir=output_folder_abs)
        if not octr.metadata.isVibration:
            continue
        octr.compute_morph(force_processing=False, save_hdd=False, destdir=output_folder_abs, verbose=True)
        octr.morph.get_morph_video()
        octr.compute_phaseChange(force_processing=False, save_hdd=False, destdir=output_folder_abs, verbose=True)
        
        n_success += 1
        nLines = octr.metadata.n_alines
        for a in range(nLines):
            fig, axs = plt.subplots(3, 1, figsize=(16, 9))

            im = axs[0].imshow(octr.morph.morph_dB_video[a, :, :], cmap='gray', aspect='auto')
            axs[0].set_title('Amplitude (dB) Morph Image')
            axs[0].set_ylabel('Depth (pxl)')
            fig.colorbar(im, ax=axs[0])

            im = axs[1].imshow(np.angle(octr.morph.morph[a, :, :]), cmap='gray', aspect='auto')
            axs[1].set_title('Phase Morph Image')
            axs[1].set_ylabel('Depth (pxl)')
            fig.colorbar(im, ax=axs[1])

            im = axs[2].imshow(octr.PChange.phase_change[a, :, :], cmap='gray', aspect='auto')
            axs[2].set_title('Slice Phase Change Image')
            axs[2].set_ylabel('Depth (pxl)')
            fig.colorbar(im, ax=axs[2])

            # Add main title
            fig.suptitle(f'{octr.metadata.session_name}, a-line={a+1}/{nLines}', fontsize=16)
            axs[2].set_xlabel('Time (nsample)')

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title
            

            # 保存用フォルダを定義
            if pack_result_folder:
                directory_abs, leaf_folder = os.path.split(output_folder_abs)
                destdir = os.path.join(directory_abs, pack_result_folder)
            else:
                destdir = output_folder_abs

            if not os.path.exists(destdir):
                os.makedirs(destdir)

            for a in range(nLines):
                # -------------------------
                # Amplitude 画像保存
                # -------------------------
                fig_amp, ax_amp = plt.subplots(figsize=(10, 6))
                im_amp = ax_amp.imshow(octr.morph.morph_dB_video[a, :, :], cmap='gray', aspect='auto')
                ax_amp.axis('on')  # 軸だけ表示（offにすると完全に画像だけ）
                fig_amp.subplots_adjust(left=0, right=1, bottom=0, top=1)  # 余白を全部消す
                amp_path = os.path.join(destdir, f"Amplitude_only_aline-{a+1}.png")
                fig_amp.savefig(amp_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig_amp)

                # -------------------------
                # Phase Change 画像保存
                # -------------------------
                fig_phase, ax_phase = plt.subplots(figsize=(10, 6))
                im_phase = ax_phase.imshow(octr.PChange.phase_change[a, :, :], cmap='gray', aspect='auto')
                ax_phase.axis('on')  # 軸だけ表示（offにすると完全に画像だけ）
                fig_phase.subplots_adjust(left=0, right=1, bottom=0, top=1)  # 余白を全部消す
                phase_path = os.path.join(destdir, f"PhaseChange_only_aline-{a+1}.png")
                fig_phase.savefig(phase_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig_phase)
                        
        
            # Maximize the figure window
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())
            if save_results:
                # filename = output_folder_abs + f"_morph-n-phase_aline-{a+1}.png"
                if pack_result_folder:
                    directory_abs, leaf_folder = os.path.split(output_folder_abs)
                    destdir = os.path.join(directory_abs, pack_result_folder)
                    filename = leaf_folder+f"image_morph-n-phase_aline-{a+1}.png"
                else:
                    destdir = output_folder_abs
                    filename = f"image_morph-n-phase_aline-{a+1}.png"
                filename_abs = os.path.join(destdir, filename)
                if not os.path.exists(destdir):
                    os.makedirs(destdir)
                # Save the figure to a specific path
                fig.savefig(filename_abs)
                

                # phase_change_image = octr.PChange.phase_change[a, :, :]
                # # 画像データを0〜255に正規化して8bit整数に変換
                # phase_change_image_normalized = ((phase_change_image - np.min(phase_change_image)) / (np.max(phase_change_image) - np.min(phase_change_image)) * 255).astype(np.uint8)
                # # PILで画像として保存
                # img = Image.fromarray(phase_change_image_normalized, mode='L')
                # filename_slice = f"slice_phase_change_aline-{a+1}.png"
                # filename_slice_abs = os.path.join(destdir, filename_slice)
                # img.save(filename_slice_abs)

                
            if show:
                plt.show(block=True)
            else:
                # Optionally, you can close the plot to free up memory
                plt.close(fig)


    print(datetime.now())
    print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")

