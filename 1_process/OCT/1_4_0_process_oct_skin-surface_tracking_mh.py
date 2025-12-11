
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import library_python.data_management.path_tools as path_tools
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager
from library_python.sensors.OCT.OCTMorph import OCTMorph

def set_up_folders(db_path, datatype="OCT_VIB_NEUR", automatic=True):
    db_path_input = os.path.join(db_path, datatype, "2_processed", "oct", "1_minimal_processing")
    input_foldernames, input_foldernames_abs, input_folder_session_abs = path_tools.get_folders_with_file(
        db_path_input, "morph.pkl", automatic=automatic, select_multiple=False
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
    data_external_hdd = False
    set_path_automatic = False
    datatype = "OCT_HAIR-DEFLECTION"

    force_processing = True
    show = True
    save_results = False
    save_figure = False
    MAX_ACQS = 12

    db_path = path_tools.define_OCT_database_path(data_external_hdd)
    print(f"Path initialized:\ndb_path = '{db_path}'")
    db_path_input, input_foldernames, input_foldernames_abs = set_up_folders(
        db_path, datatype=datatype, automatic=set_path_automatic
    )
    input_foldernames = input_foldernames[:MAX_ACQS]
    input_foldernames_abs = input_foldernames_abs[:MAX_ACQS]

    print(datetime.now())
    n_success = 0

    for acq_id, input_fn in enumerate(input_foldernames, start=1):
        t = f"Acquisition nº {acq_id}/{len(input_foldernames)}: {input_fn}"
        print(f"{datetime.now()}\t{t}")
        input_folder_abs = input_foldernames_abs[acq_id - 1]
        output_folder_abs = input_folder_abs
        output_filename = "skin_displacement_estimation.csv"
        output_filename_abs = os.path.join(output_folder_abs, output_filename)
        if not(force_processing) and os.path.exists(output_filename_abs):
            continue

        octr = OCTRecordingManager(input_folder_abs, output_folder_abs, autosave=save_results)
        octr.load_metadata(force_processing=False, save_hdd=False, destdir=input_folder_abs)
        if octr.metadata.isStructural:
            continue
        octr.compute_morph(force_processing=False, save_hdd=False, destdir=input_folder_abs, verbose=True)
        octr.morph.get_morph_video()

        depth_offset = 15
        [nalines, ndepths, nsamples] = octr.morph.morph_dB_video.shape

        df = pd.DataFrame()

        for a in range(nalines):
            d = octr.morph.morph_dB_video[a, depth_offset:, :]

            mean = np.mean(d, axis=0, keepdims=True)
            std = np.std(d, axis=0, keepdims=True)
            threshold_low = mean + 0.5 * std
            d[(d < threshold_low)] = 0
            d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
            d = d.astype(np.uint8)
            d = cv2.medianBlur(d, 3)
            d_binary = (d > np.mean(d)).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            d_morph = cv2.morphologyEx(d_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(d_morph, connectivity=8)

            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_area = np.max(areas) if areas.size > 0 else 0
                area_threshold = max_area * 0.01
                d_filtered = np.zeros_like(d_morph)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                        d_filtered[labels == i] = 1
            else:
                d_filtered = d_morph

            expected_skin_locations = np.zeros(d_filtered.shape[1], dtype=int)
            for col in range(d_filtered.shape[1]):
                nonzero_indices = np.nonzero(d_filtered[:, col])[0]
                if len(nonzero_indices) > 0:
                    expected_skin_locations[col] = nonzero_indices[0] + depth_offset
                else:
                    if col > 0:
                        expected_skin_locations[col] = expected_skin_locations[col-1]
                    else:
                        expected_skin_locations[col] = depth_offset

            column_name = f"aline_id{a}"
            df[column_name] = expected_skin_locations.astype(float)

            if show or save_figure:
                fig, axs = plt.subplots(3, 1, figsize=(16, 12))
                im = axs[0].imshow(octr.morph.morph_dB_video[a, :, :], cmap='gray', aspect='auto')
                axs[0].set_title('Original OCT Image')
                axs[0].set_ylabel('Depth (pxl)')
                fig.colorbar(im, ax=axs[0])

                im = axs[1].imshow(d_binary, cmap='gray', aspect='auto')
                axs[1].set_title('Binary Image (Before Noise Filtering)')
                axs[1].set_ylabel('Depth (pxl)')
                fig.colorbar(im, ax=axs[1])

                im = axs[2].imshow(d_filtered, cmap='gray', aspect='auto')
                axs[2].set_title('Filtered Image with Detected Skin Surface')
                axs[2].set_ylabel('Depth (pxl)')
                fig.colorbar(im, ax=axs[2])
                axs[2].plot(expected_skin_locations-depth_offset, color='red', linewidth=2, label='Detected Surface')

                axs[2].legend()
                fig.suptitle(f"{input_folder_abs}: a-line {a}/{nalines}")
                plt.tight_layout()
                if save_figure:
                    output_img = f"_skin-displacement-estimation_figure_a-line-{a}.png"
                    output_img_abs = os.path.join(output_folder_abs, output_img)
                    os.makedirs(os.path.dirname(output_img_abs), exist_ok=True)
                    fig.savefig(output_img_abs, dpi=50, bbox_inches='tight')
                if show:
                    plt.show(block=True)
                plt.close(fig)

        if save_results:
            if not os.path.exists(output_folder_abs):
                os.makedirs(output_folder_abs)
            df.to_csv(output_filename_abs, index=False)
        n_success += 1
        print(datetime.now())
        print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")
