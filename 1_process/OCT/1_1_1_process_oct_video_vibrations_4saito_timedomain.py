import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import cv2
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable





def plot_amplitude_and_displacement(octr, a, payload, *, mode="frame",
                                    save=True, show=False, out_path=None, value_max=None, low_depth_threshold=0.1):
    """mode: 'disp' (変位RMS) or 'diff' (差分RMS)"""
    if mode == "disp":
        t_vis = payload["t_vis_disp"]
        vs_vis = payload["vs_vis_disp"]
        depth_vis = payload["depth_vis_disp"]
    elif mode == "diff":
        t_vis = payload["t_vis_diff"]
        vs_vis = payload["vs_vis_diff"]
        depth_vis = payload["depth_vis_diff"]
    elif mode == "frame":
        t_vis = payload["t_vis"]
        vs_vis = payload["vs_vis"]
        depth_vis = payload["depth_vis"]
    """上：振幅（色=深さ, 明るさ=振幅） 下：|Δ変位|（深さ色） をx軸共有で描画"""
    t_min, t_max   = int(payload["t_min"]), int(payload["t_max"])
    depth_min      = int(payload["depth_min"])
    depth_range    = tuple(payload.get("depth_range", (depth_min, depth_min + 1)))
    y_lim          = tuple(payload.get("y_lim", (0, 3)))
    cmap_name      = payload.get("cmap_name", "viridis")

    # 可視化オブジェクト（payload から作る）
    import matplotlib.cm as cm, matplotlib.colors as mcolors
    cmap       = cm.get_cmap(cmap_name)
    norm_depth = mcolors.Normalize(vmin=depth_range[0], vmax=depth_range[1])
    sm_depth   = cm.ScalarMappable(cmap=cmap, norm=norm_depth)

    x_range = (t_min, t_max)


    # 上段：振幅(dB)→『色=深さ、明るさ=振幅』に合成
    amp_dB = np.array(octr.morph.morph_dB_video[a])            # (Z,T)
    amp_slice = amp_dB[depth_min:, t_min:t_max+1]              # depth>=depth_min
    vmin, vmax = np.nanpercentile(amp_slice, [5, 99])
    amp_norm = np.clip((amp_slice - vmin)/max(1e-9, (vmax - vmin)), 0, 1)   # 明るさ0..1
    depth_rows = np.arange(depth_min, depth_min + amp_slice.shape[0])
    row_colors = cmap(norm_depth(depth_rows))[:, :3]           # (Zd, 3)
    rgb = row_colors[:, None, :] * amp_norm[..., None]         # (Zd, Tspan, 3)



    # 図作成（x共有）
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # 上：振幅（深さ0が上になるようextentで反転）
    im0 = ax0.imshow(
        rgb, aspect="auto",
        extent=[t_min, t_max, depth_min + rgb.shape[0] - 1, depth_min]
    )
    ax0.set_ylabel("Depth (px)")
    ax0.set_title("Amplitude (hue=depth, value=amplitude)")
    ax0.grid(False)
    ax0.set_xlim(*x_range)

    # 下段：線色＝深さ
    # line_colors = cmap(norm_depth(depth_vis))
    # 右カラーバー（深さ）。上下で同じものを付ける
    # for ax in (ax0, ax1):
    #     div = make_axes_locatable(ax)
    #     cax = div.append_axes("right", size="2.5%", pad=0.05)
    #     cbar = fig.colorbar(sm_depth, cax=cax, orientation='vertical')
    #     cbar.set_label("Depth position (px)")
    #     cbar.ax.invert_yaxis()  # 上=浅い(200), 下=深い(1000)

    # 右カラーバー（深さ）は上段のみ
    div0 = make_axes_locatable(ax0)
    cax0 = div0.append_axes("right", size="2.5%", pad=0.05)
    cbar0 = fig.colorbar(sm_depth, cax=cax0, orientation='vertical')
    cbar0.set_label("Depth position (px)")
    cbar0.ax.invert_yaxis()

    import copy
    cmap_img = copy.copy(cmap)
    cmap_img.set_bad(alpha=0.0)

    # フル深さサイズ（上段の元画像から取得）
    amp_dB = np.array(octr.morph.morph_dB_video[a])  # (Z, T)
    depth_full = amp_dB.shape[0]

    # vs_vis(K, T') をフル深さ H(Z, T') に敷き詰め（±4px）
    if len(depth_vis) == 0 or len(t_vis) == 0:
        ax1.set_visible(False)
    else:
        Tspan = len(t_vis)
        H = np.full((depth_full, Tspan), np.nan, dtype=np.float32)
        half = 4  # 8pxビンの±4px帯

        for k, c in enumerate(depth_vis):
            y0 = max(0, int(c - half))
            y1 = min(depth_full, int(c + half) + 1)
            H[y0:y1, :] = vs_vis[k][None, :]

        # ------ ここから追加（2Dマスク：全深さ×全タイムポイント）------
        # amp_norm は (Zd, T_full_in_slice) = (depth_min以降, t_min..t_max の連続フレーム)
        # 下段 H の列は t_vis（間引きされた時間）なので、列対応を取ってからマスクを適用する
        thr = float(low_depth_threshold)            # 例: 0.1
        Zd, Tfull = amp_norm.shape                  # Zd = amp_slice の深さ数
        # t_vis(グローバル) → amp_norm内(ローカル: 0..Tfull-1)に写像
        col_idx = (t_vis - t_min).astype(int)
        col_idx = np.clip(col_idx, 0, Tfull - 1)    # 念のため範囲ガード

        # amp_norm の (Zd, Tspan) を作る（t_visに合わせて列抽出）
        mask_local_2d = amp_norm[:, col_idx] >= thr   # True=残す, False=消す 形状 (Zd, Tspan)

        # これを full depth に展開
        mask_full_2d = np.ones_like(H, dtype=bool)    # (depth_full, Tspan) 初期は全部残す
        start = depth_min
        end   = depth_min + Zd
        mask_full_2d[start:end, :] = mask_local_2d

        # マスク適用：False の場所を NaN（透明）に
        H[~mask_full_2d] = np.nan

        # カラースケール：value_max があれば固定、なければ5–99%でロバスト
        if value_max is not None:
            vmin, vmax = 0.0, float(value_max)
        else:
            vmin, vmax = np.nanpercentile(H, [5, 99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = float(np.nanmin(H)), float(np.nanmax(H))

        x0, x1 = int(t_vis[0]), int(t_vis[-1])

        # --- use 'cividis' colormap for the lower heatmap ---
        cmap_disp = plt.get_cmap("jet").copy()
        cmap_disp.set_bad(alpha=0.0)
        im1 = ax1.imshow(
            H, aspect="auto", origin="upper", cmap=cmap_disp,
            extent=[x0, x1, depth_full-1, 0], vmin=vmin, vmax=vmax,
            interpolation="nearest"
        )
        
        ax1.set_xlabel("Time index")
        ax1.set_ylabel("Depth (px)")
        ax1.set_xlim(t_min, t_max)     # 上段と共有
        ax1.set_ylim(depth_full-1, 0)  # 上=浅い, 下=深い
        ax1.grid(False)
        ax1.set_title("|Δ⟨disp⟩| heatmap (adjacent time windows, 8px depth bins)")

        # 値カラーバー（下段のみ）
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes("right", size="2.5%", pad=0.05)
        cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
        cbar1.set_label("|Δ⟨displacement⟩| (px), W=10, STEP=5")

        
    # # 下：|Δ変位| 線群
    # for k, c in enumerate(line_colors):
    #     ax1.plot(t_vis, vs_vis[k], color=c, lw=1, alpha=0.9)
    # ax1.set_xlabel("Time index")
    # ax1.set_ylabel("|Δ Displacement| (px/frame)")
    # ax1.set_ylim(*y_lim)
    # ax1.set_xlim(*x_range)
    # ax1.grid(True, alpha=0.3)
    # ax1.set_title("Reliable motion magnitude (lines)")

    fig.tight_layout()

    if save:
        if out_path is None:
            out_path = f"_displacement_a-line-{a}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=True)
    plt.close(fig)


# Add the path to the library_python module
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools1 as path_tools  # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager1 import OCTRecordingManager  # noqa: E402
from library_python.sensors.OCT.OCTMorph import OCTMorph  # noqa: E402

def set_up_folders(db_path, dataset="OCT_BRUSH", automatic=True):
    """Sets up the input and output folders and retrieves folder names with required files."""
    db_path_input = os.path.join(db_path, dataset, "1_primary", "oct")
    input_foldernames, input_foldernames_abs, input_folder_session_abs = path_tools.get_folders_with_file(
        db_path_input, "Measurement.srm", automatic=automatic, select_multiple=False
    )
    return db_path_input, input_foldernames, input_foldernames_abs

if __name__ == "__main__":
    # 0. Initialization of parameters
    data_external_hdd = False
    set_path_automatic = False
    dataset = "OCT_BRUSH"  # OCT_VIB_NEUR, OCT_HAIR-DEFLECTION
    datatype = "OCT_HAIR-DEFLECTION"  # OCT_VIB_NEUR, OCT_HAIR-DEFLECTION

    force_processing = False
    save_results = True
    show = False
    save_figure = True
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

        octr.compute_morph(force_processing=force_processing, save_hdd=False)  # save_results

        # Process the morph data without saving
        depth_offset = 15
        [nalines, ndepths, nsamples] = octr.morph.morph_dB_video.shape

        # Initialize an empty DataFrame to hold all the data
        df = pd.DataFrame()

        for a in range(nalines):
            d = octr.morph.morph_dB_video[a, depth_offset:, :]
            # Calculate the mean and standard deviation along depth
            mean = np.mean(d, axis=0, keepdims=True)
            std = np.std(d, axis=0, keepdims=True)
            # Define a threshold for noise, for example, mean ± 2*std
            threshold_low = mean + 1 * std
            # Set pixels that are considered noise to 0
            d[(d < threshold_low)] = 0
            # Normalize the image to the range 0-255
            d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
            # Convert the normalized image to uint8 type
            d = d.astype(np.uint8)
            # Apply a median blur to remove small particles
            d = cv2.medianBlur(d, 5)
            # Create a binary image and convert boolean to integer (0 or 1)
            d = (d > np.mean(d)).astype(np.uint8)

            d_flipped = np.flipud(d)
            expected_skin_locations = np.argmax(d_flipped == 1, axis=0) + depth_offset
            expected_skin_locations = 1024 - expected_skin_locations + depth_offset

            # Use the directory name as the column name
            column_name = f"aline_id{a}"
            # Add this column to the large DataFrame
            df[column_name] = expected_skin_locations

            if show or save_figure:
                p = getattr(octr.morph, "plot_payload", None)
                if p is None:
                    raise RuntimeError("plot_payload がありません（compute_morph 内で作成してください）")

                out_img = os.path.join(output_folder_abs, f"_displacement_a-line-{a}.png")
                plot_amplitude_and_displacement(octr, a, p, mode="frame", save=save_figure, show=show, out_path=out_img, value_max=2.0, low_depth_threshold=0.1)


        if save_results:
            output_filename = "skin_displacement_estimation.csv"
            output_filename_abs = os.path.join(output_folder_abs, output_filename)
            if not os.path.exists(output_folder_abs):
                os.makedirs(output_folder_abs)
            # Write to CSV file
            df.to_csv(output_filename_abs, index=False)

        n_success += 1

    print(datetime.now())
    print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")



