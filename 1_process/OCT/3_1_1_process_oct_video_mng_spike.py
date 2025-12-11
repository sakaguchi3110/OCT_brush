import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import Listbox, Button, END, SINGLE, messagebox

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager


# ============================
# Helper functions
# ============================

def load_clusters(master_npz_path):
    """
    Load selected_vibration_intervals_with_clock.npz
    and return all necessary components for GUI matching.
    """

    if not os.path.exists(master_npz_path):
        raise FileNotFoundError(master_npz_path)

    data = np.load(master_npz_path, allow_pickle=True)

    # required numeric fields
    large_start = data["large_start_selected"]
    large_end   = data["large_end_selected"]
    spike_list  = data["spike_times_list"]

    # required clock fields
    large_start_clock = data["large_start_clock"]
    large_end_clock   = data["large_end_clock"]
    spike_clock_list  = data["spike_times_clock_list"]

    # labels to display on GUI right side
    labels = large_start_clock  # “HH:MM:SS.sss”

    return (large_start, large_end,
            large_start_clock, large_end_clock,
            spike_list, spike_clock_list, labels)

import tkinter as tk
from tkinter import Listbox, Button, END, SINGLE, messagebox

def match_gui(oct_list, cluster_labels):
    """
    GUI for matching OCT folders (left) and clusters (right).
    User can delete rows or reorder cluster list.
    Returns:
        matched_oct_list
        matched_cluster_idx  (indices into original cluster_labels)
    """

    root = tk.Tk()
    root.title("Match OCT Video folders and clusters")

    lb_oct = Listbox(root, width=60, selectmode="extended")
    lb_clu = Listbox(root, width=30, selectmode="extended")
    lb_oct.grid(row=0, column=0, rowspan=4, padx=5, pady=5)
    lb_clu.grid(row=0, column=1, rowspan=4, padx=5, pady=5)

    for name in oct_list:
        lb_oct.insert(END, name)

    for lbl in cluster_labels:
        lb_clu.insert(END, lbl)

    # track original cluster indices
    idx_map = list(range(len(cluster_labels)))

    # --- buttons ---
    def move_up():
        sel = lb_clu.curselection()
        if not sel: return
        i = sel[0]
        if i == 0: return
        txt = lb_clu.get(i)
        lb_clu.delete(i)
        lb_clu.insert(i-1, txt)
        lb_clu.selection_set(i-1)
        idx_map[i], idx_map[i-1] = idx_map[i-1], idx_map[i]

    def move_down():
        sel = lb_clu.curselection()
        if not sel: return
        i = sel[0]
        if i == lb_clu.size()-1: return
        txt = lb_clu.get(i)
        lb_clu.delete(i)
        lb_clu.insert(i+1, txt)
        lb_clu.selection_set(i+1)
        idx_map[i], idx_map[i+1] = idx_map[i+1], idx_map[i]

    def delete_oct():
        sel = lb_oct.curselection()
        if not sel:
            return
        for i in reversed(sel):
            lb_oct.delete(i)

    def delete_row():
        sel = lb_clu.curselection()
        if not sel:
            return
        for i in reversed(sel):
            lb_clu.delete(i)
            del idx_map[i]

    def confirm():
        if lb_clu.size() != lb_oct.size():
            messagebox.showerror("Error", "Left and right list must have same length")
            return
        root.quit()

    Button(root, text="↑", command=move_up).grid(row=0, column=2)
    Button(root, text="↓", command=move_down).grid(row=1, column=2)
    Button(root, text="Delete", command=delete_row).grid(row=2, column=2)
    Button(root, text="Confirm", command=confirm).grid(row=3, column=2)
    Button(root, text="Delete OCT", command=delete_oct).grid(row=2, column=0)

    root.mainloop()
    
    matched_oct  = [lb_oct.get(i) for i in range(lb_oct.size())]
    matched_idx  = idx_map[:]   # cluster index mapping
    root.destroy()  # ← ここで初めて destroy

    return matched_oct, matched_idx


def save_per_folder_npz(
    oct_rel, oct_abs,
    cluster_indices,
    large_start, large_end,
    large_start_clock, large_end_clock,
    spike_list, spike_clock_list
):
    """
    Save one cluster per OCT folder.
    spike_cluster.npz contains:
        spike_times, spike_times_clock
        start_sec, end_sec
        start_clock, end_clock
        cluster_label  (folder[:19])
    """

    for i, (rel, abspath) in enumerate(zip(oct_rel, oct_abs)):
        idx = cluster_indices[i]

        out_path = os.path.join(abspath, "spike_cluster.npz")
        label = rel[:19]

        np.savez(
            out_path,
            spike_times=np.array(spike_list[idx], float),
            spike_times_clock=np.array(spike_clock_list[idx], dtype=object),
            start_sec=float(large_start[idx]),
            end_sec=float(large_end[idx]),
            start_clock=str(large_start_clock[idx]),
            end_clock=str(large_end_clock[idx]),
            cluster_label=label
        )

        print(f"Saved -> {out_path}  label={label}")


# ============================
# Main
# ============================
if __name__ == "__main__":
    # ============================
    # 0. User parameters
    # ============================
    dataset = "OCT_BRUSH"
    target_file = "phasechange.pkl"
    fs_oct = 10000.0  # OCT sampling = 10 kHz
    n_success = 0
    
    save_images = True  # True or False

    # ========== 1. load OCT video folders ==========
    db_path = path_tools.define_OCT_database_path(False)
    db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")

    oct_rel, oct_abs, _ = path_tools.get_folders_with_file(
        db_path_input, target_file, automatic=False, select_multiple=False, verbose=True
    )

    # keep video folders only
    video_rel = []
    video_abs = []
    for r, a in zip(oct_rel, oct_abs):
        if "video" in r.lower():
            video_rel.append(r)
            video_abs.append(a)

    # sort by name
    sort_idx = np.argsort(video_rel)
    video_rel = [video_rel[i] for i in sort_idx]
    video_abs = [video_abs[i] for i in sort_idx]

    # ========== 2. check if all spike_cluster.npz already exist ==========
    have_all_npz = all(
        os.path.exists(os.path.join(abs_path, "spike_cluster.npz"))
        for abs_path in video_abs
    )

    if have_all_npz:
        print("All spike_cluster.npz found. Skip GUI and master NPZ matching.")
        matched_oct_rel = video_rel
        matched_oct_abs = video_abs
    else:
        print("Some spike_cluster.npz are missing. Loading master NPZ and running GUI...")

        # 2-1. load master NPZ
        master_npz = r"C:\Users\saisa68\Projects\OCT_vib_4Saito\selected_vibration_intervals_with_clock.npz"

        (large_start, large_end,
         large_start_clock, large_end_clock,
         spike_list, spike_clock_list,
         cluster_labels) = load_clusters(master_npz)

        # 2-2. GUI matching
        matched_oct_rel, matched_cluster_idx = match_gui(video_rel, cluster_labels)

        rel2abs = {r: a for r, a in zip(video_rel, video_abs)}
        matched_oct_abs = [rel2abs[r] for r in matched_oct_rel]

        # 2-3. save per-folder NPZ
        save_per_folder_npz(
            matched_oct_rel, matched_oct_abs,
            matched_cluster_idx,
            large_start, large_end,
            large_start_clock, large_end_clock,
            spike_list, spike_clock_list
        )

    # ============================
    # 3. Process acquisitions (use per-folder NPZ)
    # ============================
    for acq_id, (folder_abs, folder_rel) in enumerate(
        zip(matched_oct_abs, matched_oct_rel)
    ):
        print(f"\n=== Acquisition {acq_id+1} / {len(matched_oct_rel)} ===")
        print(folder_rel)

        # ------- Load per-folder NPZ -------
        per_npz_path = os.path.join(folder_abs, "spike_cluster.npz")
        spike_cluster = None
        start_sec = None
        end_sec = None

        if os.path.exists(per_npz_path):
            per = np.load(per_npz_path, allow_pickle=True)
            spike_cluster = np.array(per["spike_times"], float)
            start_sec = float(per["start_sec"])
            end_sec   = float(per["end_sec"])

            if spike_cluster.size == 0:
                print("  -> spike empty, no overlay")
                spike_cluster = None
            else:
                print(f"  -> using {per_npz_path} (label={per.get('cluster_label', 'N/A')})")
        else:
            print("  -> no spike_cluster.npz in this folder, no overlay")
            spike_cluster = None

        # ------- Load OCT -------
        octr = OCTRecordingManager(folder_abs, folder_abs, autosave=False)
        octr.load_metadata(force_processing=False, save_hdd=False, destdir=folder_abs)
        if not octr.metadata.isVibration:
            print("  -> not vibration dataset, skipping")
            continue

        octr.compute_morph(False, False, folder_abs, verbose=True)
        octr.morph.get_morph_video()

        n_success += 1
        nLines = octr.metadata.n_alines

        # Time axis info
        n_time = octr.morph.morph_dB_video.shape[2]   # samples on time axis
        duration_sec = n_time / fs_oct               # seconds

        # ------- Compute spike x positions -------
        spike_positions_x = None
        if (spike_cluster is not None) and (start_sec is not None):
            rel_spike_times = spike_cluster - start_sec
            rel_spike_times = rel_spike_times[
                (rel_spike_times >= 0) & (rel_spike_times <= duration_sec)
            ]
            spike_positions_x = rel_spike_times * fs_oct

        # ============================
        # 4. Process A-lines: PLOT ONLY AMPLITUDE + SPIKES
        # ============================
        for a in range(nLines):
            fig, ax = plt.subplots(1, 1, figsize=(16, 6))

            # Show Amplitude morph only
            amp_img = octr.morph.morph_dB_video[a, :, :]
            im = ax.imshow(amp_img, cmap="gray", aspect="auto")
            fig.colorbar(im, ax=ax)

            ax.set_title(f"Amplitude (dB) Morph Image\n{folder_rel} | A-line {a+1}/{nLines}")
            ax.set_ylabel("Depth (pxl)")
            ax.set_xlabel("Time (sample index @ 10 kHz)")

            # Overlay spike lines at bottom
            if spike_positions_x is not None:
                depth_bottom = amp_img.shape[0] - 1
                depth_top = depth_bottom - max(10, amp_img.shape[0] // 15)

                for x in spike_positions_x:
                    if 0 <= x < n_time:
                        ax.plot([x, x], [depth_bottom, depth_top],
                                color="yellow", linewidth=1.0)

            plt.tight_layout()
            
            if save_images:
                save_path = os.path.join(folder_abs, f"withspike_{a+1:03d}.png")
                fig.savefig(save_path, dpi=150)
                print(f"Saved image -> {save_path}")
            
            plt.show(block=True)
            
    print("\nDone.")
    print(f"{n_success} acquisitions processed.")
