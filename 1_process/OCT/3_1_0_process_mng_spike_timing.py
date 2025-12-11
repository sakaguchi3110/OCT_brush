import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# for popup dialog
import tkinter as tk
from tkinter import simpledialog

# =========================
# 1. Load your data
# =========================

nerve_times = np.loadtxt(r"C:\Users\saisa68\Downloads\nerve_times.txt")
sync_times  = np.loadtxt(r"C:\Users\saisa68\Downloads\sync_times.txt")

nerve_times = np.sort(nerve_times)
sync_times  = np.sort(sync_times)

# --- load vibration start/end table --------------------
vib_table = np.loadtxt(
    r"C:\Users\saisa68\Downloads\vibration_intervals.txt",
    skiprows=1  # skip header: Index Start End
)

vib_start = vib_table[:, 1]
vib_end   = vib_table[:, 2]

large_start = vib_start.copy()
large_end   = vib_end.copy()
# -------------------------------------------------------


# ===== NEW: ask user for clock time of first vibration start =====

first_vib_start = float(large_start[0])  # in seconds from recording start

def ask_start_clock_time():
    """Show popup dialog and return time string 'HH:MM:SS'."""
    root = tk.Tk()
    root.withdraw()  # hide main window
    time_str = simpledialog.askstring(
        "Start time",
        "Enter clock time (HH:MM:SS) for FIRST vibration start"
    )
    root.destroy()
    return time_str

time_str = ask_start_clock_time()

# parse "HH:MM:SS" -> seconds from midnight
h, m, s = time_str.split(":")
h = int(h)
m = int(m)
s = float(s)  # allow decimal seconds

base_clock_sec = h * 3600 + m * 60 + s  # [s] at first_vib_start
# =================================================================


from matplotlib.ticker import FuncFormatter

def sec_to_hhmmss_formatter(first_vib_start, base_clock_sec):
    """Return a FuncFormatter converting x (sec from recording start) to 'HH:MM:SS'."""

    def _formatter(x, pos):
        # x is seconds from recording start
        clock_sec = base_clock_sec + (x - first_vib_start)
        # keep within 0-24h if you like
        clock_sec = clock_sec % (24 * 3600)

        hh = int(clock_sec // 3600)
        mm = int((clock_sec % 3600) // 60)
        ss = clock_sec % 60
        # show 1 decimal place for seconds
        return f"{hh:02d}:{mm:02d}:{ss:04.1f}"

    return FuncFormatter(_formatter)


def to_clock_str_array(times_sec, first_vib_start, base_clock_sec):
    """
    Convert array of times (sec from recording start) into array of 'HH:MM:SS.sss' strings.
    """
    clock_sec = base_clock_sec + (times_sec - first_vib_start)
    clock_sec = np.mod(clock_sec, 24 * 3600)

    def format_one(sec):
        hh = int(sec // 3600)
        mm = int((sec % 3600) // 60)
        ss = sec % 60
        return f"{hh:02d}:{mm:02d}:{ss:06.3f}"

    return np.array([format_one(sec) for sec in clock_sec], dtype=object)



def interactive_select_and_save(nerve_times,
                                large_start, large_end,
                                first_vib_start, base_clock_sec,
                                filename="selected_vibrations.npz"):
    
    n_clusters = len(large_start)
    if n_clusters == 0:
        print("No clusters to show.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    y_bar = 0.0  # just a dummy center (we hide y-axis later)

    # Precompute spikes and syncs in each interval
    cluster_spike_times = []
    for s, e in zip(large_start, large_end):
        spike_in = nerve_times[(nerve_times >= s) & (nerve_times <= e)]
        cluster_spike_times.append(spike_in)

    # ---- NEW: draw shaded yellow rectangles and vertical dotted lines ----
    interval_patches = []  # to change color on click

    for s, e in zip(large_start, large_end):
        # semi-transparent yellow region between Start and End
        patch = ax.axvspan(s, e, alpha=0.3, facecolor="lightgreen")  # default facecolor is ok (yellow-ish in many backends)
        interval_patches.append(patch)

        # vertical dotted lines at Start and End
        ax.vlines(
            [s, e],
            y_bar - 0.7,
            y_bar + 0.7,
            linestyles="dotted",
            linewidth=0.8
        )

    # Draw spikes (as red vertical lines) inside intervals
    for spike_arr in cluster_spike_times:
        for t in spike_arr:
            ax.vlines(
                t,
                y_bar - 0.5,
                y_bar + 0.5,
                linewidth=0.5
                # color is default; you can set here if you want
            )

    ax.set_xlabel("Time from recording start (s)")
    ax.set_yticks([])
    ax.set_title("Click intervals to select (green). Press 's' to save selected intervals.")

    # Keep track of which intervals are selected
    selected = np.ones(n_clusters, dtype=bool)

    def find_interval_index(x):
        """Return index of interval that contains x, or None."""
        idx = np.where((large_start <= x) & (large_end >= x))[0]
        if idx.size == 0:
            return None
        return int(idx[0])

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None:
            return

        x = event.xdata
        idx = find_interval_index(x)
        if idx is None:
            return

        # Toggle selection state
        selected[idx] = ~selected[idx]
        if selected[idx]:
            # selected: change color (e.g., light green)
            interval_patches[idx].set_facecolor("lightgreen")
        else:
            # not selected: back to default yellow-like color
            interval_patches[idx].set_facecolor("lightgray")

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 's':
            included_indices = np.where(selected)[0]
            if included_indices.size == 0:
                print("All intervals are excluded. Nothing to save.")
                return

            print(f"Saving {included_indices.size} intervals to {filename}")

            # numeric values [s]
            large_start_selected = large_start[included_indices]
            large_end_selected   = large_end[included_indices]

            spike_list = [cluster_spike_times[i] for i in included_indices]

            # --- NEW: make clock-time string versions ---------------
            large_start_clock = to_clock_str_array(
                large_start_selected, first_vib_start, base_clock_sec
            )
            large_end_clock = to_clock_str_array(
                large_end_selected, first_vib_start, base_clock_sec
            )

            spike_clock_list = [
                to_clock_str_array(arr, first_vib_start, base_clock_sec)
                for arr in spike_list
            ]

            spike_obj       = np.array(spike_list,       dtype=object)
            spike_clock_obj = np.array(spike_clock_list, dtype=object)
            # --------------------------------------------------------

            np.savez(
                filename,
                # numeric (seconds)
                large_start_selected=large_start_selected,
                large_end_selected=large_end_selected,
                spike_times_list=spike_obj,

                # NEW: clock-time strings
                large_start_clock=large_start_clock,
                large_end_clock=large_end_clock,
                spike_times_clock_list=spike_clock_obj,

                # metadata
                selected_indices=included_indices,
                first_vib_start=first_vib_start,
                base_clock_sec=base_clock_sec,
            )

            print("Saved.")

    ax.xaxis.set_major_formatter(
        sec_to_hhmmss_formatter(first_vib_start, base_clock_sec)
    )
    ax.set_xlabel("Clock time (HH:MM:SS)")
    
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key   = fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    plt.show()

    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)


    # ==============================================
    
    
interactive_select_and_save(
    nerve_times=nerve_times,
    sync_times=sync_times,
    large_start=large_start,
    large_end=large_end,
    first_vib_start=first_vib_start,
    base_clock_sec=base_clock_sec,
    filename="selected_vibration_intervals_with_clock.npz"
)

