import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import time
# import tkinter as tk # tkinter is not directly used here, can be removed if not needed elsewhere

class InteractiveVectorEditor:
    def __init__(self, vector, img=None, title="", verbose=False): # Set img default to None
        self.vector = vector
        self.modified_vector = vector.copy()
        self.img = img if img is not None else [] # Ensure self.img is a list or array
        self.title = title
        self.verbose = verbose

        self.press = None
        self.zoom_mode = False
        self.currentMode = 'Pen'
        # Corrected typo: Eraser
        self.modes = ('Pen', 'Eraser')

        self.fig, self.ax_img = plt.subplots()
        if self.title:
            self.fig.suptitle(self.title)

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid_zoom = self.fig.canvas.mpl_connect('scroll_event', self.on_zoom)

        self.ax_signal = self.ax_img.twinx()

        self.rect = None
        self.start = None
        self.end = None
        self.history = []
        self.pen_xy = np.empty((0, 2), int)

        self.pen_thickness = 3
        # Initialize the yellow pen line (remains persistent)
        self.line, = self.ax_signal.plot([], [], 'y-', linewidth=self.pen_thickness, zorder=10) # High zorder

        # --- Store handle for the main signal plot ---
        # Initialize the main blue signal line (remains persistent)
        self.signal_line, = self.ax_signal.plot([], [], color='blue', zorder=5)

        # --- Button Creation ---
        # Adjusted positions slightly for potentially better layout
        ax_save = self.fig.add_axes([0.75, 0.01, 0.15, 0.06]) # Use fig.add_axes for better control
        self.btn_save = Button(ax_save, 'Save & Close')
        self.btn_save.on_clicked(self.save_and_close)
        self.btn_save.label.set_color('white')
        self.btn_save.color = 'green'
        self.btn_save.hovercolor = 'darkgreen' # Darker hover

        ax_nan_save = self.fig.add_axes([0.55, 0.01, 0.19, 0.06])
        self.btn_nan_save = Button(ax_nan_save, 'Set all NaN & Close') # Shorter text
        self.btn_nan_save.on_clicked(self.all_nan_and_close)
        self.btn_nan_save.label.set_color('white')
        self.btn_nan_save.color = 'orange'
        self.btn_nan_save.hovercolor = 'darkorange'

        # --- Embed the 2D Image ---
        if self.img is not None and len(self.img) > 0 and isinstance(self.img, np.ndarray) and self.img.ndim == 2:
            self.im_handle = self.ax_img.imshow(self.img, aspect='auto', extent=[0, self.img.shape[1], self.img.shape[0], 0], cmap='gray', zorder=0)
            self.ax_img.set_ylim(self.img.shape[0], 0)
            self.ax_img.set_xlim(0, self.img.shape[1])
            # Hide image axes ticks/labels if desired
            # self.ax_img.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        else:
            # Set reasonable default limits if no image
            self.im_handle = None
            data_len = len(vector)
            min_val = np.nanmin(vector[np.isfinite(vector)]) if np.any(np.isfinite(vector)) else 0
            max_val = np.nanmax(vector[np.isfinite(vector)]) if np.any(np.isfinite(vector)) else 1
            padding = (max_val - min_val) * 0.1 + 1 # Add padding, ensure non-zero range
            self.ax_img.set_xlim(0, data_len)
            self.ax_img.set_ylim(max_val + padding, min_val - padding) # Inverted Y

        # Adjust layout to prevent overlap
        self.fig.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.9) # Make space for buttons/radio

        # --- Plot the Initial Signal ---
        self.update_signal_plot()
        self.last_update_time = time.time()
        
        # Add radio buttons for tool selection
        # Positioned on the left side
        ax_radio = self.fig.add_axes([0.01, 0.8, 0.1, 0.15], frameon=True) # Use fig.add_axes
        self.radio = RadioButtons(ax_radio, self.modes)
        self.radio.on_clicked(self.select_tool)

        # --- Modify Radio Button Circle Size (Robust Method) ---
        # Access circles via the axes patches instead of a direct attribute
        # Note: Default radius is often 0.05
        desired_radius = 0.06 # Adjust this value as needed
        for patch in self.radio.ax.patches:
            # Check if the patch is a Circle (radio buttons use Circles)
            if isinstance(patch, plt.Circle):
                patch.set_radius(desired_radius)

        # Maximize window (keep try-except for compatibility)
        try:
            figManager = plt.get_current_fig_manager()
            # Check for different backend managers
            if hasattr(figManager, 'window') and hasattr(figManager.window, 'state'):
                 figManager.window.state("zoomed")
            elif hasattr(figManager, 'frame') and hasattr(figManager.frame, 'Maximize'): # TkAgg
                 figManager.frame.Maximize(True)
            elif hasattr(figManager, 'window') and hasattr(figManager.window, 'showMaximized'): # Qt
                 figManager.window.showMaximized()
        except Exception as e:
             print(f"Note: Window maximization might not work on all backends. Error: {e}")

        plt.show()

    def update_signal_plot(self):
        # --- Update the data of the existing signal line ---
        x_indices = np.arange(len(self.modified_vector))
        self.signal_line.set_data(x_indices, self.modified_vector)

        # --- Safely remove eraser rectangle if it exists ---
        # Check if rect exists AND is still part of the axes patches
        if self.rect and self.rect in self.ax_signal.patches:
            self.rect.remove()
        # Always reset self.rect after attempting removal, as it should only exist during drag
        # self.rect = None # Resetting here might be too early if called outside on_release

        # Sync Y limits
        if self.im_handle:
            img_ylim = self.ax_img.get_ylim()
            # Add slight padding if needed, or handle inversions carefully
            self.ax_signal.set_ylim(img_ylim)
        else: # Auto-scale based on current vector data if no image
            valid_data = self.modified_vector[np.isfinite(self.modified_vector)]
            if len(valid_data) > 0:
                 min_val = np.nanmin(valid_data)
                 max_val = np.nanmax(valid_data)
                 padding = (max_val - min_val) * 0.1 + 1 # Add padding, ensure non-zero range
                 # Set inverted Y limits for consistency
                 self.ax_signal.set_ylim(max_val + padding, min_val - padding)
            # else: leave ylim as is if no valid data points

        # Ensure x-limits are also synchronized (important after zoom/pan)
        self.ax_signal.set_xlim(self.ax_img.get_xlim())

        # Ensure signal axes background is transparent
        self.ax_signal.patch.set_alpha(0.0)

        # Redraw the figure efficiently
        self.fig.canvas.draw_idle()


    def on_press(self, event):
        # Ignore clicks outside the signal axes or when a toolbar tool is active
        # Check event.inaxes against both axes if interaction with image is needed
        if event.inaxes != self.ax_signal or (hasattr(self.fig.canvas, 'toolbar') and self.fig.canvas.toolbar and self.fig.canvas.toolbar.mode != ''):
            return
        # Ignore clicks on buttons/widgets
        if event.inaxes == self.btn_save.ax or event.inaxes == self.btn_nan_save.ax or event.inaxes == self.radio.ax:
             return

        self.press = True
        self.start = event.xdata

        if self.currentModeIsPen():
             self.pen_xy = np.empty((0, 2), int)
             self.line.set_data([], []) # Clear visual pen line data


    def on_motion(self, event):
        # Ignore motion if not pressed, outside axes, or toolbar active
        if not self.press or event.inaxes != self.ax_signal or (hasattr(self.fig.canvas, 'toolbar') and self.fig.canvas.toolbar and self.fig.canvas.toolbar.mode != ''):
            return

        # Optional: Rate limiting (can be commented out if not needed)
        # current_time = time.time()
        # if current_time - self.last_update_time < 1/60: # Limit to ~60 FPS
        #     return
        # self.last_update_time = current_time

        x, y = event.xdata, event.ydata
        if x is None or y is None: return # Ignore events outside plot area limits

        if self.currentModeIsPen():
            x_int, y_int = int(x), int(y)
            if self.verbose:
                print(f"Pen on motion (x,y) = {x_int},\t{y_int}")

            self.pen_xy = np.append(self.pen_xy, [[x_int, y_int]], axis=0)
            self.line.set_data(self.pen_xy[:, 0], self.pen_xy[:, 1])
            self.fig.canvas.draw_idle()

        elif self.currentModeIsEraser():
            if self.start is None: return

            # Remove the previous rectangle if it exists and is still attached
            if self.rect and self.rect.axes:
                self.rect.remove()
            self.rect = None # Reset always before creating new

            self.end = x
            rect_x = min(self.start, self.end)
            rect_width = abs(self.end - self.start)
            # Use current signal plot Y limits for the rectangle height
            rect_y_bottom, rect_y_top = self.ax_signal.get_ylim() # Note: might be inverted (top < bottom)
            rect_height = abs(rect_y_top - rect_y_bottom)
            rect_y = min(rect_y_bottom, rect_y_top) # Start rectangle at the lower y value visually


            # Create and add the new rectangle patch
            self.rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                       color='yellow', alpha=0.3, zorder=6) # zorder slightly above signal
            self.ax_signal.add_patch(self.rect)
            self.fig.canvas.draw_idle()


    def on_release(self, event):
         # Check if press was initiated
        if not self.press:
             return
        # Reset press state regardless of where release happens
        self.press = False

        # Check if toolbar was active during the press-drag-release sequence
        if hasattr(self.fig.canvas, 'toolbar') and self.fig.canvas.toolbar and self.fig.canvas.toolbar.mode != '':
            # If toolbar was used (e.g., zoom box), clear any pen/eraser artifacts
            if self.currentModeIsPen():
                self.pen_xy = np.empty((0, 2), int)
                self.line.set_data([], [])
            if self.currentModeIsEraser() and self.rect and self.rect.axes:
                self.rect.remove()
                self.rect = None
            self.fig.canvas.draw_idle()
            return # Don't process as a pen/eraser action

        # --- Process Pen ---
        if self.currentModeIsPen():
            self.line.set_data([], []) # Clear visual pen line

            # Check if release occurred within axes and if there's data
            if event.inaxes == self.ax_signal and self.pen_xy.shape[0] > 0 and event.xdata is not None and event.ydata is not None:
                # Add the final point
                self.pen_xy = np.append(self.pen_xy, [[int(event.xdata), int(event.ydata)]], axis=0)

                if self.pen_xy.shape[0] < 2: # Need at least two points
                     self.pen_xy = np.empty((0, 2), int)
                     self.fig.canvas.draw_idle()
                     return

                # --- Processing the pen stroke (simplified duplicate handling) ---
                unique_x, indices = np.unique(self.pen_xy[:, 0], return_index=True)
                unique_pen_xy = self.pen_xy[indices] # Keep first occurrence

                max_index = len(self.modified_vector) - 1
                valid_indices = (unique_pen_xy[:, 0] >= 0) & (unique_pen_xy[:, 0] <= max_index)
                clipped_pen_xy = unique_pen_xy[valid_indices]

                if clipped_pen_xy.shape[0] < 2:
                    self.pen_xy = np.empty((0, 2), int)
                    self.fig.canvas.draw_idle()
                    return

                x_values = clipped_pen_xy[:, 0]
                y_values = clipped_pen_xy[:, 1]
                interpolated_x = np.arange(int(np.min(x_values)), int(np.max(x_values)) + 1)
                interpolated_y = np.interp(interpolated_x, x_values, y_values)

                original_values_segment = {}
                vector_changed = False # Flag to check if any change occurred

                for x_interp, y_interp in zip(interpolated_x, interpolated_y):
                    x_idx = int(x_interp)
                    final_y = np.nan # Default to NaN if snapping fails or no image

                    # Optional: Snap to brightest pixel
                    if self.img is not None and len(self.img) > 0 and isinstance(self.img, np.ndarray) and 0 <= x_idx < self.img.shape[1]:
                         y_center = int(round(y_interp)) # Round interpolated Y to nearest int for centering search
                         y_min_search = max(0, y_center - self.pen_thickness)
                         y_max_search = min(self.img.shape[0], y_center + self.pen_thickness + 1)

                         if y_min_search < y_max_search: # Ensure valid image slice
                              img_column_segment = self.img[y_min_search:y_max_search, x_idx]
                              if len(img_column_segment) > 0:
                                   try: # Handle case where segment might be all NaN
                                        brightest_pixel_offset = np.nanargmax(img_column_segment)
                                        final_y = y_min_search + brightest_pixel_offset
                                   except ValueError:
                                        final_y = int(round(y_interp)) # Fallback if argmax fails
                                   if self.verbose:
                                        print(f"Pen update @ x={x_idx}: Interp Y={y_interp:.1f}, Snapped Y={final_y} (Range {y_min_search}-{y_max_search-1})")
                              else:
                                   final_y = int(round(y_interp)) # Fallback if slice somehow empty
                         else:
                              final_y = int(round(y_interp)) # Fallback if search range invalid
                    else:
                        # No image or index out of bounds: use interpolated y directly
                        final_y = int(round(y_interp))
                        if self.verbose:
                             print(f"Pen update @ x={x_idx}: Interp Y={y_interp:.1f} (No image/OOB)")

                    # Store original value if not already stored and value changes
                    current_val = self.modified_vector[x_idx]
                    # Check for NaN equality correctly
                    if x_idx not in original_values_segment and (current_val != final_y and not (np.isnan(current_val) and np.isnan(final_y))):
                        original_values_segment[x_idx] = current_val
                        self.modified_vector[x_idx] = final_y
                        vector_changed = True
                    elif x_idx in original_values_segment: # If already stored, just update
                         self.modified_vector[x_idx] = final_y
                         vector_changed = True # Ensure flag is set

                if vector_changed and original_values_segment:
                     indices_affected = np.array(list(original_values_segment.keys()), dtype=int) # Ensure int indices
                     original_vals = np.array(list(original_values_segment.values()))
                     self.history.append((indices_affected, None, original_vals))
                     self.update_signal_plot() # Update plot only if changes were made

            # Reset pen path data regardless of whether changes were applied
            self.pen_xy = np.empty((0, 2), int)


        # --- Process Eraser ---
        elif self.currentModeIsEraser():
             # Remove visual rectangle safely
            if self.rect and self.rect.axes:
                self.rect.remove()
            self.rect = None

            # Use self.start and self.end captured during on_motion
            if self.start is not None and self.end is not None:
                # Determine start/end indices, clip to vector bounds
                start_idx = max(0, int(round(min(self.start, self.end))))
                # Add 1 to end_idx because slicing is exclusive of the end point
                end_idx = min(len(self.modified_vector), int(round(max(self.start, self.end))) + 1)

                if start_idx < end_idx: # Ensure valid range
                    indices_to_erase = np.arange(start_idx, end_idx)
                    # Check if any values in the range are *not* already NaN
                    if not np.all(np.isnan(self.modified_vector[start_idx:end_idx])):
                         original_values = self.modified_vector[start_idx:end_idx].copy()
                         self.history.append((indices_to_erase, None, original_values))

                         self.modified_vector[start_idx:end_idx] = np.nan
                         if self.verbose:
                              print(f"Eraser applied from index {start_idx} to {end_idx-1}")
                         self.update_signal_plot() # Update plot
                    elif self.verbose:
                         print(f"Eraser range {start_idx} to {end_idx-1} already NaN.")


            # Reset start/end points for next eraser action
            self.start = None
            self.end = None


    def on_key(self, event):
        if event.key == 'ctrl+z':
            if len(self.history) > 0:
                print("Undo triggered")
                indices, _, original_values = self.history.pop()
                # Ensure indices are integers for indexing
                indices = np.asarray(indices, dtype=int)
                # Restore, handling potential index errors if vector changed length (shouldn't happen here)
                try:
                    # Check bounds just in case
                    valid_mask = (indices >= 0) & (indices < len(self.modified_vector))
                    self.modified_vector[indices[valid_mask]] = original_values[valid_mask]
                    print(f"Restored {np.sum(valid_mask)} points.")
                    self.update_signal_plot()
                except IndexError as e:
                     print(f"Error during undo (IndexError): {e}. History might be corrupted.")

            else:
                print("Undo history empty")
        elif event.key == 'escape': # Optional: Escape key to cancel current action
             if self.press:
                  print("Action cancelled by Escape key")
                  self.press = False
                  if self.currentModeIsPen():
                       self.pen_xy = np.empty((0, 2), int)
                       self.line.set_data([], [])
                  if self.currentModeIsEraser() and self.rect and self.rect.axes:
                       self.rect.remove()
                       self.rect = None
                  self.start = None
                  self.end = None
                  self.fig.canvas.draw_idle()


    def on_zoom(self, event):
        # Default Matplotlib zoom/pan handles the scaling.
        # We just need to ensure our signal plot limits stay synced.
        # Use draw_idle to request a redraw after zoom/pan action finishes
        self.fig.canvas.draw_idle()
        # Call update_signal_plot to resync limits and redraw signal correctly
        # Need a slight delay or check if zoom is finished? Often draw_idle is enough.
        # If limits drift, might need to explicitly call update_signal_plot here too.
        # self.update_signal_plot() # uncomment if limits seem unsynced after zoom/pan


    def save_and_close(self, event):
        print("Saving modifications and closing.")
        global modified_vector
        modified_vector = self.modified_vector.copy()
        plt.close(self.fig)

    def all_nan_and_close(self, event):
        print("Setting all to NaN, saving, and closing.")
        # Create undo state for the entire vector
        indices = np.arange(len(self.modified_vector))
        original = self.modified_vector.copy()
        self.history.append((indices, None, original)) # Allow undoing 'all NaN'

        self.modified_vector[:] = np.nan
        global modified_vector
        modified_vector = self.modified_vector.copy()
        plt.close(self.fig)

    def select_tool(self, label):
        if self.currentMode == label: return # No change
        self.currentMode = label
        print(f"Tool changed to: {self.currentMode}")
        # Cancel any ongoing action when switching tools
        self.press = False
        if self.rect and self.rect.axes:
            self.rect.remove()
            self.rect = None
        self.pen_xy = np.empty((0, 2), int)
        self.line.set_data([], [])
        self.start = None
        self.end = None
        self.fig.canvas.draw_idle()

    def currentModeIsPen(self):
        return self.currentMode.casefold() == 'pen'.casefold()

    def currentModeIsEraser(self):
        # Corrected typo
        return self.currentMode.casefold() == 'eraser'.casefold()


