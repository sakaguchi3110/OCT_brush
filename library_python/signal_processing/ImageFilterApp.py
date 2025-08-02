import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageFilter, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageFilterApp:
    def __init__(self, root, image_array):
        self.root = root
        self.root.title("Image Filter Application")

        # Convert the 2D array to a PIL image
        self.original_image = Image.fromarray(image_array).convert("L")
        self.filtered_image = self.original_image.copy()

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Create a frame for the image
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Display the image in a subplot
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.original_image, cmap='gray', aspect='auto')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.get_tk_widget().pack()

        # Create a frame for the controls
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # X range controls
        tk.Label(self.control_frame, text="X Range:").grid(row=0, column=0, sticky=tk.W)
        self.x_min = tk.Entry(self.control_frame)
        self.x_min.grid(row=0, column=1)
        self.x_max = tk.Entry(self.control_frame)
        self.x_max.grid(row=0, column=2)

        # Y range controls
        tk.Label(self.control_frame, text="Y Range:").grid(row=1, column=0, sticky=tk.W)
        self.y_min = tk.Entry(self.control_frame)
        self.y_min.grid(row=1, column=1)
        self.y_max = tk.Entry(self.control_frame)
        self.y_max.grid(row=1, column=2)

        # Filter options
        tk.Label(self.control_frame, text="Filter:").grid(row=2, column=0, sticky=tk.W)
        self.filter_var = tk.StringVar(value="None")
        tk.Radiobutton(self.control_frame, text="None", variable=self.filter_var, value="None").grid(row=2, column=1)
        tk.Radiobutton(self.control_frame, text="Median Removal (Column)", variable=self.filter_var, value="MedianColumn").grid(row=3, column=1)
        tk.Radiobutton(self.control_frame, text="Median Removal (Line)", variable=self.filter_var, value="MedianLine").grid(row=4, column=1)
        tk.Radiobutton(self.control_frame, text="Blur", variable=self.filter_var, value="Blur").grid(row=5, column=1)

        # Blur value
        tk.Label(self.control_frame, text="Blur Value:").grid(row=6, column=0, sticky=tk.W)
        self.blur_value = tk.Entry(self.control_frame)
        self.blur_value.grid(row=6, column=1)

        # Update button
        self.update_button = tk.Button(self.control_frame, text="Update", command=self.apply_filters)
        self.update_button.grid(row=7, columnspan=3)

    def display_image(self):
        # Clear the previous image
        self.ax.clear()
        
        # Display the updated image in the subplot with aspect ratio auto
        self.ax.imshow(self.filtered_image, cmap='gray', aspect='auto')
        
        # Draw the canvas
        self.canvas.draw()

    def apply_filters(self):
        x_min = int(self.x_min.get()) if self.x_min.get() else 0
        x_max = int(self.x_max.get()) if self.x_max.get() else self.original_image.width
        y_min = int(self.y_min.get()) if self.y_min.get() else 0
        y_max = int(self.y_max.get()) if self.y_max.get() else self.original_image.height

        region = (x_min, y_min, x_max, y_max)
        
        filtered_image = self.original_image.crop(region)

        filter_type = self.filter_var.get()
        
        if filter_type == "MedianColumn":
            filtered_image = filtered_image.filter(ImageFilter.MedianFilter(size=3))
        
        elif filter_type == "MedianLine":
            filtered_image_np = np.array(filtered_image)
            for i in range(filtered_image_np.shape[0]):
                filtered_image_np[i] = np.median(filtered_image_np[i])
            filtered_image = Image.fromarray(filtered_image_np)

        elif filter_type == "Blur":
            blur_value = int(self.blur_value.get()) if self.blur_value.get() else 2
            filtered_image = filtered_image.filter(ImageFilter.GaussianBlur(blur_value))

        # Paste the filtered region back to the original image
        new_image = self.original_image.copy()
        new_image.paste(filtered_image, region)

        # Update the filtered image and display it
        self.filtered_image = new_image
        self.display_image()

