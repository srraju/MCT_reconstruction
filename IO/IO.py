import sys
import os
import glob 
import time
import numpy as np
import imageio
import tifffile
from tifffile import imwrite
import matplotlib
matplotlib.use('TkAgg')  # Force use of Tkinter backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import warnings
from skimage.io import imread, imsave
from contextlib import contextmanager

from concurrent.futures import ThreadPoolExecutor
import re

@contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as fnull:
        old_stderr = sys.stderr
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def safe_imread(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with suppress_stderr():
            return imread(path)

class ImageHandler:
    def __init__(self, batch_process, folder_path):
        self.batch_process=batch_process
        self.folder_path = folder_path
        self.file_names = []
        self.output_folder = os.path.join(folder_path, "output")
        os.makedirs(self.output_folder, exist_ok=True)
        self.images = []
        self.image_volume=None
        self.label_volume=None
        self.label_directory=None
        


    def natural_sort_key(self,s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]

    def read_volume(self,folder):
        self.folder_path=folder
        if self.folder_path.endswith(".npy"):
            print(f"Reading memory-mapped .npy volume from {self.folder_path}")
            volume = np.load(self.folder_path, mmap_mode="r")
            print(f"Shape: {volume.shape}, dtype: {volume.dtype}")

        elif os.path.isdir(self.folder_path):
            filenames = sorted(
                [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))],
                key=self.natural_sort_key
            )
            if not filenames:
                raise ValueError(f"No image files found in folder: {self.folder_path}")

            volume = []
            for fname in filenames:
                full_path = os.path.join(self.folder_path, fname)
                img = safe_imread(full_path).astype(np.uint16)
                img = np.squeeze(img)
                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                elif img.ndim != 3:
                    raise ValueError(f"Unsupported image shape: {img.shape} in {fname}")
                volume.append(img)

            volume = np.concatenate(volume, axis=0)
            print(f"Loaded and combined {len(filenames)} image(s) into volume")

        else:
            volume = safe_imread(self.folder_path).astype(np.uint16)
            volume = np.squeeze(volume)

        print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
        self.file_names = filenames if os.path.isdir(self.folder_path) else [os.path.basename(self.folder_path)]
        print(self.file_names)
        return volume

    def get_base_filename(self,filename):
        # Remove extension
        base = os.path.splitext(filename)[0]
        # Remove trailing number with optional underscore/dash before it
        # e.g. image_001 -> image, scan-23 -> scan
        base = re.sub(r'[_-]?\d+$', '', base)
        return base

    def save_volume(self, volume, subfolder=None, format=None):
        if not self.file_names:
            raise ValueError("self.file_names is empty, cannot generate output filename")

        # Get first filename from self.file_names as base
        original_fname = self.file_names[0]
        base_name = self.get_base_filename(original_fname)

        # Choose extension based on format argument or default to .npy
        ext = '.' + (format if format else 'npy').lower()

        # Construct full output path
        filename = os.path.join(self.output_folder, subfolder, base_name + ext)
        self.label_directory=os.path.join(self.output_folder, subfolder)

        # Make sure output folder exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save volume accordingly
        if ext == '.npy':
            np.save(filename, volume)
            print(f"Saved volume as .npy to {filename}")
        elif ext in ['.tif', '.tiff']:
            imwrite(filename, volume.astype(np.uint16))
            print(f"Saved volume as TIFF stack to {filename}")
        else:
            raise ValueError("Output file extension must be .npy or .tif/.tiff")


    def save_images(self, volume,subfolder=None):
        print("Saving image sequence in parallel...")
        start_time = time.time()

        # Use self.file_names to generate output filenames
        filenames = [os.path.basename(f) for f in self.file_names]

        # Handle case when number of slices != number of filenames:
        if len(filenames) != volume.shape[0]:
            # Generate generic names: slice_000.tif, slice_001.tif ...
            filenames = [f"slice_{i:03d}.tif" for i in range(volume.shape[0])]

        # Prepare the images as slices along axis 0
        slices = [volume[i, ...] for i in range(volume.shape[0])]

        with ThreadPoolExecutor() as executor:
            list(executor.map(self._save_single_image, slices, [subfolder]*len(slices), filenames))

        print(f"Time taken to save {len(slices)} images: {time.time() - start_time:.2f} seconds")


    def _save_single_image(self, image, subfolder, file_name):
        name, ext = os.path.splitext(file_name)
        if not ext:
            ext = '.tif'  # default extension if none
        if '_' in name:
            prefix, suffix = name.rsplit('_', 1)
            out_name = f"{prefix}-out_{suffix}{ext}"
        else:
            out_name = f"{name}-out{ext}"

        output_path = os.path.join(self.output_folder, subfolder, out_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #print(f"Saving: {output_path}")

        # Save the image slice
        imsave(output_path, image.astype(np.uint16),check_contrast=False)


    def crop_image(self, image, x1, y1, sizex, sizey):
        return image[y1:y1 + sizey, x1:x1 + sizex]


    def visualize_with_slider(self, volume=None, cmap='gray'):
        if volume is None:
            volume = self.volume
        assert volume.ndim == 3, "Volume must be 3D"

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        slice_idx = 0
        img = ax.imshow(volume[slice_idx], cmap=cmap)
        ax.set_title(f"Slice {slice_idx + 1}/{volume.shape[0]}")
        
        # Slider axis and slider object
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, 'Slice', 0, volume.shape[0] - 1, valinit=slice_idx, valstep=1)

        # Update function
        def update(val):
            idx = int(slider.val)
            img.set_data(volume[idx])
            ax.set_title(f"Slice {idx + 1}/{volume.shape[0]}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()


    def save_volume_stack(self, volume, axis):
        if axis == 'x':
            volume_stack = np.swapaxes(volume, 0, 1)
            output_filename = 'volume_stack_along_x.tif'
        elif axis == 'y':
            volume_stack = np.swapaxes(volume, 0, 2)
            output_filename = 'volume_stack_along_y.tif'
        elif axis == 'z':
            volume_stack = volume
            output_filename = 'volume_stack_along_z.tif'
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        volume_stack_scaled = (65535 * (volume_stack - volume_stack.min()) / (volume_stack.ptp())).astype(np.uint16)
        output_path = os.path.join(self.output_folder, output_filename)
        imsave(output_path, volume_stack_scaled)

    @staticmethod
    def save_tiff_stack(rois, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        with ThreadPoolExecutor() as executor:
            tasks = [executor.submit(tifffile.imwrite, os.path.join(output_folder, f'roi_{i:03d}.tif'), roi.astype(np.uint8)) for i, roi in enumerate(rois)]
            for task in tasks:
                task.result()
