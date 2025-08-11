import numpy as np
import os
from scipy import ndimage as ndi
from joblib import Parallel, delayed
from skimage.io import imread
from skimage.morphology import disk, square, rectangle, diamond
from tifffile import imwrite
import matplotlib
matplotlib.use('TkAgg')  # Force use of Tkinter backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def read_segmented_volume(path):
    if path.endswith(".npy"):
        print(f"Reading memory-mapped .npy volume from {path}")
        volume = np.load(path, mmap_mode="r")
        print(f"Shape: {volume.shape}, dtype: {volume.dtype}")
        return volume

    if os.path.isdir(path):
        filenames = sorted(
            [f for f in os.listdir(path) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))],
            key=natural_sort_key
        )
        if not filenames:
            raise ValueError(f"No image files found in folder: {path}")

        volumes = []
        for fname in filenames:
            full_path = os.path.join(path, fname)
            img = imread(full_path).astype(np.uint16)
            img = np.squeeze(img)
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            elif img.ndim != 3:
                raise ValueError(f"Unsupported image shape: {img.shape} in {fname}")
            volumes.append(img)

        volume = np.concatenate(volumes, axis=0)
        print(f"Loaded and combined {len(filenames)} image(s) into volume")
    else:
        volume = imread(path).astype(np.uint16)
        volume = np.squeeze(volume)

    print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
    return volume

class Filters:
    def __init__(self, volume, filtered_volume,kernel_shape='disk', kernel_size=3, iterations=1, n_jobs=-1):
        self.volume = volume
        self.filtered_volume=filtered_volume
        self.iterations = iterations
        self.n_jobs = n_jobs
        self.kernel = self._create_kernel(kernel_shape, kernel_size)

    def _create_kernel(self, shape, size):
        if shape == 'disk':
            return disk(size)
        elif shape == 'square':
            return square(size)
        elif shape == 'diamond':
            return diamond(size)
        elif shape == 'rectangle':
            assert isinstance(size, (tuple, list)) and len(size) == 2, "Rectangle size must be tuple of 2 integers"
            return rectangle(*size)
        else:
            raise ValueError(f"Unsupported kernel shape: {shape}")

    def _apply_operation(self, slice_2d, operation):
        self.filtered_volume = slice_2d.copy()
        for _ in range(self.iterations):
            if operation == 'dilation':
                self.filtered_volume = ndi.grey_dilation(self.filtered_volume, footprint=self.kernel)
            elif operation == 'erosion':
                self.filtered_volume = ndi.grey_erosion(self.filtered_volume, footprint=self.kernel)
            elif operation == 'opening':
                self.filtered_volume = ndi.grey_opening(self.filtered_volume, footprint=self.kernel)
            elif operation == 'closing':
                self.filtered_volume = ndi.grey_closing(self.filtered_volume, footprint=self.kernel)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        return self.filtered_volume


    def apply(self, operation):
        print(f"Applying '{operation}' with {self.iterations} iterations using {self.n_jobs} parallel workers...")
        processed_slices = Parallel(n_jobs=self.n_jobs)(
            delayed(self._apply_operation)(self.volume[i], operation)
            for i in range(self.volume.shape[0])
        )
        result_volume = np.stack(processed_slices, axis=0)
        print(f"Operation '{operation}' complete.")
        return result_volume

    def fill_holes(self, threshold=0):
        """
        Fill holes in binary slices (2D) slice-by-slice in parallel.

        Args:
            threshold (int): Optional threshold to binarize if input is not binary
        Returns:
            np.ndarray: Volume with holes filled
        """
        print(f"Filling holes slice-by-slice in parallel...")

        def process_slice(slice_2d):
            binary = slice_2d > threshold
            filled = ndi.binary_fill_holes(binary)
            return filled.astype(slice_2d.dtype)

        filled_slices = Parallel(n_jobs=self.n_jobs)(
            delayed(process_slice)(self.volume[i])
            for i in range(self.volume.shape[0])
        )
        self.filtered_volume = np.stack(filled_slices, axis=0)
        print("Hole filling complete.")
        return self.filtered_volume

    def save_volume(self, volume, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_path.endswith('.npy'):
            np.save(output_path, volume)
            print(f"Saved volume as .npy to {output_path}")
        elif output_path.endswith('.tif') or output_path.endswith('.tiff'):
            imwrite(output_path, volume.astype(np.uint16))
            print(f"Saved volume as TIFF stack to {output_path}")
        else:
            raise ValueError("Output path must end with .npy or .tif/.tiff")


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

def get_image_dir(image_path):
    if os.path.isdir(image_path):
        return os.path.abspath(image_path)
    else:
        return os.path.abspath(os.path.dirname(image_path))


if __name__ == "__main__":
    image_path = "F:/Synchrotron_MCT/21344/Dwell_ON_OFF/Dwell26/output/labeled_slices/isolated_volume/Dwell26_isolated_volume_(1000, 1000, 10).tif"
    image_volume = read_segmented_volume(image_path)
    image_volume=image_volume[1000:1100]
    print(f"Original size: {image_volume.nbytes / (1024 ** 2):.2f} MB")


    # Initialize filter with square kernel of size 5, 2 iterations
    filt = Filters(image_volume, kernel_shape='square', kernel_size=30, iterations=4,n_jobs=30)

    # Apply erosion
    result = filt.apply('erosion')
    result = filt.fill_holes(threshold=0) 

    output_path=os.path.join(get_image_dir(image_path),"output","filtered_volume.tif")

    # Save result
    filt.save_volume(result, output_path)
    filt.visualize_with_slider(result)
