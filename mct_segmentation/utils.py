import os
import numpy as np
import time
import tifffile

def read_and_sort_tiff_stack(folder_path):
    tiff_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')],
                        key=lambda x: int(''.join(filter(str.isdigit, x))))
    images = [imageio.imread(os.path.join(folder_path, f)) for f in tiff_files]
    return np.stack(images), tiff_files

def save_tiff_stack(rois, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for i, roi in enumerate(rois):
        tifffile.imwrite(os.path.join(output_folder, f'roi_{i:03d}.tif'), roi.astype(np.uint8))

'''
def sorted_tiff_files(folder_path):
    """Return a naturally sorted list of .tif files in a folder."""
    return sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
'''

def normalize_image(image):
    """Normalize image to range [0, 1]."""
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)

def time_it(func):
    """Decorator to time a function's execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

def make_output_folder(path):
    """Create an output folder if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path
