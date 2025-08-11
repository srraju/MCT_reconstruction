"""
Updated Calibrate_SampleTilt.py with Memory Optimizations
Key changes:

Explicitly delete large intermediate variables.

Use uint16 type to reduce memory.

Call gc.collect() to free memory.

Optionally support loading .npy files via memory-mapping.
"""
import os
import numpy as np
import pandas as pd
from skimage import measure, data
from skimage.io import imread
import re
import gc
import time
import cc3d
_ = data.camera()  # Trigger skimage cache


from joblib import Parallel, delayed, cpu_count

def count_pixels(label_id, labeled_volume):
    count = np.sum(labeled_volume == label_id)
    return {'Label': label_id, 'Voxel Count': count}

def count_label_voxels_parallel(label_volume, connectivity=26, num_jobs=None, excel_path="label_counts.xlsx"):
    print("Labeling volume...")
    t1=time.time()
    labeled_volume = cc3d.connected_components(label_volume, connectivity=connectivity)
    t2=time.time()
    print(f"labeled in {(t2-t2)/60}")

    unique_labels = np.unique(labeled_volume)
    print(f"labeled in {(time.time()-t2)}")
    unique_labels = unique_labels[unique_labels != 0]  # Skip background (label 0)
    print(f"Number of labels: {len(unique_labels)}")



    results = Parallel(n_jobs=30, backend='threading')(
        delayed(count_pixels)(label_id, labeled_volume) for label_id in unique_labels
    )

    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print(f"Saved voxel counts to: {excel_path}")

    return df


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

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
        for fname in filenames:
            print(fname)
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

if __name__ == "__main__":
    segmented_image_path = "F:/Synchrotron_MCT/21344/Dwell_ON_OFF/Dwell34/output/labeled_slices/isolated_volume"
    segmented_volume = read_segmented_volume(segmented_image_path)

    print(f"Original size: {segmented_volume.nbytes / (1024 ** 2):.2f} MB")

    count_label_voxels_parallel(segmented_volume, excel_path="label_counts.xlsx")
    '''

    results = calibrate(segmented_volume)

    dirname = os.path.dirname(segmented_image_path)
    df = pd.DataFrame(results)
    savepath = os.path.join(dirname, "Calibrate", "tilt_calibration_results.xlsx")
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    df.to_excel(savepath, index=False)
    '''
