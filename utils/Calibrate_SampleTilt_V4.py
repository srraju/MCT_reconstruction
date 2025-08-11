import os
import re
import gc
import time
import numpy as np
import pandas as pd
from skimage import data
from skimage.io import imread
from skimage.measure import regionprops, label
import cc3d
from joblib import Parallel, delayed
_ = data.camera()  # Cache init

# Calibration helpers
def angle_with_axis(vec1, axis):
    cos_theta = np.dot(vec1, axis) / (np.linalg.norm(vec1) * np.linalg.norm(axis))
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return min(angle_deg, 180 - angle_deg)

z_axis = np.array([0, 0, 1])
voxel_size_x, voxel_size_y, voxel_size_z = 1, 1, 1

# Parallel region processor
def process_region(label_id, labeled_volume):
    region_mask = (labeled_volume == label_id)
    prop = regionprops(region_mask.astype(np.uint8))[0]

    minz, miny, minx, maxz, maxy, maxx = prop.bbox
    if maxz - minz <= 1:
        return None

    feret_x, feret_y, feret_z = maxx - minx, maxy - miny, maxz - minz
    voxel_count = prop.area
    pore_volume = voxel_count * voxel_size_x * voxel_size_y * voxel_size_z
    volume = feret_x * feret_y * feret_z

    z, y, x = np.where(region_mask)
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    MAX_POINTS = 500_000
    if points.shape[0] > MAX_POINTS:
        idx = np.random.choice(points.shape[0], MAX_POINTS, replace=False)
        points = points[idx]

    center = points.mean(axis=0)
    centered_points = points - center
    cov = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    semi_axes_lengths = 2 * np.sqrt(eigenvalues)
    major_axis_vector = eigenvectors[:, 0]
    intermediate_axis_vector = eigenvectors[:, 1]
    minor_axis_vector = eigenvectors[:, 2]

    result = {
        'Slice': minz,
        'Label': label_id,
        'Centroid X': center[0],
        'Centroid Y': center[1],
        'Centroid Z': center[2],
        'x1': minx, 'x2': maxx,
        'y1': miny, 'y2': maxy,
        'z1': minz, 'z2': maxz,
        'Feret Size X': feret_x,
        'Feret Size Y': feret_y,
        'Feret Size Z': feret_z,
        'Pore volume': pore_volume,
        'Bounding box volume': volume,
        'Major Axis Vector': major_axis_vector,
        'Minor Axis Vector': minor_axis_vector,
        'Intermediate Axis Vector': intermediate_axis_vector,
        'Major Axis Length': semi_axes_lengths[0],
        'Minor Axis Length': semi_axes_lengths[2],
        'Intermediate Axis Length': semi_axes_lengths[1],
        'Major Axis Angle Z (deg)': angle_with_axis(major_axis_vector, z_axis),
        'Intermediate Axis Angle Z (deg)': angle_with_axis(intermediate_axis_vector, z_axis),
        'Minor Axis Angle Z (deg)': angle_with_axis(minor_axis_vector, z_axis),
        'Moments of Inertia': eigenvalues
    }

    # Clean up memory
    del region_mask, points, centered_points, cov
    gc.collect()
    return result

# Parallel voxel count + calibration
def count_label_voxels_parallel(label_volume, connectivity=26, excel_path="label_counts.xlsx", n_jobs=30):
    print("Labeling volume...")
    t1 = time.time()
    labeled_volume = cc3d.connected_components(label_volume, connectivity=connectivity)
    print(f"Labeled in {(time.time() - t1):.2f} seconds")

    unique_labels = np.unique(labeled_volume)
    unique_labels = unique_labels[unique_labels != 0]
    print(f"Number of labels: {len(unique_labels)}")
    t3=time.time()

    print("Starting parallel region processing...")
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_region)(label_id, labeled_volume) for label_id in unique_labels
    )

    results = [r for r in results if r is not None]

    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print(f"Saved voxel/ellipsoid data to: {excel_path}")
    print(f"Total time for processing: {(time.time()-t1)/60} mins")
    return df

# Helpers
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def read_segmented_volume(path):
    if path.endswith(".npy"):
        print(f"Reading memory-mapped .npy volume from {path}")
        volume = np.load(path, mmap_mode="r")
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
            img = imread(os.path.join(path, fname)).astype(np.uint16)
            img = np.squeeze(img)
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            volumes.append(img)

        volume = np.concatenate(volumes, axis=0)
    else:
        volume = imread(path).astype(np.uint16)
        volume = np.squeeze(volume)

    print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
    return volume

# Run block
if __name__ == "__main__":
    segmented_image_path = "F:/Synchrotron_MCT/21344/Dwell_ON_OFF/Dwell34/output/labeled_slices/isolated_volume"
    segmented_volume = read_segmented_volume(segmented_image_path)

    print(f"Original size: {segmented_volume.nbytes / (1024 ** 2):.2f} MB")
    count_label_voxels_parallel(segmented_volume, excel_path="label_counts.xlsx", n_jobs=30)
