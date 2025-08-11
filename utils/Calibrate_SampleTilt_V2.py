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

def calibrate(label_volume):
    print("Labelling the volume")
    t1=time.time()
    
    labeled_volume = cc3d.connected_components(label_volume, connectivity=26)

    #labeled_volume = measure.label(label_volume, connectivity=3)
    t2=time.time()
    print(f"Time for labelling: {(t2-t1)/60}")
    print("Measuring the properties")
    props = measure.regionprops(labeled_volume)
    print(f'Number of slices in labeled volume: {labeled_volume.shape[0]}')
    print(f"Time for measuring properties: {(time.time()-t2)/60}")

    (voxel_size_x, voxel_size_y, voxel_size_z) = (1, 1, 1)
    results = []

    def angle_with_axis(vec1, axis):
        cos_theta = np.dot(vec1, axis) / (np.linalg.norm(vec1) * np.linalg.norm(axis))
        angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return min(angle_deg, 180 - angle_deg)

    z_axis = np.array([0, 0, 1])

    for prop in props:
        print(f"Processing label: {prop.label}")
        minz, miny, minx, maxz, maxy, maxx = prop.bbox
        if maxz - minz <= 1:
            continue

        feret_x, feret_y, feret_z = maxx - minx, maxy - miny, maxz - minz
        pore_volume_voxel_count = prop.area
        pore_volume = pore_volume_voxel_count * voxel_size_x * voxel_size_y * voxel_size_z
        volume = (maxy - miny) * (maxx - minx) * (maxz - minz)

        region_mask = (labeled_volume == prop.label)
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

        major_axis_length = semi_axes_lengths[0]
        intermediate_axis_length = semi_axes_lengths[1]
        minor_axis_length = semi_axes_lengths[2]

        major_axis_angle_z = angle_with_axis(major_axis_vector, z_axis)
        intermediate_axis_angle_z = angle_with_axis(intermediate_axis_vector, z_axis)
        minor_axis_angle_z = angle_with_axis(minor_axis_vector, z_axis)



        results.append({
            'Slice': minz,
            'Label': prop.label,
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
            'Major Axis Length': major_axis_length,
            'Minor Axis Length': minor_axis_length,
            'Intermediate Axis Length': intermediate_axis_length,
            'Major Axis Angle Z (deg)': major_axis_angle_z,
            'Intermediate Axis Angle Z (deg)': intermediate_axis_angle_z,
            'Minor Axis Angle Z (deg)': minor_axis_angle_z,
            'Moments of Inertia': eigenvalues,
        })

        # Cleanup
        del region_mask, z, y, x, points, centered_points, cov
        gc.collect()

    return results


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
    segmented_image_path = "F:/Synchrotron_MCT/21344/Dwell_ON_OFF/Dwell26/output/labeled_slices/isolated_volume"
    segmented_volume = read_segmented_volume(segmented_image_path)

    print(f"Original size: {segmented_volume.nbytes / (1024 ** 2):.2f} MB")

    results = calibrate(segmented_volume)

    dirname = os.path.dirname(segmented_image_path)
    df = pd.DataFrame(results)
    savepath = os.path.join(dirname, "Calibrate", "tilt_calibration_results.xlsx")
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    df.to_excel(savepath, index=False)
