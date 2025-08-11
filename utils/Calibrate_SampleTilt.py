import os
import numpy as np
from skimage.io import imread
import re
import pandas as pd
#os.environ['SKIMAGE_NO_POOCH'] = '1'
#os.environ['POOCH_BASE_DIR'] = r'C:/Raju/Codes/MCT_Analysis/mct_segmentation_package/scikit_cache'
#Force skimage cache to initialize now (avoids parallel crash)
from skimage import data
_ = data.camera()
from skimage import measure
from skimage.transform import resize

import cv2
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

#from mct_segmentation_package.examples.Segmentation_Pipeline import Measure

def calibrate(label_volume):
    labeled_volume = measure.label(label_volume, connectivity=3)
    props = measure.regionprops(labeled_volume)
    print(f'Number of slices in labeled volume: {labeled_volume.shape[0]}')


    (voxel_size_x, voxel_size_y, voxel_size_z) = (1, 1, 1)
    results = []

    def angle_with_axis(vec1, axis):
        cos_theta = np.dot(vec1, axis) / (np.linalg.norm(vec1) * np.linalg.norm(axis))
        angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return min(angle_deg, 180 - angle_deg)  # ensures angle is in [0, 90]°

    z_axis = np.array([0, 0, 1])

    for prop in props:
        minz, miny, minx, maxz, maxy, maxx = prop.bbox

        if maxz - minz <= 1:
            continue

        feret_x = maxx - minx
        feret_y = maxy - miny
        feret_z = maxz - minz

        pore_volume_voxel_count = prop.area
        voxel_volume = voxel_size_x * voxel_size_y * voxel_size_z
        pore_volume = pore_volume_voxel_count * voxel_volume

        region_mask = labeled_volume == prop.label
        z, y, x = np.where(region_mask)
        #points = np.stack([x, y, z], axis=1)
        points = np.stack([x, y, z], axis=1).astype(np.float32)
        MAX_POINTS = 500_000

        if points.shape[0] > MAX_POINTS:
            print(f"Subsampling label {prop.label} from {points.shape[0]} to {MAX_POINTS} points")
            idx = np.random.choice(points.shape[0], MAX_POINTS, replace=False)
            points = points[idx]
            print("sampling done")

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

        # Corrected angle computation (against Z-axis)
        major_axis_angle_z = angle_with_axis(major_axis_vector, z_axis)
        intermediate_axis_angle_z = angle_with_axis(intermediate_axis_vector, z_axis)
        minor_axis_angle_z = angle_with_axis(minor_axis_vector, z_axis)


        volume = (maxy - miny) * (maxx - minx) * (maxz - minz)
        verts, faces, _, _ = measure.marching_cubes(region_mask, level=0)
        surface_area = measure.mesh_surface_area(verts, faces)

        anisotropy = major_axis_length / minor_axis_length if minor_axis_length != 0 else 0
        compactness = (volume ** (2 / 3)) / surface_area if surface_area > 0 else 0
        eccentricity = np.sqrt(1 - (minor_axis_length ** 2 / major_axis_length ** 2)) if major_axis_length != 0 else 0
        flatness = intermediate_axis_length / minor_axis_length if minor_axis_length != 0 else 0
        elongation = major_axis_length / minor_axis_length if minor_axis_length != 0 else 0
        sphericity = (6 * volume ** (2 / 3)) / surface_area if surface_area != 0 else 0


        results.append({
            'Slice': minz,
            'Label': prop.label,
            'Centroid X': center[0],
            'Centroid Y': center[1],
            'Centroid Z': center[2],
            'x1': minx,
            'x2': maxx,
            'y1': miny,
            'y2': maxy,
            'z1': minz,
            'z2': maxz,
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
            'surface_area': surface_area,
            'Anisotropy': anisotropy,
            'Compactness': compactness,
            'Eccentricity': eccentricity,
            'Flatness': flatness,
            'Elongation': elongation,
            'Sphericity': sphericity,
            'Moments of Inertia': eigenvalues,

        })

    return results


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def read_segmented_volume(path):
    """
    Reads a single image file or a stack from a folder of images.
    Returns the loaded volume and prints its shape and type.
    """
    if os.path.isdir(path):
        # Folder: check all image files
        filenames = sorted(
            [f for f in os.listdir(path) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))],
            key=natural_sort_key
        )
        if not filenames:
            raise ValueError(f"No image files found in folder: {path}")
        print(filenames)

        volumes = []
        for fname in filenames:
            full_path = os.path.join(path, fname)
            img = imread(full_path)
            img = np.squeeze(img)
            if img.ndim == 2:
                img = img[np.newaxis, ...]  # Convert 2D to 3D (1 slice)
            elif img.ndim != 3:
                raise ValueError(f"Unsupported image shape: {img.shape} in {fname}")
            volumes.append(img)

        volume = np.concatenate(volumes, axis=0)
        print(f"Loaded and combined {len(filenames)} image(s) into one volume from folder: {path}")
        print(f"Shape: {volume.shape} (Z, Y, X)")
    elif os.path.isfile(path):
        # Single file: read as 2D or 3D image
        volume = imread(path)
        volume=np.squeeze(volume)
        print(f"Loaded image file: {path}")
        print(f"Image(s) are of {volume.dtype}, Shape: {volume.shape} ({'3D stack' if volume.ndim == 3 else '2D image'})")
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    return volume




if __name__ == "__main__":
    segmented_image_path="F:/Synchrotron_MCT/21344/Dwell_ON_OFF/Dwell31/output/labeled_slices/isolated_volume/"
    segmented_volume=read_segmented_volume(segmented_image_path)
    print(f"Original size: {segmented_volume.nbytes / (1024**2):.2f} MB")
    #segmented_volume=segmented_volume[2000:6000]
    results=calibrate(segmented_volume)
    
    dirname=os.path.dirname(segmented_image_path)

    df = pd.DataFrame(results)
    savepath=os.path.join(dirname,"Calibrate","tilt_calibration_results.xlsx")
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    df.to_excel(savepath, index=False)