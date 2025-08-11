import numpy as np
import tifffile
import os
import re
from skimage.io import imread  # Ensure skimage is installed: pip install scikit-image

def extract_circular_regions(volume, positions, radius, out_path):
    radius_sq = radius ** 2
    output_volume = np.zeros_like(volume, dtype=volume.dtype)
    

    for x, y, z in positions:
        x, y, z = int(round(x)), int(round(y)), int(round(z))

        if z < 0 or z >= volume.shape[0]:
            print(f"Skipping Z={z}, out of bounds.")
            continue

        # Define bounding box
        y_min = max(0, y - radius)
        y_max = min(volume.shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(volume.shape[2], x + radius + 1)

        # Extract region
        region = volume[z, y_min:y_max, x_min:x_max]

        # Create a circular mask in local coordinates
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (yy - y)**2 + (xx - x)**2 <= radius_sq

        # Apply mask to the region
        output_slice = output_volume[z]
        output_slice[y_min:y_max, x_min:x_max][mask] = region[mask]

    # Save as TIFF
    out_file = os.path.join(out_path, "circular_regions_volume.tif")
    tifffile.imwrite(out_file, output_volume)
    print(f"Circular region volume saved to {out_file}")

    return output_volume


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
    image_path = "F:/Synchrotron_MCT/21344/Dwell_ON_OFF/Dwell26/output/labeled_slices/isolated_volume"
    image_volume = read_segmented_volume(image_path)
    print(f"Original size: {image_volume.nbytes / (1024 ** 2):.2f} MB")

    positions_path = "F:/Synchrotron_MCT/21344/Dwell_ON_OFF/Dwell26/output"
    positions_filename = "centroid_positions.txt"
    positions = np.loadtxt(os.path.join(positions_path, positions_filename), delimiter='\t')

    out_path = os.path.join(image_path, "output/extracted_structure_grid")
    os.makedirs(out_path, exist_ok=True)

    circular_volume = extract_circular_regions(
        volume=image_volume,
        positions=positions,
        radius=10,
        out_path=out_path
    )
