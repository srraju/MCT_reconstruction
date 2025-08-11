import os
import numpy as np
from skimage.io import imread
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def read_image(index_path):
    index, full_path = index_path
    img = imread(full_path)
    img = np.squeeze(img)
    if img.ndim == 2:
        img = img[np.newaxis, ...]  # Make 2D image 3D (1 slice)
    elif img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape} in {full_path}")
    return index, img

def read_segmented_volume(path, parallel=True, max_threads=32, use_memmap=False, memmap_path=None):
    """
    Reads a 3D volume from a folder or single image file.
    Supports parallel loading and optional memory mapping for large volumes.
    """
    if os.path.isdir(path):
        filenames = sorted(
            [f for f in os.listdir(path) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))],
            key=natural_sort_key
        )
        if not filenames:
            raise ValueError(f"No image files found in folder: {path}")

        full_paths = [(i, os.path.join(path, fname)) for i, fname in enumerate(filenames)]

        print(f"Reading {len(full_paths)} images from folder: {path}")
        shape_hint = None

        volume_dict = {}

        if parallel:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = {executor.submit(read_image, item): item for item in full_paths}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Reading"):
                    index, img = future.result()
                    if shape_hint is None:
                        shape_hint = img.shape[1:]  # Assume all slices are same size
                    volume_dict[index] = img
        else:
            for index, full_path in tqdm(full_paths, desc="Reading"):
                index, img = read_image((index, full_path))
                if shape_hint is None:
                    shape_hint = img.shape[1:]
                volume_dict[index] = img

        sorted_keys = sorted(volume_dict)
        num_slices = sum(volume_dict[k].shape[0] for k in sorted_keys)

        if use_memmap:
            if memmap_path is None:
                memmap_path = os.path.join(path, "volume_memmap.dat")
            volume = np.memmap(memmap_path, dtype=img.dtype, mode='w+', shape=(num_slices, *shape_hint))
        else:
            volume = np.zeros((num_slices, *shape_hint), dtype=img.dtype)

        # Copy data into pre-allocated array/memmap
        slice_index = 0
        for k in sorted_keys:
            img_block = volume_dict[k]
            num = img_block.shape[0]
            volume[slice_index:slice_index + num] = img_block
            slice_index += num

        print(f"Loaded volume shape: {volume.shape} (Z, Y, X)")
        return volume

def convert_tiff_stack_to_npy(folder_path, output_path):
    volume=read_segmented_volume(folder_path,parallel=True,max_threads=32,use_memmap=False,memmap_path=None)
    #volume = np.concatenate(volume, axis=0)
    print(f"Saving volume of shape {volume.shape} to {output_path}")
    np.save(output_path, volume)
    print("Done.")

def convert_npy_to_tiff(npy_volume):

    #volume = np.load("your_volume.npy")
    imwrite("your_volume.tif", volume.astype(np.uint16))  # or np.uint8, depending on your data


if __name__ == "__main__":
    input_folder = "F:/Synchrotron_MCT/21344/Dwell_ON_OFF/Dwell31/output/labeled_slices/isolated_volume/"
    output_npy = os.path.join(input_folder, "segmented_volume.npy")
    convert_tiff_stack_to_npy(input_folder, output_npy)
    convert_npy_to_tiff(input_folder,npy_volume)
