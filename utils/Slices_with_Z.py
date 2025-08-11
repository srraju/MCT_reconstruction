import numpy as np
import tifffile
import os
from pathlib import Path

def create_stack_from_positions(txt_file_path, image_folder, outname, image_prefix='image_', image_ext='.tif'):
    output_path=os.path.join(image_folder,"output")
    os.makedirs(output_path, exist_ok=True)
    output_path=os.path.join(output_path,outname)

    # Load text file with X,Y,Z columns
    positions = np.loadtxt(txt_file_path, delimiter='\t')  # adjust delimiter if needed (e.g., '\t' or ' ')
    z_positions = np.unique(positions[:, 2].astype(int))  # Get unique Z values

    # Sort Z positions
    z_positions.sort()

    stack = []
    for z in z_positions:
        # Construct image filename with Z postfix
        filename = f"{image_prefix}{z:04d}{image_ext}"
        print(filename)
        file_path = Path(image_folder) / filename

        if not file_path.exists():
            print(f"Warning: Image {file_path} not found. Skipping.")
            continue

        image = tifffile.imread(str(file_path))
        stack.append(image)

    if stack:
        stack_array = np.stack(stack, axis=0)  # Z-stack
        tifffile.imwrite(output_path, stack_array)
        print(f"Stack saved to {output_path}")
    else:
        print("No images found. Stack not created.")



create_stack_from_positions(
    txt_file_path='F:/Synchrotron_MCT/21344/Dwell_ON_OFF/PoreRegistration/Dwell26Pores/Dwell34.txt',
    image_folder='F:/Synchrotron_MCT/21344/Dwell_ON_OFF/Dwell34',
    outname='Dwell34_Zs_stack.tif',
    image_prefix='Dwell34_',
    image_ext='.tif'      
)
