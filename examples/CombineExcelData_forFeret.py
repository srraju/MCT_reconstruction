import os
import pandas as pd
import re
def extract_x_y_z(folder_name):
    """
    Extracts x, y, z integer values from a folder name of the form ...x123_y456_z789...
    Returns (x, y, z) or (None, None, None) if not found.
    """
    s = os.path.basename(folder_name)
    match = re.search(r'x(\d+)_y(\d+)_z(\d+)', s)

    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    else:
        return None, None, None



def combine_feret_excel(root_dir,path_to_feret_sizes_3d_eigen):
    """
    Combines all feret_sizes_3d_eigen.xlsx files from subfolders into one Excel file.
    """
    combined_df = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name,path_to_feret_sizes_3d_eigen)
        if not os.path.isdir(folder_path):
            continue

        file_path = os.path.join(folder_path, "feret_sizes_3d_eigen.xlsx")
        if os.path.exists(file_path):
            try:
                df = pd.read_excel(file_path)
                x, y, z = extract_x_y_z(folder_name)
                if None in (x, y, z):
                    print(f"Warning: Could not extract X, Y, Z from folder '{folder_name}'")

                df["Folder"] = folder_name  # Add folder name column
                combined_df.append(df)
                df["X"]=x
                df["Y"]=y
                df["Z"]=z
                print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
        else:
            print(f"Skipped (not found): {file_path}")

    if combined_df:
        final_df = pd.concat(combined_df, ignore_index=True)
        output_path = os.path.join(root_dir, "combined_feret_data.xlsx")
        final_df=final_df.sort_values(by='Z')
        final_df.to_excel(output_path, index=False)
        print(f"\nCombined Excel saved to: {output_path}")
    else:
        print("No valid Excel files found to combine.")


if __name__ == "__main__":
    root_directory = r"D:/Deakin/work/GrantWorks/DSTG/insituDynamicLoading/Dwell_MCT_analysis/MCT processing/OnsetPores/Dwell16onlyROIs" 
    path_to_feret_sizes_3d_eigen = r"output/labeled_images/isolated_volume/output"
    combine_feret_excel(root_directory, path_to_feret_sizes_3d_eigen)
