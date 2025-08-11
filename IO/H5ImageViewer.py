import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Tk, Label, Button, Entry, Frame, filedialog, StringVar, Canvas, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tifffile
import os
import re
from concurrent.futures import ThreadPoolExecutor
#import hdf5plugin


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def safe_read(dataset):
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode()
    return value

def browse_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("HDF5 files", "*.h5")])
    if file_paths:
        file_list.set("\n".join(sorted(file_paths, key=natural_sort_key)))
        num_files.set(f"Selected {len(file_paths)} master file(s)")

def read_MCT_h5_files():
    global datasets, metadata, dataset_shape, Image_file_names

    selected_files = file_list.get().split("\n")
    datasets.clear()
    metadata.clear()
    dataset_shape = None
    for file in selected_files:
        f = h5py.File(file, "r")  # keep file open
        datasets[file] = {
            "file": f,
            "data": f["/MCT/DATA"]
        }
        if dataset_shape is None:
            dataset_shape = datasets[file]["data"].shape

        metadata[file] = {
            "DETECTOR_TO_ORIGIN_DISTANCE": safe_read(f["MCT/CT_RECON_FDK/DETECTOR_TO_ORIGIN_DISTANCE"]),
            "Source to Detector distance": safe_read(f["MCT/CT_RECON_FDK/SOURCE_TO_DETECTOR_DISTANCE"]),
            "Source to Origin distance": safe_read(f["MCT/CT_RECON_FDK/SOURCE_TO_ORIGIN_DISTANCE"]),
            "BHC": safe_read(f["MCT/PRE_PROCESSING/BEAM_HARDENING_CORRECTION/C"]),
            "Energy": safe_read(f["MCT/PRE_PROCESSING/TIE_HOM/ENERGY"]),
            "Gamma": safe_read(f["MCT/PRE_PROCESSING/TIE_HOM/GAMMA"]),
            "R": safe_read(f["MCT/PRE_PROCESSING/TIE_HOM/R"]),
            "COR": safe_read(f["MCT/RAW_CORRECTION/FIND_COR/COMPUTED_COR"]),
            "Tilt angle": safe_read(f["MCT/RAW_CORRECTION/FIND_COR/COMPUTED_TILT_ANGLE"]),
            "Vert COR": safe_read(f["MCT/RAW_CORRECTION/FIND_COR/VERTICAL_OFFSET_PIXELS"])
        }

    Image_file_names = list(datasets.keys())
    num_files.set(f"Loaded {len(datasets)} master file(s)")

def update_image():
    global img_plot, colorbar, index
    try:
        vmin = float(min_val.get())
        vmax = float(max_val.get())
        index = int(image_index.get())

        first_file = next(iter(datasets))
        data = datasets[first_file]["data"]

        if index < 1 or index > data.shape[0]:
            messagebox.showinfo("Out of range", "Error: Image index out of range")
            image_index.delete(0, tk.END)
            image_index.insert(0, max(1, min(index, data.shape[0])))
            return

        img_data = data[index - 1]
        ax.clear()
        img_plot = ax.imshow(img_data, cmap="gray", vmin=vmin, vmax=vmax)
        if "colorbar" in globals() and colorbar:
            colorbar.remove()
        colorbar = fig.colorbar(img_plot, ax=ax, label="Intensity")
        ax.set_xticks([])
        ax.set_yticks([])

        canvas.draw()
        Image_file_name.set(Image_file_names[0])
        metadata_text.set("\n".join([f"{key}: {value}" for key, value in metadata[first_file].items()]))

    except Exception as e:
        print("Error displaying image:", e)


def prev_image():
    global index
    index -= 1
    image_index.delete(0, tk.END)
    image_index.insert(0, index)
    update_image()

def next_image():
    global index
    index += 1
    image_index.delete(0, tk.END)
    image_index.insert(0, index)
    update_image()

def convert_to_tiff():
    try:
        if not datasets:
            messagebox.showinfo("File list empty", "Error: Please load the files")
            return
        start_idx = int(start_index.get())
        end_idx = int(end_index.get())

        first_file = next(iter(datasets))
        data = datasets[first_file]["data"]
        dir_name = os.path.dirname(first_file)
        base_name = os.path.splitext(os.path.basename(first_file))[0]

        for idx in range(start_idx - 1, end_idx):
            img_data = data[idx]
            tifffile.imwrite(f"{os.path.join(dir_name, base_name)}_{idx:04d}.tif", img_data)

        print("TIFF conversion complete!")
    except Exception as e:
        print("Error converting to TIFF:", e)
def convert_to_tiff():
    try:
        if not datasets:
            messagebox.showinfo("File list empty", "Error: Please load the files")
            return
        start_idx = int(start_index.get())
        end_idx = int(end_index.get())

        first_file = next(iter(datasets))
        data = datasets[first_file]["data"]
        dir_name = os.path.dirname(first_file)
        base_name = os.path.splitext(os.path.basename(first_file))[0]

        for idx in range(start_idx - 1, end_idx):
            img_data = data[idx]
            tifffile.imwrite(f"{os.path.join(dir_name, base_name)}_{idx:04d}.tif", img_data)

        print("TIFF conversion complete!")
    except Exception as e:
        print("Error converting to TIFF:", e)


def convert_all_to_tiff():
    if not datasets:
        messagebox.showinfo("File list empty", "Error: Please load the files")
        return

    def convert_one(file_path, data, base_name, dir_name):
        try:
            for idx in range(data.shape[0]):
                out_name = f"{os.path.join(dir_name, base_name)}_{idx:04d}.tif"
                tifffile.imwrite(out_name, data[idx])
        except Exception as e:
            print(f"Error converting {file_path} to TIFF: {e}")

    with ThreadPoolExecutor() as executor:
        for file_path, meta in datasets.items():
            data = meta["data"]
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            dir_name = os.path.dirname(file_path)
            executor.submit(convert_one, file_path, data, base_name, dir_name)

    messagebox.showinfo("Conversion Complete", "All datasets converted to TIFF.")

def App():
    global file_list, num_files, image_index, index, min_val, max_val
    global metadata_text, Image_file_name, start_index, end_index, datasets, metadata, fig, ax, canvas

    root = Tk()
    root.title("MCT H5 Image Viewer")

    Label(root, text="MCT reconstructio\nH5 Image Viewer", font=("Arial", 14)).grid(row=0, column=0, columnspan=4)

    Label(root, text="Open H5 file(s):").grid(row=1, column=0)
    file_list = StringVar()
    num_files = StringVar()
    Label(root, textvariable=num_files, width=50, anchor="w").grid(row=1, column=1)

    Button(root, text="Browse", command=browse_files).grid(row=1, column=2)
    Button(root, text="Read", command=read_MCT_h5_files).grid(row=1, column=3)

    display_frame = Frame(root)
    display_frame.grid(row=3, column=0, columnspan=4)
    image_index_var = StringVar(value="1")
    min_val_var = StringVar(value="0")
    max_val_var = StringVar(value="20")

    Label(display_frame, text="Image Index:").grid(row=0, column=0)
    image_index = Entry(display_frame, width=5, textvariable=image_index_var)
    image_index.grid(row=0, column=1)
    index = int(image_index.get())

    Label(display_frame, text="Min:").grid(row=0, column=2)
    min_val = Entry(display_frame, width=5, textvariable=min_val_var)
    min_val.grid(row=0, column=3)
    Label(display_frame, text="Max:").grid(row=0, column=4)
    max_val = Entry(display_frame, width=5, textvariable=max_val_var)
    max_val.grid(row=0, column=5)
    Button(display_frame, text="View", command=update_image).grid(row=0, column=6)
    Button(display_frame, text="←", font=("Arial", 14), command=prev_image).grid(row=0, column=7)
    Button(display_frame, text="→", font=("Arial", 14), command=next_image).grid(row=0, column=8)
    Image_file_name = StringVar()
    Label(display_frame, textvariable=Image_file_name, width=50, anchor="w").grid(row=1, column=1, columnspan=8)

    canvas_frame = Frame(root)
    canvas_frame.grid(row=4, column=0, columnspan=4)
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack()

    tiff_frame = Frame(root)
    tiff_frame.grid(row=5, column=0, columnspan=4)
    Label(tiff_frame, text="Convert to TIFF:\tStart ").grid(row=0, column=0)
    start_index = Entry(tiff_frame, width=5)
    start_index.grid(row=0, column=1)
    Label(tiff_frame, text="End ").grid(row=0, column=2)
    end_index = Entry(tiff_frame, width=5)
    end_index.grid(row=0, column=3)
    Button(tiff_frame, text="Convert", command=convert_to_tiff).grid(row=0, column=4)
    Button(tiff_frame, text="Convert All to tiff", command=convert_all_to_tiff).grid(row=0, column=5)

    metadata_frame = Frame(root)
    metadata_frame.grid(row=6, column=0, columnspan=4)
    Label(metadata_frame, text="Metadata:").grid(row=0, column=0)
    metadata_text = StringVar()
    Label(metadata_frame, textvariable=metadata_text, justify="left", anchor="w").grid(row=1, column=0, columnspan=4)
    Button(metadata_frame, text="Exit", command=root.quit).grid(row=2, column=0, columnspan=4)

    datasets = {}
    metadata = {}

    root.mainloop()

if __name__ == "__main__":
    App()
