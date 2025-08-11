import os
from PIL import Image
from tkinter import messagebox
import pandas as pd
os.environ['SKIMAGE_NO_POOCH'] = '1'
os.environ['POOCH_BASE_DIR'] = r'C:/Raju/Codes/MCT_Analysis/mct_segmentation_package/scikit_cache'

#Force skimage cache to initialize now (avoids parallel crash)
from skimage import data
_ = data.camera()

from skimage import measure
import numpy as np
from skimage.io import imsave
import matplotlib
matplotlib.use('TkAgg')  # Force use of Tkinter backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
from skimage.color import label2rgb
import cc3d


class ExtractROIs:
    def __init__(self, root_dir,handler,segmenter, folder_path,label_folder_path,boxsize,extract_images,
        extract_label_images,positions_file_path,X,Y,Z,measure):
        # Initialize with root (for GUI), image_handler, and segment instances
        self.root_dir=root_dir
        self.handler = handler
        self.segment = segmenter
        self.measure = measure
        self.folder_path = folder_path
        self.basename=os.path.basename(self.folder_path)
        self.label_folder_path = label_folder_path
        self.ROIoutput_directory = os.path.join(self.folder_path,"rois") # for extracting rois
        self.boxsize = boxsize
        self.positions_file_path = positions_file_path
        self.extract_images = extract_images
        self.extract_label_images = extract_label_images
        
        self.isolated_volume_output_directory = None #os.path.join(self.segment.labeled_slices_dir,"isolated_volume")
        self.X = X
        self.Y = Y
        self.Z = Z
        self.isolated_volume = None
        self.isolated_volume_measure = None


    def isolate_volume_measure3d(self):
        #os.makedirs(self.label_folder_path, exist_ok=True)
        measure_output_directory=os.path.join(self.isolated_volume_output_directory,"output")
        os.makedirs(measure_output_directory,exist_ok=True)

        # Isolate volume and create binary volume
        binary_volume = self.isolated_volume > 0
        print(f"The roi being analysed: {self.measure.roi}")

        # Call the Feret_size_horizontal_vertical_3d method from segment class
        results=self.measure.Feret_size_horizontal_vertical_3d(binary_volume, self.handler.image_volume, self.measure.roi,measure_output_directory)
        # comment/ uncomment the next two lines to visualize the isolated volume overlayed with eigenvectors
        labeled_volume = measure.label(self.isolated_volume, connectivity=3)
        self.plot_eigenvectors_with_volume(results,labeled_volume,object_index=0)
        self.view_overlay_stack_matplotlib(self.handler.image_volume, self.isolated_volume, alpha=0.4)


    def isolate_volume_measure2d(self):
        #os.makedirs(self.label_folder_path, exist_ok=True)
        measure_output_directory=os.path.join(self.isolated_volume_output_directory,"output")
        os.makedirs(measure_output_directory,exist_ok=True)
        # Isolate volume and create binary volume
        binary_volume = self.isolated_volume > 0
        #self.handler.visualize_with_slider(binary_volume)
        # Call the Feret_size_horizontal_vertical_3d method from segment class
        self.measure.Feret_size_horizontal_vertical(binary_volume, self.measure.roi,measure_output_directory)
        #labeled_volume = measure.label(self.isolated_volume, connectivity=3)




    
    def isolate_volume(self):
        print('\n\nSelecting volume...')

        label_folder_path = self.label_folder_path
        if label_folder_path is None:
            label_folder_path = os.path.join(self.folder_path,"output/labeled_images")
        #output_directory = os.path.join(label_folder_path,'isolated_volume')
        self.isolated_volume_output_directory = os.path.join(label_folder_path,"isolated_volume")
        os.makedirs(self.isolated_volume_output_directory, exist_ok=True)
        print(f'Label folder: {label_folder_path}')
        print(f'output_directory:{self.isolated_volume_output_directory}')

        if not os.path.isdir(label_folder_path):
            messagebox.showerror("Error", "Please provide valid label path.")
            return
        
        # Read and process TIFF stack
        print('Reading labels')

        labeled_volume=self.handler.label_volume
        #labeled_files, labeled_volume = read_and_sort_tiff_stack(label_folder_path, 0, 2000)
        print(f'The shape of labeled volume is {labeled_volume.shape}')
        #labeled_volume = measure.label(labeled_volume, connectivity=3)
        labeled_volume = cc3d.connected_components(labeled_volume, connectivity=26)
        print(f"Unique labels after measure.label(): {np.unique(labeled_volume)}")
        print(f"Found {labeled_volume.max()} labels")

        x, y, z = self.X, self.Y, self.Z
        #target_label = labeled_volume[z, y, x]
        #print(f"Target label at ({x}, {y}, {z}): {target_label}")
        self.isolated_volume = self.isolate_component(labeled_volume,x,y,z)
        outname=os.path.join(self.isolated_volume_output_directory,f"{self.basename}_isolated_volume_{x,y,z}.tif")
        imsave(outname, self.isolated_volume,check_contrast=False)
    
    def isolate_component(self,labeled_volume, x, y, z):
        # Get the label at the specified coordinate
        target_label = labeled_volume[z, y, x]  # Note: numpy uses z, y, x indexing
        # Create a mask where only the target label is retained
        isolated_volume = np.where(labeled_volume == target_label, target_label, 0).astype(np.uint16)
        print(f"Isolated volume unique labels: {np.unique(isolated_volume)}")
        return isolated_volume


    def plot_eigenvectors(self,results, object_index=0):
        """
        Plot major, intermediate, and minor axis vectors of a labeled object 
        from the results of the calibrate() function.
        
        Parameters:
            results (list): Output list from the calibrate() function.
            object_index (int): Index of the object to visualize.
        """
        obj = results[object_index]
        
        # Extract centroid
        centroid = np.array([obj['Centroid X'], obj['Centroid Y'], obj['Centroid Z']])

        # Axis vectors and lengths
        axes = {
            'Major': (obj['Major Axis Vector'], obj['Major Axis Length'], 'r'),
            'Intermediate': (obj['Intermediate Axis Vector'], obj['Intermediate Axis Length'], 'g'),
            'Minor': (obj['Minor Axis Vector'], obj['Minor Axis Length'], 'b'),
        }

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the eigenvectors
        for name, (vector, length, color) in axes.items():
            # Scale the vector by half its length in both directions
            start = centroid - 0.5 * length * vector
            end = centroid + 0.5 * length * vector
            ax.quiver(
                centroid[0], centroid[1], centroid[2],
                vector[0], vector[1], vector[2],
                length=length,
                color=color,
                label=f"{name} Axis"
            )

        # Plot the image coordinate axes for reference
        axis_len = max(obj['Feret Size X'], obj['Feret Size Y'], obj['Feret Size Z']) * 0.6
        ax.quiver(centroid[0], centroid[1], centroid[2], axis_len, 0, 0, color='k', linestyle='dashed', label='X-axis')
        ax.quiver(centroid[0], centroid[1], centroid[2], 0, axis_len, 0, color='k', linestyle='dotted', label='Y-axis')
        ax.quiver(centroid[0], centroid[1], centroid[2], 0, 0, axis_len, color='k', linestyle='dashdot', label='Z-axis')

        ax.set_xlim([centroid[0] - axis_len, centroid[0] + axis_len])
        ax.set_ylim([centroid[1] - axis_len, centroid[1] + axis_len])
        ax.set_zlim([centroid[2] - axis_len, centroid[2] + axis_len])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Eigenvectors for Object #{obj['Label']}")
        ax.legend()
        plt.tight_layout()
        plt.show()



    def plot_eigenvectors_with_volume(self, results, segmented_volume, object_index=0):
        """
        Plot eigenvectors (major, intermediate, minor) of a labeled object 
        and overlay the 3D volume rendering of that object.

        Parameters:
            results (list): Output from calibrate() function.
            segmented_volume (3D np.ndarray): Labeled volume.
            object_index (int): Index of the object to visualize.
        """
        obj = results[object_index]
        label = obj['Label']

        # Mask for selected object
        region_mask = (segmented_volume == label)

        # Skip empty masks
        if np.max(region_mask) == 0:
            print("Selected object mask is empty.")
            return

        # Marching Cubes
        verts, faces, _, _ = marching_cubes(region_mask.astype(np.uint8), level=0.5)

        # Adjust coordinates: marching_cubes returns (Z, Y, X), but we need (X, Y, Z)
        verts = verts[:, [2, 1, 0]]

        # Plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot 3D surface
        mesh = Poly3DCollection(verts[faces], alpha=0.3, facecolor='gray')
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

        # Extract centroid
        centroid = np.array([obj['Centroid X'], obj['Centroid Y'], obj['Centroid Z']])

        # Axis vectors and lengths
        axes = {
            'Major': (obj['Major Axis Vector'], obj['Major Axis Length'], 'r'),
            'Intermediate': (obj['Intermediate Axis Vector'], obj['Intermediate Axis Length'], 'g'),
            'Minor': (obj['Minor Axis Vector'], obj['Minor Axis Length'], 'b'),
        }

        scale_factor = 4.0  # Adjust this to increase or decrease axis lengths

        # Plot the eigenvectors
        for name, (vector, length, color) in axes.items():
            scaled_length = length * scale_factor
            start = centroid - 0.5 * scaled_length * vector
            end = centroid + 0.5 * scaled_length * vector
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linewidth=3,
                label=f'{name} Axis'
            )

        # Coordinate axes for reference
        axis_len = max(obj['Feret Size X'], obj['Feret Size Y'], obj['Feret Size Z']) * 0.6
        fixed_axis_len = 10
        ax.quiver(centroid[0], centroid[1], centroid[2], fixed_axis_len, 0, 0, color='k', linestyle='dashed', label='X-axis')
        ax.quiver(centroid[0], centroid[1], centroid[2], 0, fixed_axis_len, 0, color='k', linestyle='dotted', label='Y-axis')
        ax.quiver(centroid[0], centroid[1], centroid[2], 0, 0, fixed_axis_len, color='k', linestyle='dashdot', label='Z-axis')

        ax.set_xlim(centroid[0] - fixed_axis_len, centroid[0] + fixed_axis_len)
        ax.set_ylim(centroid[1] - fixed_axis_len, centroid[1] + fixed_axis_len)
        ax.set_zlim(centroid[2] - fixed_axis_len, centroid[2] + fixed_axis_len)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Object #{label} with Eigenvectors')
        ax.legend()
        plt.tight_layout()
        plt.show()



    def view_overlay_stack_matplotlib(self,image_volume, segmented_volume, alpha=0.5):
        """
        Interactive grayscale + segmentation overlay viewer using matplotlib.

        Parameters:
        - image_volume: 3D numpy array (Z, Y, X), grayscale stack
        - segmented_volume: 3D numpy array (Z, Y, X), integer label stack
        - alpha: float (0 to 1), transparency of overlay
        """

        # Normalize grayscale image for display
        image_volume_norm = (image_volume - np.min(image_volume)) / (np.max(image_volume) - np.min(image_volume))
        num_slices = image_volume.shape[0]

        # Set up initial plot
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        # First overlay slice
        overlay = label2rgb(segmented_volume[0], image=image_volume_norm[0], alpha=alpha, bg_label=0, bg_color=None)
        img_display = ax.imshow(overlay)
        _, height, width = image_volume.shape
        center_x, center_y = width // 2, height // 2
        window = 20  # Number of pixels to show around center

        ax.set_xlim([center_x - window, center_x + window])
        ax.set_ylim([center_y + window, center_y - window])  # Note: y-axis is flipped in imshow

        ax.set_title(f'Slice 0')
        ax.axis('off')

        # Slider for navigating through slices
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slice_slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=0, valfmt='%0.0f')

        def update(val):
            z = int(slice_slider.val)
            overlay = label2rgb(segmented_volume[z], image=image_volume_norm[z], alpha=alpha, bg_label=0, bg_color=None)
            img_display.set_data(overlay)
            ax.set_title(f'Slice {z}')
            fig.canvas.draw_idle()

        slice_slider.on_changed(update)
        plt.show()



    def plot_orthogonal_projections(self):
        # Compute maximum intensity projections (MIP) along each axis
        mip_axial = np.max(self.isolated_volume, axis=0)  # Projection along the Z-axis (XY plane)
        mip_coronal = np.max(self.isolated_volume, axis=1)  # Projection along the Y-axis (XZ plane)
        mip_sagittal = np.max(self.isolated_volume, axis=2)  # Projection along the X-axis (YZ plane)

        # Plotting the projections
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Axial MIP (XY plane)
        axes[0].imshow(mip_axial, cmap="gray")
        axes[0].set_title('XY')
        axes[0].axis('off')

        # Coronal MIP (XZ plane)
        axes[1].imshow(mip_coronal, cmap="gray")
        axes[1].set_title('XZ')
        axes[1].axis('off')

        # Sagittal MIP (YZ plane)
        axes[2].imshow(mip_sagittal, cmap="gray")
        axes[2].set_title('YZ')
        axes[2].axis('off')

        plt.tight_layout()
        #plt.show()

    def plot_volume_rendered_projections(self):
        # Normalize the volume for visualization
        self.isolated_volume = (self.isolated_volume - self.isolated_volume.min()) / (self.isolated_volume.max() - self.isolated_volume.min())

        # Calculate projections along different axes
        xy_projection = np.sum(self.isolated_volume, axis=0)  # Projection along the Z-axis
        xz_projection = np.sum(self.isolated_volume, axis=1)  # Projection along the Y-axis
        yz_projection = np.sum(self.isolated_volume, axis=2)  # Projection along the X-axis

        # Set up figure and axes for the three projections
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # XY projection
        img1=axs[0].imshow(xy_projection, cmap="rainbow")
        axs[0].set_title("XY Projection (Top View)")
        axs[0].set_xlabel("X-axis")
        axs[0].set_ylabel("Y-axis")
        plt.colorbar(img1, ax=axs[0])

        # XZ projection
        img2=axs[1].imshow(xz_projection, cmap="rainbow")
        axs[1].set_title("XZ Projection (Side View)")
        axs[1].set_xlabel("X-axis")
        axs[1].set_ylabel("Z-axis")
        plt.colorbar(img2, ax=axs[1])

        # YZ projection
        img3=axs[2].imshow(yz_projection, cmap="rainbow")
        axs[2].set_title("YZ Projection (Front View)")
        axs[2].set_xlabel("Y-axis")
        axs[2].set_ylabel("Z-axis")
        plt.colorbar(img3, ax=axs[2])

        # Adjust layout and display the figure
        plt.tight_layout()
        #plt.show()
        figname=f"{self.basename}_orthoProjections_{self.X}_{self.Y}_{self.Z}"
        fig.savefig(os.path.join(self.segment.image_handler.output_folder,figname))
        print(f"Saved the volume rendered image: {os.path.join(self.segment.image_handler.output_folder,figname)}")

    def extract_rois(self):
        print(self.positions_file_path)

        positions = self.read_positions(self.positions_file_path)
        # Read and process TIFF stack
        try:
            if self.extract_images:
                volume = self.segment.image_volume
                grayscale_subsets = self.extract_subsets(volume, positions,self.boxsize)
                self.save_subsets(grayscale_subsets, self.ROIoutput_directory, prefix="grayscale")

            if self.extract_label_images:
                labeled_volume = self.segment.volume
                labeled_subsets = self.extract_subsets(labeled_volume, positions,self.boxsize)
                self.save_subsets(labeled_subsets, self.ROIoutput_directory, prefix="labeled")

            messagebox.showinfo("Success", "Processing and extraction completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def read_positions(self,positions_file_path):
        return pd.read_csv(positions_file_path, delim_whitespace=True, header=None, names=['x', 'y', 'z'])

    def extract_subsets(self,volume, positions, box_size):
        print(f'The box size is : {box_size}')
        half_size = box_size // 2
        subsets = []

        for _, row in positions.iterrows():
            x, y, z = int(row['x']), int(row['y']), int(row['z'])

            subset = volume[
                max(z - half_size, 0):min(z + half_size + 1, volume.shape[0]),
                max(y - half_size, 0):min(y + half_size + 1, volume.shape[1]),
                max(x - half_size, 0):min(x + half_size + 1, volume.shape[2])
            ]

            subsets.append((x, y, z, subset))

        return subsets

    def save_subsets(self, subsets, output_directory, prefix="subset"):
        for i, (x, y, z, subset) in enumerate(subsets):
            subset_folder = os.path.join(output_directory, f'{prefix}_subset_{i}_x{x}_y{y}_z{z}')
            os.makedirs(subset_folder, exist_ok=True)
            for j, img in enumerate(subset):
                Image.fromarray(img).save(os.path.join(subset_folder, f'image_{j}.tif'))

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)
    def browse_label_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.label_folder_path.set(folder)
    def browse_file(self):
        file = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file:
            self.positions_file.set(file)

    def browse_output_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.ROIoutput_directory.set(directory)

