from skimage.measure import regionprops_table
import pandas as pd
import numpy as np
import time
import os
import pandas as pd
os.environ['SKIMAGE_NO_POOCH'] = '1'
os.environ['POOCH_BASE_DIR'] = r'C:/Raju/Codes/MCT_Analysis/mct_segmentation_package/scikit_cache'
from skimage import measure
#from ellipsoid_fit import ellipsoid_fit  # Ensure this function is accessible

#Force skimage cache to initialize now (avoids parallel crash)
from skimage import data
_ = data.camera()
import cc3d
from skimage import measure
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

def process_volume_chunk_eigen(volume_chunk, intensity_chunk, chunk_index, z_index):
    #labeled_volume = measure.label(volume_chunk, connectivity=3)
    labeled_volume=cc3d.connected_components(volume_chunk, connectivity=26)
    props = measure.regionprops(labeled_volume)
    print(f'Number of slices in labeled volume: {labeled_volume.shape[0]}')
    # Picking intensities from the label region only
    masked_intensity_chunk = intensity_chunk * volume_chunk

    # Assuming `volume_chunk` is binary or labeled, and `intensity_chunk` holds the intensity values

    (voxel_size_x,voxel_size_y,voxel_size_z)=(1,1,1)
    results = []

    for prop in props:
        minz, miny, minx, maxz, maxy, maxx = prop.bbox

        # Ignore single-pixel ROIs along the Z-axis
        if maxz - minz <= 1:
            continue

        feret_x = maxx - minx  # Width in the X-axis
        feret_y = maxy - miny  # Height in the Y-axis
        feret_z = maxz - minz  # Depth in the Z-axis

        #pore volume
        pore_volume_voxel_count = prop.area
        voxel_volume = voxel_size_x * voxel_size_y * voxel_size_z
        pore_volume = pore_volume_voxel_count * voxel_volume


        # Compute the inertia tensor and its eigenvalues/eigenvectors
        #eigenvalues, eigenvectors = np.linalg.eigh(prop.inertia_tensor)
        #sorted_eigenvalues = np.sort(eigenvalues)
        # Compute the inertia tensor and its eigenvalues/eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(prop.inertia_tensor)
        sorted_eigenvalues = np.sort(eigenvalues)

        # Clamp small negative values to 0
        sorted_eigenvalues = np.clip(sorted_eigenvalues, a_min=0, a_max=None)

        major_axis_vector = eigenvectors[:, np.argmax(eigenvalues)]
        minor_axis_vector = eigenvectors[:, np.argmin(eigenvalues)]
        intermediate_axis_vector = eigenvectors[:, np.argsort(eigenvalues)[1]]

        # Calculate axis lengths based on eigenvalues
        major_axis_length = 2 * np.sqrt(sorted_eigenvalues[2])
        intermediate_axis_length = 2 * np.sqrt(sorted_eigenvalues[1])
        minor_axis_length = 2 * np.sqrt(sorted_eigenvalues[0])

        # Calculate angle of the axes with respect to the Z direction
        major_axis_angle_z = np.arccos(np.abs(major_axis_vector[0])) * (180 / np.pi)  # using major_axis_vector[0] for Z
        intermediate_axis_angle_z = np.arccos(np.abs(intermediate_axis_vector[0])) * (180 / np.pi)  # using intermediate_axis_vector[0] for Z
        minor_axis_angle_z = np.arccos(np.abs(minor_axis_vector[0])) * (180 / np.pi)  # using minor_axis_vector[0] for Z

        # Volume and Surface Area
        volume = (maxy - miny) * (maxx - minx) * (maxz - minz)
        # Compute surface area using the marching cubes algorithm
        region_mask = labeled_volume == prop.label
        verts, faces, _, _ = measure.marching_cubes(region_mask, level=0)
        surface_area = measure.mesh_surface_area(verts, faces)

        # Anisotropy (measure of elongation)
        anisotropy = major_axis_length / minor_axis_length if minor_axis_length != 0 else 0

        # Compactness (comparison with an ellipsoid)
        compactness = (volume**(2/3)) / surface_area if surface_area > 0 else 0

        # Eccentricity (degree to which the shape deviates from a sphere)
        eccentricity = np.sqrt(1 - (minor_axis_length**2 / major_axis_length**2)) if major_axis_length != 0 else 0

        # Flatness (comparison of intermediate axis with minor axis)
        flatness = intermediate_axis_length / minor_axis_length if minor_axis_length != 0 else 0

        # Elongation (comparison of major axis with minor axis)
        elongation = major_axis_length / minor_axis_length if minor_axis_length != 0 else 0

        # Sphericity (measure of how spherical the shape is)
        sphericity = (6 * volume**(2/3)) / surface_area if surface_area != 0 else 0

        # Moments of inertia (eigenvalues are the moments of inertia)
        moments_of_inertia = sorted_eigenvalues
        # Calculate intensity statistics
        # Update masked intensity calculation to apply to the segmented regions only
        masked_intensity_region = masked_intensity_chunk[minz:maxz, miny:maxy, minx:maxx]
        masked_intensity_values = masked_intensity_region[labeled_volume[minz:maxz, miny:maxy, minx:maxx] == prop.label]

        intensity_max = masked_intensity_values.max() if masked_intensity_values.size > 0 else 0
        intensity_min = masked_intensity_values.min() if masked_intensity_values.size > 0 else 0
        intensity_mean = masked_intensity_values.mean() if masked_intensity_values.size > 0 else 0
        intensity_std = masked_intensity_values.std() if masked_intensity_values.size > 0 else 0


        results.append({
            'Chunk Index': chunk_index,
            'Slice': minz+z_index,
            'Label': prop.label,
            'Centroid X': prop.centroid[2],  # X-axis
            'Centroid Y': prop.centroid[1],  # Y-axis
            'Centroid Z': prop.centroid[0] + z_index,  # Z-axis
            'x1': minx,
            'x2': maxx,
            'y1': miny,
            'y2': maxy,
            'z1': minz,
            'z2': maxz,
            'Feret Size X': feret_x,
            'Feret Size Y': feret_y,
            'Feret Size Z': feret_z,
            'Pore volume' : pore_volume,
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
            'Moments of Inertia': moments_of_inertia,
            'intensity_max' : intensity_max,
            'intensity_min' : intensity_min,
            'intensity_mean' : intensity_mean,
            'intensity_std' : intensity_std
        })
    return results

# process_volume_chunk uses ellipsoid_fit to calculate the eigenvalues and eigenvectors
def process_volume_chunk(volume_chunk, intensity_chunk, chunk_index, z_index):
    labeled_volume = measure.label(volume_chunk, connectivity=3)
    props = measure.regionprops(labeled_volume)
    print(f'Number of slices in labeled volume: {labeled_volume.shape[0]}')

    masked_intensity_chunk = intensity_chunk * volume_chunk

    (voxel_size_x, voxel_size_y, voxel_size_z) = (1, 1, 1)
    results = []

    '''def angle_with_axis(vec1, axis):
        cos_theta = np.dot(vec1, axis) / (np.linalg.norm(vec1) * np.linalg.norm(axis))
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))'''
    
    def angle_with_axis(vec1, axis):
        cos_theta = np.dot(vec1, axis) / (np.linalg.norm(vec1) * np.linalg.norm(axis))
        angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return min(angle_deg, 180 - angle_deg)  # ensures angle is in [0, 90]°

    # Define the Z-axis for angle calculations
    # [0,0,1] for angle w.r.to. lab reference frame, or
    # define the numpy array for sample reference frame
    z_axis = np.array([0, 0, 1])
    #z_axis = np.array([-0.0041133,   0.0258497,   0.99965738])

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
        points = np.stack([x, y, z], axis=1)
        MAX_POINTS = 500_000

        if points.shape[0] > MAX_POINTS:
            #continue

            print(f"Subsampling label {prop.label} from {points.shape[0]} to {MAX_POINTS} points")
            idx = np.random.choice(points.shape[0], MAX_POINTS, replace=False)
            points = points[idx]

        center = points.mean(axis=0)
        centered_points = points - center
        cov = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)
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

        masked_intensity_region = masked_intensity_chunk[minz:maxz, miny:maxy, minx:maxx]
        masked_intensity_values = masked_intensity_region[labeled_volume[minz:maxz, miny:maxy, minx:maxx] == prop.label]

        intensity_max = masked_intensity_values.max() if masked_intensity_values.size > 0 else 0
        intensity_min = masked_intensity_values.min() if masked_intensity_values.size > 0 else 0
        intensity_mean = masked_intensity_values.mean() if masked_intensity_values.size > 0 else 0
        intensity_std = masked_intensity_values.std() if masked_intensity_values.size > 0 else 0

        results.append({
            'Chunk Index': chunk_index,
            'Slice': minz + z_index,
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
            'intensity_max': intensity_max,
            'intensity_min': intensity_min,
            'intensity_mean': intensity_mean,
            'intensity_std': intensity_std
        })

    return results

class Measure:
    def __init__(self, image_handler, segmenter,roi, ncores=4, chunksize=20, checkfolder=False, outfolder=None):
        self.image_handler = image_handler
        self.segmenter=segmenter
        self.roi = roi
        self.ncores = ncores
        self.chunksize = chunksize
        self.checkfolder=checkfolder
        self.outfolder=outfolder


    def Feret_size_horizontal_vertical_Calc(self):
        print('Performing slice measures')
        self.Feret_size_horizontal_vertical(self.segmenter.volume,self.roi,output_folder=self.image_handler.output_folder)

    def Feret_size_horizontal_vertical_3d_Calc(self):
        print('Performing 3d measures')
        self.Feret_size_horizontal_vertical_3d(self.segmenter.volume,self.segmenter.image_volume,self.roi)

    def Feret_size_horizontal_vertical(self, volume, roi,output_folder=None):
        t2=time.time()
        if output_folder is None:
            output_folder = self.image_handler.output_folder
        # Processing 2d slices
        output_filename = os.path.join(output_folder, 'feret_sizes.xlsx')
        writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')
        binary_volume = (volume == roi).astype(np.uint8)
        print(f"The shape of binary volume is {binary_volume.shape}")
        roi_list=[]
        slice_list=[]
        labels_list = []
        centroid_x_list = []
        centroid_y_list = []
        feret_x_list = []
        feret_y_list = []
        area_list = []
        x1_list=[]
        x2_list=[]
        y1_list=[]
        y2_list=[]
        for z in range(binary_volume.shape[0]):
            slice_2d = binary_volume[z, :, :]
            num_labels, labelsimg, stats, centroids = cv2.connectedComponentsWithStats(slice_2d)
            
            for i in range(0, num_labels):  # Start from 1 to skip the background
                stat = stats[i].tolist()
                centroid = centroids[i].tolist()

                x1 = stats[i][cv2.CC_STAT_LEFT]
                x2 = x1 + stats[i][cv2.CC_STAT_WIDTH]
                y1 = stats[i][cv2.CC_STAT_TOP]
                y2 = y1 + stats[i][cv2.CC_STAT_HEIGHT]
                area = stats[i][cv2.CC_STAT_AREA]
                
                feret_x = x2 - x1
                feret_y = y2 - y1
                roi_list.append(roi)
                slice_list.append(z)
                labels_list.append(i)
                centroid_x_list.append(centroids[i][0])
                centroid_y_list.append(centroids[i][1])
                x1_list.append(x1)
                x2_list.append(x2)
                y1_list.append(y1)
                y2_list.append(y2)
                feret_x_list.append(feret_x)
                feret_y_list.append(feret_y)
                area_list.append(area)

                
        df = pd.DataFrame({
            'ROI': roi_list,
            'Slice': slice_list,
            'Label': labels_list,
            'Centroid X': centroid_x_list,
            'Centroid Y': centroid_y_list,
            'x1': x1_list,
            'x2': x2_list,
            'y1': y1_list,
            'y2': y2_list,
            'Feret Size X': feret_x_list,
            'Feret Size Y': feret_y_list,
             'Area': area_list
            })
        df = df.sort_values(by=['Slice', 'Label'])
        df.to_excel(writer, sheet_name='Statistics', index=False)

        writer.close()
        print(f'Processing time for 2d slice measures: {(time.time()-t2)/60} mins')
        print(f'The slice measures are exported to {output_filename}')

    def Feret_size_horizontal_vertical_3d(self, volume, intensity_volume, roi,output_folder=None):
        t3=time.time()
        if output_folder is None:
            output_folder = self.image_handler.output_folder
        print(f"The roi value in measures:{roi}")

        output_filename = os.path.join(output_folder, 'feret_sizes_3d_eigen.xlsx')
        writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')

        volume = (volume == roi).astype(np.uint8)
        print(f"The max number in volume is {volume.max()}")



        ChunkNumber = volume.shape[0] // self.chunksize
        remainder = volume.shape[0] % self.chunksize

        if ChunkNumber == 0:
            print("Only one set; skipping parallel processing.")
            # Process the whole volume in a single batch if needed
            results = process_volume_chunk(volume, intensity_volume, 0, 0)
        else:
            # Split the volume along the Z-axis
            volume_chunks = []
            intensity_chunks = []
            z_indices = []
            start_idx = 0

            for i in range(ChunkNumber):
                end_idx = start_idx + self.chunksize
                
                # If it's the last chunk and has fewer than 10 slices, merge with the previous chunk
                if i == ChunkNumber - 1 and (volume.shape[0] - start_idx) < 10:
                    volume_chunks[-1] = np.concatenate((volume_chunks[-1], volume[start_idx:end_idx, :, :]), axis=0)
                    intensity_chunks[-1] = np.concatenate((intensity_chunks[-1], intensity_volume[start_idx:end_idx, :, :]), axis=0)
                    z_indices[-1] = start_idx  # Update the last chunk's start index
                else:
                    volume_chunks.append(volume[start_idx:end_idx, :, :])
                    intensity_chunks.append(intensity_volume[start_idx:end_idx, :, :])
                    z_indices.append(start_idx)  # Record the starting Z index for this chunk)
                
                start_idx = end_idx

            # If there are remainder slices, check their size
            if remainder > 0:
                # If the last set has less than 10 slices, merge it with the previous chunk
                if (volume.shape[0] - start_idx) < 10:
                    volume_chunks[-1] = np.concatenate((volume_chunks[-1], volume[start_idx:, :, :]), axis=0)
                    intensity_chunks[-1] = np.concatenate((intensity_chunks[-1], intensity_volume[start_idx:, :, :]), axis=0)
                    #z_indices[-1] = z_index[-1]  # Update the last chunk's start index
                else:
                    volume_chunks.append(volume[start_idx:, :, :])
                    intensity_chunks.append(intensity_volume[start_idx:, :, :])
                    z_indices.append(start_idx)  # Record the starting Z index for the last chunk

            print(f'The shape of volume: {volume.shape}\nChunk size: {self.chunksize}\nNumber of chunks: {len(volume_chunks)}')

            # including intensity image
            results = Parallel(n_jobs=self.ncores, backend='threading')(
                delayed(process_volume_chunk)(chunk, intensity_chunk, idx, z_indices[idx]) for idx, (chunk, intensity_chunk) in enumerate(zip(volume_chunks, intensity_chunks))
            )

            results = [item for sublist in results for item in sublist if item is not None]

        print('Measuring the properties: done\n\n')

        df = pd.DataFrame(results)

        if df.empty:
            print("Warning: No objects detected. DataFrame is empty. Skipping Feret calculation.")
            writer.close()
            return

        if 'Centroid Z' not in df.columns:
            print("Warning: 'Centroid Z' column not found in DataFrame. Skipping Feret calculation.")
            writer.close()
            return

            
        df = df.sort_values(by=['Centroid Z'])
        df.to_excel(writer, sheet_name='Statistics', index=False)
        writer.close()
        print(f'The 3d measures are exported to {output_filename}')
        print(f'Processing time for 3d measures: {(time.time()-t3)/60} mins')

        # Store centroids for ROI export
        #self.centroids = [(res['Centroid X'], res['Centroid Y'], res['Centroid Z']) for res in results]
        self.centroids = sorted([(res['Centroid X'], res['Centroid Y'], res['Centroid Z']) for res in results], key=lambda x: x[2])

        # Store centroids for ROI export
        if self.checkfolder:
            centroids_file = os.path.join(outfolder, 'centroids.txt')
            self.checkfolder = False
        else:
            centroids_file = os.path.join(output_folder, 'centroids.txt')
        with open(centroids_file, 'w') as f:
            for centroid in self.centroids:
                f.write(f"{centroid[0]}\t{centroid[1]}\t{centroid[2]}\n")

        print(f'Centroids exported to {centroids_file}')
        return(results)

    def Feret_size_horizontal_vertical_3d_check(self, volume, intensity_volume, roi):
        t3=time.time()

        output_folder = self.image_handler.output_folder
        output_filename = os.path.join(output_folder, 'feret_sizes_3d_eigen.xlsx')
        writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')

        volume = (volume == roi).astype(np.uint8)
        labeled_volume=cc3d.connected_components(volume, connectivity=26)


        props_table = regionprops_table(labeled_volume, properties=['label', 'slice','area', 'centroid', 'bbox',
        'inertia_tensor', 'inertia_tensor_eigvals',
        'orientation', 'eccentricity', 'solidity',
        'equivalent_diameter'])
        props_table_df = pd.DataFrame(props_table)
        print(f"Time taken for props table is {time.time()-t3}")
        props_table_df.to_excel(writer, sheet_name='Statistics', index=False)
        writer.close()


    '''
 
    def export_ROIs(self, positions_file_path):
        df = pd.read_csv(positions_file_path, sep='\t', header=None, names=['X', 'Y', 'Z'])
        positions = df.values.tolist()
        roi_images = []
        for centroid in self.centroids:
            x_center, y_center, z_center = centroid
            for position in positions:
                if abs(position[0] - x_center) <= 5 and abs(position[1] - y_center) <= 5 and abs(position[2] - z_center) <= 5:
                    x_start = max(0, int(x_center) - 5)
                    x_end = min(512, int(x_center) + 5)  # Assuming image size is 512x512
                    y_start = max(0, int(y_center) - 5)
                    y_end = min(512, int(y_center) + 5)
                    z_start = max(0, int(z_center) - 5)
                    z_end = min(len(self.segmented_images), int(z_center) + 5)
                    for i in range(z_start, z_end):
                        image = list(self.segmented_images.values())[i]
                        roi = image[y_start:y_end, x_start:x_end]
                        roi_images.append(roi)
                        print(roi_images)

        for i, roi in enumerate(roi_images):
            output_filename = f'ROI_{i}.tif'
            output_path = os.path.join(self.image_handler.output_folder, output_filename)
            imsave(output_path, roi)
            print(f"ROI {i} saved to {output_path}")
        '''
    def save_segmented_slices(self):
        labeled_slices_dir=os.path.join(self.image_handler.output_folder, 'labeled_slices')
        os.makedirs(labeled_slices_dir, exist_ok=True)
        volume = np.stack(self.segmented_images,axis=2)
        volume_8bit = (volume * 256 / volume.max()).astype(np.uint8)

        print(f'Saving slices in {labeled_slices_dir}')
        if self.segmented_images:
            for z in range(len(self.segmented_images)):
                filename_with_extension = os.path.basename(self.image_handler.file_names[z])
                file_name, extension = os.path.splitext(filename_with_extension)
                match = re.search(r'\d+$',file_name)
                if match:
                    digits = match.group()
                    new_filename = f"labeled_slices_{digits}{extension}"
                    new_path = os.path.join(labeled_slices_dir, new_filename)
                slice_2d = volume_8bit[:, :, z]
                print(new_path)
                imageio.imwrite(new_path, slice_2d)

        print('Saving labelled slices: done')            

    def save_segmented_stack(self):
        print('Saving segmented images as stack...')
        output_filename = 'segmented_images_stack.tif'
        output_path = os.path.join(self.image_handler.output_folder, output_filename)
        #self.segmenter.volume = np.stack(self.segmented_images,axis=0)
        self.segmenter.volume_8bit = (self.segmenter.volume * 256 / self.segmenter.volume.max()).astype(np.uint8)
        tiff.imwrite(output_path, self.segmenter.volume_8bit, photometric='minisblack')
        print(f'Saving segmented images as stack: {output_filename} : done')
            
    def save_segmented_stacks(self):
        print("Saving segmented stacks as TIFF files")
        for label, stack in self.segmented_stacks.items():
            self.image_handler.save_volume_stack(np.array(stack), axis='z')
            #self.image_handler.save_volume_stack(np.array(stack), axis='x')
            #self.image_handler.save_volume_stack(np.array(stack), axis='y')

