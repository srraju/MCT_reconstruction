import os
from mct_segmentation_package.IO.IO import ImageHandler
from mct_segmentation_package.examples.Segmentation_Pipeline import Pipeline
from mct_segmentation_package.Filters.filters import Filters,get_image_dir
from mct_segmentation_package.mct_segmentation.plotting import visualize_with_slider
import tkinter as tk
import numpy as np


if __name__ == "__main__":
    #root = tk.Tk()
    #segment = Segmenter()  # You must define or import this
    #App()
    #root.mainloop()

    # Define input and output folders
    cwd = os.getcwd()
    root_dir = r"D:/Deakin/work/GrantWorks/DSTG/insituDynamicLoading/Dwell_MCT_analysis/MCT processing/OnsetPores/Dwell16onlyROIs/grayscale_subset_1_x1044_y1580_z1632"
    input_folder = root_dir #r"D:/Deakin/work/GrantWorks/DSTG/insituDynamicLoading/Dwell_MCT_analysis/MCT processing/OnsetPores/Dwell21onlyROIs/grayscale_subset_0_x527_y1042_z1119"
    label_folder = r"D:/Deakin/work/GrantWorks/DSTG/insituDynamicLoading/Dwell_MCT_analysis/Dwell16/output1000-3000/ROIs/grayscale_subset_1_x1044_y1580_z1632/output/labeled_images"
    positions_file_path = os.path.join(input_folder,"Positions.txt")
    batch_process=False

    read_grayscale_images=True
    read_labeled_images=False
    start_idx,end_idx,step = 0,6000,1

    # Initialize ImageHandler
    handler = ImageHandler(batch_process,input_folder)
    pipeline = Pipeline(
        root_dir=root_dir,
        image_handler=handler,
        separate_background=False,
        segment_images=True,
        do_watershed_3d=False,
        threshold_option=1,#(1. define thresholds, 2. low otsu, 3. select histogram)
        thr1=0, thr2=23100, thr3=23200, thr4=35000,
        roi=1, 
        chunksize=200,
        ncores=30,
        boxsize=100,
        extract_images=True,
        extract_label_images=False,
        positions_file_path=positions_file_path,X=50,Y=50,Z=50
    )

    if read_grayscale_images:
        handler.image_volume=handler.read_volume(input_folder)
        handler.image_volume=handler.image_volume[start_idx:end_idx:step]
        #visualize_with_slider(handler.image_volume,cmap='gray')
    if read_labeled_images:
        handler.label_volume=handler.read_volume(label_folder)

        #visualize_with_slider(handler.label_volume)

    # Save volume as a stack
    # Save volume as sequence of Images
    #handler.save_images(handler.image_volume)

    do_segmentation = True
    if do_segmentation:
        pipeline.run_segmentation()
        #visualize_with_slider(handler.label_volume)

    
    save_result = True
    if save_result:
        pipeline.save_segmented_data(save_image_volume=False, save_image_slices=False,save_segmented_volume=False,save_segmented_slices=True)

    # 2d or 3d measures
    measure_slices = True
    measure_3d_roi = False # True will error out (will not work on a the full volume, only on isolated volumes)
    
    if measure_slices:
        print("Performing slice measures")
        pipeline.run_measurements("2d")
    
    if measure_3d_roi:
        pipeline.run_measurements("3d")
    

    '''
    Morphological operations work on labeled volume.
    Apply morphological operations for calibrating the sample volume
    i.e. to determine the tilt of the sample w.r.to lab reference frame.
    Only do this on high speed computers with many cores.
    '''
    apply_morphological_operations = False

    if apply_morphological_operations:
        # for filters

        # Initialize filter with square kernel of size 5, 2 iterations
        filt = Filters(handler.label_volume, filtered_volume=None, kernel_shape='square', kernel_size=7, iterations=4,n_jobs=30)

        # Apply erosion
        filt.filtered_volume=filt.apply('erosion')
        #result = filt.fill_holes(threshold=0)
        output_path=os.path.join(get_image_dir(input_folder),"output","filtered_volume.tif")
        filt.save_volume(filt.filtered_volume, output_path)
        #filt.visualize_with_slider(filt.filtered_volume)
        #pipeline.segmenter.volume = result#[result[i].astype(np.uint16) for i in range(result.shape[0])]

    '''
    These methods isolate the voids volume and perform 2d and 3d measures on the isolated volume.
    The isolated volume is saved in the output folder.
    '''
    isolate_ROI = True
    if isolate_ROI:
        pipeline.extractROI.X=50
        pipeline.extractROI.Y=49
        pipeline.extractROI.Z=49
        
        if apply_morphological_operations:
            handler.label_volume=filt.filtered_volume
            
        else:
            pipeline.extractROI.label_folder_path=handler.label_directory
        
        print("\n Isolating volume:")
        pipeline.extractROI.isolate_volume()

        print("\n Performing 2d measures on isolated volume:")
        pipeline.extractROI.isolate_volume_measure2d()
        print("\n Performing 3d measures on isolated volume:")
        pipeline.extractROI.isolate_volume_measure3d()
        print("\n Printing segmented projection image:")
        pipeline.extractROI.plot_volume_rendered_projections()


    extract_roi = False
    if extract_roi:
        folder_path=handler.folder_path
        label_folder_path=pipeline.segmenter.labeled_slices_dir
        boxsize = 100
        extract_images = True
        extract_label_images = False
        pipeline.extractROIs()


    # Not implemented yet
    '''    # Visualize the isolated volume with ellipsoid vectors and volume with slider
    visualize_isolated_volume = False
    View_results = False
    if View_results:
        # Visualize the segmented volume with slider
        if handler.label_volume is not None:
            visualize_with_slider(handler.label_volume, cmap='jet')
        else:
            print("No labeled volume to visualize.")   

    '''
    #pipeline.run_all() #If we want to run all the below


    
