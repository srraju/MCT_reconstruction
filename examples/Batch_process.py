import os
import traceback
import pandas as pd
from mct_segmentation_package.IO.IO import ImageHandler
from mct_segmentation_package.examples.Segmentation_Pipeline import Pipeline

# Store batch results in a list for summary
batch_summary = []

def process_folder(root_dir,input_folder):
    result = {
        "Folder": os.path.basename(input_folder),
        "Status": "Started",
        "Error": ""
    }

    try:
        print(f"\nProcessing: {input_folder}")
        positions_file_path = os.path.join(input_folder, "Positions.txt")

        batch_process=True

        handler = ImageHandler(batch_process,input_folder)
        handler.select_files()
        handler.open_images()

        pipeline = Pipeline(root_dir=root_dir,
            image_handler=handler,
            separate_background=False,
            segment_images=True,
            do_watershed_3d=False, #Not implemented yet
            threshold_option=2,
            thr1=-200, thr2=10, thr3=10.1, thr4=27000,
            roi=1,
            chunksize=200,
            ncores=20,
            boxsize=100,
            extract_images=True,
            extract_label_images=False,
            positions_file_path=positions_file_path,
            X=50, Y=50, Z=50
        )

        pipeline.run_segmentation()
        pipeline.save_segmented_data()
        thresholds = pipeline.segmenter.thresholds
        for i in range(4):
            result[f"thr{i+1}"] = thresholds[i] if len(thresholds) > i else None

        # Optional measurements
        # pipeline.run_measurements("3d")
        # pipeline.run_measurements("2d")

        # ROI isolation
        pipeline.extractROI.X = 50
        pipeline.extractROI.Y = 50
        pipeline.extractROI.Z = 50
        pipeline.extractROI.label_folder_path = pipeline.segmenter.labeled_slices_dir
        pipeline.extractROI.isolate_volume()
        pipeline.extractROI.isolate_volume_measure3d()
        pipeline.extractROI.plot_volume_rendered_projections()

        result["Status"] = "Success"
        print(f"Completed: {input_folder}")

    except Exception as e:
        result["Status"] = "Failed"
        result["Error"] = str(e)
        print(f"Error processing {input_folder}: {e}")
        traceback.print_exc()

    batch_summary.append(result)


if __name__ == "__main__":
    root_dir = "D:/Deakin/work/GrantWorks/DSTG/insituDynamicLoading/Dwell_MCT_analysis/MCT processing/OnsetPores/Dwell21onlyROIs"
    output_summary_file = os.path.join(root_dir, "batch_summary.xlsx")

    # Collect all subfolders to process
    folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    # Process folders sequentially
    for folder in folders:
        process_folder(root_dir,folder)

    # Save summary report to Excel
    df = pd.DataFrame(batch_summary)
    df.to_excel(output_summary_file, index=False)
    print(f"\nBatch summary saved to: {output_summary_file}")
