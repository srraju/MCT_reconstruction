import numpy as np
import itk
import pandas as pd
import argparse
import os

def read_tiff_stack(path):
    """Read a multi-page TIFF stack using ITK and convert to NumPy."""
    image = itk.imread(path, itk.UC)  # For segmented, force to unsigned char
    return image

def read_intensity_stack(path):
    """Read intensity TIFF stack as float32."""
    image = itk.imread(path, itk.F)  # For intensity, read as float
    return image

def feret_measurements_itk(segmented_path, intensity_path=None, output_excel='feret_measurements.xlsx'):
    # Load segmented volume (required)
    segmented_itk = read_tiff_stack(segmented_path)

    # Load intensity volume (optional)
    intensity_itk = None
    if intensity_path and os.path.exists(intensity_path):
        intensity_itk = read_intensity_stack(intensity_path)

    # Connected components
    cc_filter = itk.ConnectedComponentImageFilter.New(segmented_itk)
    cc_filter.Update()
    labeled_itk = cc_filter.GetOutput()

    # Relabel to sort by size
    relabel = itk.RelabelComponentImageFilter.New(labeled_itk)
    relabel.Update()
    labeled_itk = relabel.GetOutput()

    # Label shape statistics
    label_shape_filter = itk.LabelImageToShapeLabelMapFilter.New(labeled_itk)
    label_shape_filter.ComputeFeretDiameterOn()
    label_shape_filter.Update()
    label_map = label_shape_filter.GetOutput()

    # Optional intensity stats filter
    stats_filter = None
    if intensity_itk:
        stats_filter = itk.LabelStatisticsImageFilter.New(Input=intensity_itk, LabelInput=labeled_itk)
        stats_filter.Update()

    results = []

    for i in range(label_map.GetNumberOfLabelObjects()):
        obj = label_map.GetLabelObject(i)
        label = obj.GetLabel()
        volume = obj.GetPhysicalSize()
        centroid = obj.GetCentroid()
        bounding_box = obj.GetBoundingBox()
        feret_diameter = obj.GetFeretDiameter()
        min_feret = obj.GetMinimumFeretDiameter()

        result = {
            "Label": label,
            "Volume": volume,
            "Centroid_X": centroid[0],
            "Centroid_Y": centroid[1],
            "Centroid_Z": centroid[2],
            "FeretDiameter_Max": feret_diameter,
            "FeretDiameter_Min": min_feret,
            "BoundingBox_X": bounding_box[0],
            "BoundingBox_Y": bounding_box[1],
            "BoundingBox_Z": bounding_box[2],
            "BoundingBox_Width": bounding_box[3],
            "BoundingBox_Height": bounding_box[4],
            "BoundingBox_Depth": bounding_box[5]
        }

        if stats_filter and stats_filter.HasLabel(label):
            result["MeanIntensity"] = stats_filter.GetMean(label)
        elif intensity_itk:
            result["MeanIntensity"] = np.nan  # Label not found
        results.append(result)

    # Save to Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"Saved {len(results)} region measurements to {output_excel}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Region Analysis using ITK")
    parser.add_argument("--segmented", required=True, help="Path to segmented TIFF stack (binary or labeled)")
    parser.add_argument("--intensity", help="Path to intensity TIFF stack (optional)")
    parser.add_argument("--output", default="feret_measurements.xlsx", help="Output Excel file name")

    args = parser.parse_args()

    feret_measurements_itk(args.segmented, args.intensity, args.output)
