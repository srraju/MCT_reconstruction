import pandas as pd
import os
import matplotlib.pyplot as plt

input_folder = "D:/Deakin/work/GrantWorks/DSTG/insituDynamicLoading/Dwell_MCT_analysis/MCT processing/OnsetPores/Angleswrtozxy"
excel_filename = "Angles_wrto_zxy.xlsx"
sheet_name = ["Dwell16","Dwell21","Dwell26","Dwell31","Dwell33","Dwell34"]  
file_path = os.path.join(input_folder, excel_filename)

# Read Excel
df = pd.read_excel(file_path, sheet_name=sheet_name)

# List of columns to plot
columns_to_plot = [
    "Major Axis Angle Z (deg)",
    "Intermediate Axis Angle Z (deg)",
    "Minor Axis Angle Z (deg)"
]

# ploting histogram  in a grid per column overlayed with data from different sheets
fig, axes = plt.subplots(1, 3, figsize=(20, 7)) 
axes = axes.flatten()  # Flatten to 1D for easy iteration   
for i, column in enumerate(columns_to_plot):
    for sheet in sheet_name:
        data = df[sheet][column].dropna()
        axes[i].hist(data, bins=30, alpha=0.5, label=sheet, edgecolor='black')  # Overlay histograms
    axes[i].set_title(f"{column}")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Frequency")
    axes[i].grid(False)
    axes[i].set_xlim(0, 90)  # Set x-axis limit
    axes[i].set_xticks(range(0, 91, 10))  # Set x-ticks
    axes[i].legend()  # Add legend to distinguish sheets
plt.tight_layout()
plt.show()  