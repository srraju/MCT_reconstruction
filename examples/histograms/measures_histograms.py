import pandas as pd
import os
import matplotlib.pyplot as plt

input_folder = "D:/Deakin/work/GrantWorks/DSTG/insituDynamicLoading/Dwell_MCT_analysis/MCT processing/OnsetPores/Angleswrtozxy"
excel_filename = "Angles_wrto_zxy.xlsx"
sheet_name = "Dwell34"  
file_path = os.path.join(input_folder, excel_filename)

# Read Excel
df = pd.read_excel(file_path, sheet_name=sheet_name)

# List of columns to plot
columns_to_plot = [
    "Major Axis Angle with Calibration (deg)",
    "Intermediate Axis Angle with Calibration (deg)",
    "Minor Axis Angle with Calibration (deg)"
]

# Plot in a grid (m x n). Change plt.subplots(m, n, figsize=(20, 10))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()  # Flatten to 1D for easy iteration
label = ["Major Axis Angle (deg)","Intermediate Axis Angle (deg)","Minor Axis Angle (deg)"]
for i, column in enumerate(columns_to_plot):
    data = df[column].dropna()
    axes[i].hist(data, bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title("Dwell34")
    axes[i].set_xlabel(label[i])
    axes[i].set_ylabel("Frequency")
    axes[i].grid(False)
    axes[i].set_xlim(0, 90)  # Set x-axis limit
    axes[i].set_xticks(range(0, 91, 10))  # Set x-ticks
    axes[i].set_ylim(0, 20)  # Set y-axis limit
    axes[i].set_yticks(range(0,21, 5))  # Set y-ticks
plt.tight_layout()
plt.show()
