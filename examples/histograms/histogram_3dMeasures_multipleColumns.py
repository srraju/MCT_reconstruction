import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

input_folder = "D:/Deakin/work/GrantWorks/DSTG/insituDynamicLoading/Dwell_MCT_analysis/MCT processing/OnsetPores/Dwell34PoresROIs"
excel_filename = "combined_feret_data.xlsx"
file_path = os.path.join(input_folder,excel_filename)
df = pd.read_excel(file_path)

# Columns to overlay (numeric only)
columns_to_plot = [
    "Major Axis Angle Z (deg)",
    "Intermediate Axis Angle Z (deg)",
    "Minor Axis Angle Z (deg)"
]

plt.figure(figsize=(6, 4))

for col in columns_to_plot:
    sns.kdeplot(df[col].dropna(), label=col, linewidth=2)

plt.title("Overlay of Smooth Histograms (KDE)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

