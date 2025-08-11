import pandas as pd
import os
import matplotlib.pyplot as plt

input_folder = r"D:/Deakin/work/GrantWorks/DSTG/insituDynamicLoading/Dwell_MCT_analysis/MCT processing/OnsetPores/Dwell21onlyROIs"
excel_filename = "combined_feret_data.xlsx"
file_path = os.path.join(input_folder,excel_filename)
df = pd.read_excel(file_path)

# Choose the column to plot
column_name = "Intermediate Axis Angle Z (deg)"  # Replace with the actual column name in your Excel
#df[column_name]=df[column_name]*(0.722**3)
# Plot histogram
plt.figure(figsize=(6, 5))
plt.hist(df[column_name].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title(f"{column_name} for voids in Dwell16")
plt.xlabel(column_name)
plt.ylabel("Frequency")
#plt.xlim(0,90)
#plt.xticks(range(0, 91, 10))  # Set x-ticks
plt.grid(True)
plt.tight_layout()
plt.show()