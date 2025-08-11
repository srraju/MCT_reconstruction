import pandas as pd
import numpy as np
import re

def parse_vector(vec_str):
    # Insert commas between float numbers using regex
    cleaned = re.sub(r'(?<=\d)\s+(?=[-]?\d)', ', ', vec_str.strip())
    return np.array(eval(cleaned))
def angle_with_z_axis(vector):
    z_axis = np.array([0, 0, 1])
    vector = np.array(vector)
    cos_theta = np.dot(vector, z_axis) / (np.linalg.norm(vector) * np.linalg.norm(z_axis))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# Load CSV or DataFrame with vectors
file_path="D:/Deakin/work/GrantWorks/DSTG/insituDynamicLoading/Dwell_MCT_analysis/Dwell26/output/labeled_slices/Calibrate/Dwell26_tilt_calibration_results.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Assume vectors are stored as strings like "[0.2, 0.5, 0.8]"
df['Major Axis Vector'] = df['Major Axis Vector'].apply(parse_vector)
df['Intermediate Axis Vector'] = df['Intermediate Axis Vector'].apply(parse_vector)
df['Minor Axis Vector'] = df['Minor Axis Vector'].apply(parse_vector)

# Recalculate correct angles
df['Major Axis Angle Z (deg)'] = df['Major Axis Vector'].apply(angle_with_z_axis)
df['Intermediate Axis Angle Z (deg)'] = df['Intermediate Axis Vector'].apply(angle_with_z_axis)
df['Minor Axis Angle Z (deg)'] = df['Minor Axis Vector'].apply(angle_with_z_axis)

# Save corrected results
df.to_csv("corrected_angles.csv", index=False)
print("Corrected angles saved to 'corrected_angles.csv'")
