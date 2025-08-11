import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Force use of Tkinter backend
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import re

def parse_vector(vec_str):
    # Insert commas between float numbers using regex
    cleaned = re.sub(r'(?<=\d)\s+(?=[-]?\d)', ', ', vec_str.strip())
    return np.array(eval(cleaned))

def plot_eigenvectors(results, object_index=0):
    """
    Plot major, intermediate, and minor axis vectors of a labeled object 
    from the results of the calibrate() function.
    
    Parameters:
        results (pd.DataFrame): DataFrame from the calibration results.
        object_index (int): Index of the object to visualize.
    """
    obj = results.iloc[object_index]
    
    # Extract centroid
    centroid = np.array([obj['Centroid X'], obj['Centroid Y'], obj['Centroid Z']])

    # Axis vectors and lengths
    axes = {
        'Major': (parse_vector(obj['Major Axis Vector']), obj['Major Axis Length'], 'r'),
        'Intermediate': (parse_vector(obj['Intermediate Axis Vector']), obj['Intermediate Axis Length'], 'g'),
        'Minor': (parse_vector(obj['Minor Axis Vector']), obj['Minor Axis Length'], 'b'),
    }


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the eigenvectors
    for name, (vector, length, color) in axes.items():
        start = centroid - 0.5 * length * vector
        end = centroid + 0.5 * length * vector
        ax.quiver(
            centroid[0], centroid[1], centroid[2],
            vector[0], vector[1], vector[2],
            length=length,
            color=color,
            label=f"{name} Axis"
        )

    # Plot reference axes
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

if __name__ == "__main__":
    # Read Excel file
    file_path = "F:/Synchrotron_MCT/21344/Dwell_ON_OFF/Dwell26/output/labeled_slices/Calibrate/tilt_calibration_results.xlsx"
    
    # Make sure engine is set correctly if needed
    results = pd.read_excel(file_path, engine='openpyxl')

    # Plot eigenvectors for the first object
    plot_eigenvectors(results, object_index=0)
