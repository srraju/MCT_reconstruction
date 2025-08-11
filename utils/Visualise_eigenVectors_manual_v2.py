import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.io import imread
import os

# Your eigenvectors and centroid (example from your results)
major_vector = np.array([-0.02235804, 0.693798, 0.71982251])
minor_vector = np.array([0.99932816, 0.03642377, -0.00406728])
intermediate_vector = np.array([-0.02904052, 0.71924797, -0.69414624])

# Optional: axis lengths if you want to scale arrows
major_length = 50
minor_length = 30
intermediate_length = 40

# Your centroid (X, Y, Z)
centroid = np.array([920, 1040, 988])  # Replace with your actual centroid

# Load the volume
volume_path = "F:/Synchrotron_MCT/21344/Dwell_ON_OFF/Dwell34/output/labeled_slices/isolated_volume/Dwell34_isolated_volume_(1000, 1000, 10)40001-6000.tif"
volume = imread(volume_path)

# Surface extraction
verts, faces, normals, _ = measure.marching_cubes(volume, level=0)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
mesh = Poly3DCollection(verts[faces], alpha=0.3)
mesh.set_facecolor('lightblue')
ax.add_collection3d(mesh)

# Plot eigenvectors
def draw_vector(vec, color, length, label):
    ax.quiver(*centroid, *(vec * length), color=color, linewidth=2, label=label)

draw_vector(major_vector, 'red', major_length, 'Major')
draw_vector(intermediate_vector, 'green', intermediate_length, 'Intermediate')
draw_vector(minor_vector, 'blue', minor_length, 'Minor')

# Annotate centroid
ax.scatter(*centroid, color='black', s=40, label='Centroid')

# Axis labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Set limits for better viewing
ax.set_xlim(0, volume.shape[2])
ax.set_ylim(0, volume.shape[1])
ax.set_zlim(0, volume.shape[0])

ax.legend()
plt.title("3D Volume with Eigen Axes")
plt.tight_layout()
plt.show()
