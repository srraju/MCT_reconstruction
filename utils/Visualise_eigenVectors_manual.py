import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter instead of Qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Manually input your eigenvectors here
major_vector = np.array([-0.02235804,  0.693798,    0.71982251])
minor_vector = np.array([ 0.99932816,  0.03642377, -0.00406728])
intermediate_vector = np.array([-0.02904052,  0.71924797, -0.69414624])

# Optional: scale the vectors to make them more visible
major_length = 1.0
intermediate_length = 1.0
minor_length = 1.0

# Optional: you can change the origin here
origin = np.array([0, 0, 0])

# Create a 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the eigenvectors
ax.quiver(*origin, *(major_length * major_vector), color='r', label='Major Axis')
ax.quiver(*origin, *(intermediate_length * intermediate_vector), color='g', label='Intermediate Axis')
ax.quiver(*origin, *(minor_length * minor_vector), color='b', label='Minor Axis')

# Plot the standard X, Y, Z axes
axis_len = 1.2
ax.quiver(0, 0, 0, axis_len, 0, 0, color='k', linestyle='dashed', label='X-axis')
ax.quiver(0, 0, 0, 0, axis_len, 0, color='k', linestyle='dotted', label='Y-axis')
ax.quiver(0, 0, 0, 0, 0, axis_len, color='k', linestyle='dashdot', label='Z-axis')

# Set axes labels and limits
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Eigenvectors with respect to X, Y, Z frame")
ax.legend()
plt.tight_layout()
plt.show()
