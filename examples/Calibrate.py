def calibrate(label_volume):
    labeled_volume = measure.label(label_volume, connectivity=3)
    props = measure.regionprops(labeled_volume)

    # Assuming label_volume is binary or labeled, and intensity_volume holds the intensity values

    (voxel_size_x,voxel_size_y,voxel_size_z)=(1,1,1)
    results = []
    for prop in props:
        minz, miny, minx, maxz, maxy, maxx = prop.bbox

        # Ignore single-pixel ROIs along the Z-axis
        if maxz - minz <= 1:
            continue

        feret_x = maxx - minx  # Width in the X-axis
        feret_y = maxy - miny  # Height in the Y-axis
        feret_z = maxz - minz  # Depth in the Z-axis

        #pore volume
        pore_volume_voxel_count = prop.area
        voxel_volume = voxel_size_x * voxel_size_y * voxel_size_z
        pore_volume = pore_volume_voxel_count * voxel_volume


        eigenvalues, eigenvectors = np.linalg.eigh(prop.inertia_tensor)
        sorted_eigenvalues = np.sort(eigenvalues)

        # Clamp small negative values to 0
        sorted_eigenvalues = np.clip(sorted_eigenvalues, a_min=0, a_max=None)

        major_axis_vector = eigenvectors[:, np.argmax(eigenvalues)]
        minor_axis_vector = eigenvectors[:, np.argmin(eigenvalues)]
        intermediate_axis_vector = eigenvectors[:, np.argsort(eigenvalues)[1]]

        # Calculate axis lengths based on eigenvalues
        major_axis_length = 2 * np.sqrt(sorted_eigenvalues[2])
        intermediate_axis_length = 2 * np.sqrt(sorted_eigenvalues[1])
        minor_axis_length = 2 * np.sqrt(sorted_eigenvalues[0])

        # Calculate angle of the axes with respect to the Z direction
        major_axis_angle_z = np.arccos(np.abs(major_axis_vector[0])) * (180 / np.pi)  # using major_axis_vector[0] for Z
        intermediate_axis_angle_z = np.arccos(np.abs(intermediate_axis_vector[0])) * (180 / np.pi)  # using intermediate_axis_vector[0] for Z
        minor_axis_angle_z = np.arccos(np.abs(minor_axis_vector[0])) * (180 / np.pi)  # using minor_axis_vector[0] for Z

        # Volume and Surface Area
        volume = feret_x*feret_y*feret_z

        # Moments of inertia (eigenvalues are the moments of inertia)
        moments_of_inertia = sorted_eigenvalues


        results.append({
            'Label': prop.label,
            'Centroid X': prop.centroid[2],  # X-axis
            'Centroid Y': prop.centroid[1],  # Y-axis
            'Centroid Z': prop.centroid[0],  # Z-axis
            'x1': minx,
            'x2': maxx,
            'y1': miny,
            'y2': maxy,
            'z1': minz,
            'z2': maxz,
            'Feret Size X': feret_x,
            'Feret Size Y': feret_y,
            'Feret Size Z': feret_z,
            'Pore volume' : pore_volume,
            'Bounding box volume': volume,
            'Major Axis Vector': major_axis_vector,
            'Minor Axis Vector': minor_axis_vector,
            'Intermediate Axis Vector': intermediate_axis_vector,
            'Major Axis Length': major_axis_length,
            'Minor Axis Length': minor_axis_length,
            'Intermediate Axis Length': intermediate_axis_length,
            'Major Axis Angle Z (deg)': major_axis_angle_z,
            'Intermediate Axis Angle Z (deg)': intermediate_axis_angle_z,
            'Minor Axis Angle Z (deg)': minor_axis_angle_z,
            'Moments of Inertia': moments_of_inertia,
        })
    return results