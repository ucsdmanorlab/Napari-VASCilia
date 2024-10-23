import matplotlib
matplotlib.use('Qt5Agg')
from skimage.measure import label, regionprops
from magicgui import magicgui
import pandas as pd
import os
import numpy as np
#-------------- Qui
from qtpy.QtWidgets import QApplication
from .VASCilia_utils import save_attributes  # Import the utility functions

class ComputeOrientationAction:

    """
    This class handles the action of orientation computation for labeled volumes.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):

        """
        Initializes the Compute Orientation action with a reference to the main plugin.

        Args:
         plugin: The main plugin instance that this action will interact with.
        """

        self.plugin = plugin

    def create_orientation_widget(self):
        @magicgui(call_button='Compute Orientation', method={'choices': ['Height_only', 'Height_Distance']})
        def orientation_widget(method: str):
            self.calculate_orientation(method)
        return orientation_widget

    def update_orientation_lines(self, event=None):
        lines = []
        angles = []
        self.plugin.text_annotations = []  # For storing angle text
        for i in range(0, len(self.plugin.orientation), 2):
            right_point = self.plugin.orientation[i]
            right_point[2] = 0
            self.plugin.orientation[i] = right_point  # Update the original list
            left_point = self.plugin.orientation[i + 1]
            left_point[2] = 0
            self.plugin.orientation[i + 1] = left_point  # Update the original list

            # Compute the line and angle
            line = [right_point[:2][::-1], left_point[:2][::-1]]  # Using 2D points for the line
            lines.append(line)

            deltaY = right_point[1] - left_point[1]
            deltaX = right_point[0] - left_point[0]
            angle_rad = np.arctan2(deltaY, deltaX)
            angle_deg = np.degrees(angle_rad)
            angles.append(round(angle_deg, 2))
            # Store centroid for text annotation
            self.plugin.text_annotations.append(f"{angle_deg:.2f}°")

        lines_array = np.array(lines)
        lines_with_z = np.zeros((lines_array.shape[0], lines_array.shape[1], 3))
        lines_with_z[:, :, :2] = lines_array
        # Swap the x and y coordinates
        self.plugin.lines_with_z_swapped = lines_with_z[:, :, [1, 0, 2]]
        self.plugin.Orientation_Lines_layer.data = self.plugin.lines_with_z_swapped
        self.plugin.Orientation_text_layer.text = self.plugin.text_annotations
        orientation_dir = self.plugin.rootfolder + '/' + self.plugin.filename_base + '/orientation/'
        if not os.path.exists(orientation_dir):
            os.makedirs(orientation_dir)
        annotations_df = pd.DataFrame({
            'ID': self.plugin.id_list,
            'Angle': angles
        })
        annotations_df.to_csv(orientation_dir + 'angle_annotations.csv', index=False, encoding='utf-8-sig')
        save_attributes(self.plugin, self.plugin.pkl_Path)

    def calculate_orientation(self, method: str):
        self.plugin.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()
        orientation_dir = self.plugin.rootfolder + '/' + self.plugin.filename_base + '/orientation/'
        if not os.path.exists(orientation_dir):
            os.makedirs(orientation_dir)
        props = regionprops(self.plugin.labeled_volume)

        # Initialize an empty list to store the orientation points
        self.plugin.orientation = []
        centroid_2d_list = []
        self.plugin.id_list = []
        for prop in props:
            self.plugin.id_list.append(prop.label)
            # Convert 3D centroid to 2D by ignoring the z-coordinate
            centroid_2d = prop.centroid[:2]
            centroid_2d_list.append(centroid_2d)
            voxel_coords = np.transpose(np.nonzero(self.plugin.labeled_volume == prop.label))
            # Project the voxel coordinates to 2D by ignoring the z-coordinate
            voxel_coords_2d = voxel_coords[:, :2]
            if method == 'Height_only':
                # Find the lowest points to the left and right of the centroid in 2D space
                right_points = voxel_coords_2d[voxel_coords_2d[:, 1] > centroid_2d[1]]
                left_points = voxel_coords_2d[voxel_coords_2d[:, 1] < centroid_2d[1]]
                if right_points.size > 0:
                    lowest_right_point_2d = right_points[np.argmax(right_points[:, 0])]
                    # Add the point with z-coordinate set to 0
                    self.plugin.orientation.append(np.append(lowest_right_point_2d, 0))

                if left_points.size > 0:
                    lowest_left_point_2d = left_points[np.argmax(left_points[:, 0])]
                    # Add the point with z-coordinate set to 0
                    self.plugin.orientation.append(np.append(lowest_left_point_2d, 0))

            elif method == 'Height_Distance':
                coords = np.where(self.plugin.labeled_volume == prop.label)
                component = np.zeros_like(self.plugin.labeled_volume)
                component[coords] = 1
                projection = np.sum(component, axis=2)
                projection[projection > 1] = 1
                y_indices, x_indices = np.where(projection == 1)
                highest_point_index = np.argmin(y_indices)
                x_highest = x_indices[highest_point_index]
                y_highest = y_indices[highest_point_index]
                peakpoint = (y_highest, x_highest)
                # Find the lowest points to the left and right of the centroid in 2D space
                right_points = voxel_coords_2d[voxel_coords_2d[:, 1] > peakpoint[1]]
                left_points = voxel_coords_2d[voxel_coords_2d[:, 1] < peakpoint[1]]

                # Function to calculate distance from the centroid
                def distance_from_centroid(point, centroid):
                    return np.sqrt((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2)

                # Calculate distances for all points on the right
                right_distances = [distance_from_centroid(point, peakpoint) for point in right_points]
                # Combine right points with their distances and sort them based on distance, then by height
                right_points_with_distance = sorted(zip(right_points, right_distances), key=lambda x: (-x[1], x[0][0]))
                # Calculate distances for all points on the left
                left_distances = [distance_from_centroid(point, peakpoint) for point in left_points]
                # Combine left points with their distances and sort them based on distance, then by height
                left_points_with_distance = sorted(zip(left_points, left_distances), key=lambda x: (-x[1], x[0][0]))
                # Select the point with the maximum distance and then the lowest (considering y-coordinate as height)
                if right_points_with_distance:
                    # Extracting the point information; since it's sorted by distance and then height, the first one meets the criteria
                    farthest_lowest_right_point = right_points_with_distance[0][0]
                    self.plugin.orientation.append(
                        np.append(farthest_lowest_right_point, 0))  # Appending with z-coordinate set to 0
                if left_points_with_distance:
                    # Extracting the point information; since it's sorted by distance and then height, the first one meets the criteria
                    farthest_lowest_left_point = left_points_with_distance[0][0]
                    self.plugin.orientation.append(
                        np.append(farthest_lowest_left_point, 0))  # Appending with z-coordinate set to 0

        # Convert the list of points to an array
        self.plugin.orientation = np.array(self.plugin.orientation)
        if 'Orientation Points' in self.plugin.viewer.layers:
            self.plugin.viewer.layers['Orientation Points'].data = self.plugin.orientation
        else:
            self.plugin.Orientation_Points_layer = self.plugin.viewer.add_points(self.plugin.orientation, size=15,
                                                                                 face_color='magenta',
                                                                                 name='Orientation Points')

        # Find the angles
        # Initialize arrays for lines, angles, and centroids for text annotations
        lines = []
        angles = []
        self.plugin.text_positions = []  # For storing centroids as points for text annotations
        self.plugin.text_annotations = []  # For storing angle text
        for i in range(0, len(self.plugin.orientation), 2):
            right_point = self.plugin.orientation[i]
            left_point = self.plugin.orientation[i + 1]
            centroid = centroid_2d_list[i // 2]
            centroid = list(centroid)
            if right_point[0] >= left_point[0]:
                centroid[0] = right_point[0]
            else:
                centroid[0] = left_point[0]
            centroid = tuple(centroid)
            centroid = centroid[::-1]
            # Compute the line and angle
            line = [right_point[:2][::-1], left_point[:2][::-1]]  # Using 2D points for the line
            lines.append(line)

            deltaY = right_point[1] - left_point[1]
            deltaX = right_point[0] - left_point[0]
            angle_rad = np.arctan2(deltaY, deltaX)
            angle_deg = np.degrees(angle_rad)
            angles.append(round(angle_deg, 2))
            # Store centroid for text annotation
            self.plugin.text_positions.append(centroid[:2])  # Ensure 2D
            self.plugin.text_annotations.append(f"{angle_deg:.2f}°")

        lines_array = np.array(lines)
        lines_with_z = np.zeros((lines_array.shape[0], lines_array.shape[1], 3))
        lines_with_z[:, :, :2] = lines_array
        # Swap the x and y coordinates
        self.plugin.lines_with_z_swapped = lines_with_z[:, :, [1, 0, 2]]

        if 'Orientation Lines' in self.plugin.viewer.layers:
            self.plugin.viewer.layers['Orientation Lines'].data = self.plugin.lines_with_z_swapped
        else:
            self.plugin.Orientation_Lines_layer = self.plugin.viewer.add_shapes(self.plugin.lines_with_z_swapped,
                                                                                shape_type='line', edge_color='yellow',
                                                                                name='Orientation Lines')

        self.plugin.text_positions = [np.append(pos[::-1], 0) for pos in self.plugin.text_positions]
        self.plugin.text_positions = [pos + [25, 0, 0] for pos in self.plugin.text_positions]
        self.plugin.text_positions = np.array(self.plugin.text_positions).astype(int)
        if 'Angle Annotations' in self.plugin.viewer.layers:
            self.plugin.viewer.layers['Angle Annotations'].data = self.plugin.text_positions
        else:
            self.plugin.Orientation_text_layer = self.plugin.viewer.add_points(self.plugin.text_positions, size=0.05,
                                                                               face_color='transparent',
                                                                               name='Angle Annotations')

        self.plugin.Orientation_text_layer.text = self.plugin.text_annotations  # Assign text annotations to the points
        self.plugin.Orientation_text_layer.text_color = 'lime'
        self.plugin.Orientation_text_layer.text_size = 12

        for layer in self.plugin.viewer.layers:
            if layer.name in ['Angle Annotations', 'Orientation Points', 'Orientation Lines', 'Original Volume',
                              'Labeled Image', 'ID Annotations']:
                layer.visible = True
            else:
                layer.visible = False
        self.plugin.viewer.layers.move(self.plugin.viewer.layers.index('Angle Annotations'),
                                       len(self.plugin.viewer.layers) - 1)
        self.plugin.viewer.dims.ndisplay = 3
        # Create a DataFrame from the ID and angle lists
        annotations_df = pd.DataFrame({
            'ID': self.plugin.id_list,
            'Angle': angles
        })
        annotations_df.to_csv(orientation_dir + 'angle_annotations.csv', index=False, encoding='utf-8-sig')
        # Attach event listeners to the start and end points layers
        self.plugin.Orientation_Points_layer.events.data.connect(self.update_orientation_lines)
        self.plugin.delete_allowed = False
        save_attributes(self.plugin, self.plugin.pkl_Path)
        self.plugin.loading_label.setText("")
        QApplication.processEvents()