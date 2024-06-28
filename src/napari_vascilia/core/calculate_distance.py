import numpy as np
from qtpy.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QDesktopWidget, QMessageBox, QApplication
from .VASCilia_utils import save_attributes  # Import the utility functions

class CalculateDistanceAction:
    """
    This class handles the action of calculating distances for labeled volumes.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the CalculateDistanceAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def execute(self):
        """
        Executes the action to calculate distances for labeled volumes.
        """

        def find_centroid(binary_image):
            assert np.isin(binary_image, [0, 1]).all(), "Image should be binary"
            object_pixels = np.argwhere(binary_image == 1)
            centroid_x = int(np.mean(object_pixels[:, 0]))
            centroid_y = int(np.mean(object_pixels[:, 1]))
            return centroid_x, centroid_y

        def center_widget_on_screen(widget):
            frame_geometry = widget.frameGeometry()
            screen_center = QDesktopWidget().availableGeometry().center()
            frame_geometry.moveCenter(screen_center)
            widget.move(frame_geometry.topLeft())

        if self.plugin.analysis_stage == 5 or self.plugin.analysis_stage == 6:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Distance Calculation')
            msg_box.setText('Distances are already calculated')
            msg_box.exec_()
            return

        progress_dialog = QDialog()
        progress_dialog.setWindowTitle('Distance Calculation Progress')
        progress_dialog.setFixedSize(300, 100)
        layout = QVBoxLayout()
        progress_bar = QProgressBar(progress_dialog)
        layout.addWidget(progress_bar)
        progress_dialog.setLayout(layout)
        center_widget_on_screen(progress_dialog)
        progress_dialog.show()
        progress_bar.setMaximum(100)
        progress_per_item = 100 / (self.plugin.num_components + 1 + len(self.plugin.filtered_ids))

        self.plugin.start_points = []
        self.plugin.end_points = []
        self.plugin.IDs = []
        IDtoPointsMAP_list = []
        tempid = 0

        for cc in range(self.plugin.num_components + 1 + len(self.plugin.filtered_ids)):
            if cc == 0 or cc in self.plugin.filtered_ids:
                continue

            self.plugin.IDs.append(cc)
            IDtoPointsMAP_list.append((tempid, cc))
            tempid += 1
            coords = np.where(self.plugin.labeled_volume == cc)
            component = np.zeros_like(self.plugin.labeled_volume)
            component[coords] = 1
            projection = np.sum(component, axis=2)
            projection[projection > 1] = 1
            y_indices, x_indices = np.where(projection == 1)
            highest_point_index = np.argmin(y_indices)
            x_highest = x_indices[highest_point_index]
            y_highest = y_indices[highest_point_index]
            z_highest = np.where(component[y_highest, x_highest, :] == 1)[0]
            self.plugin.start_points.append([y_highest, x_highest, z_highest.item(0)])
            centroid = find_centroid(projection)

            for y in range(centroid[0], projection.shape[0]):
                if projection[y, centroid[1]] == 0:
                    bottom_y = y - 1
                    break

            x_2d, y_2d = centroid[1], bottom_y

            for z in range(component.shape[2]):
                if np.any(component[:, :, z] != 0):
                    break

            self.plugin.end_points.append([y_2d, x_2d, z])

            current_progress = (cc + 1) * progress_per_item
            progress_bar.setValue(int(current_progress))
            QApplication.processEvents()

        self.plugin.IDtoPointsMAP = tuple(IDtoPointsMAP_list)
        self.plugin.analysis_stage = 5
        lines = [np.vstack([start, end]) for start, end in zip(self.plugin.start_points, self.plugin.end_points)]
        if self.plugin.analysis_stage == 6:
            self.plugin.viewer.layers['Peak Points'].data = self.plugin.start_points
            self.plugin.viewer.layers['Base Points'].data = self.plugin.end_points
            self.plugin.viewer.layers['Lines'].data = lines
        else:
            self.plugin.start_points_layer = self.plugin.viewer.add_points(self.plugin.start_points, size=15, face_color='red', name='Peak Points')
            self.plugin.end_points_layer = self.plugin.viewer.add_points(self.plugin.end_points, size=15, face_color='green', name='Base Points')
            self.plugin.lines_layer = self.plugin.viewer.add_shapes(lines, shape_type='path', edge_color='cyan', edge_width=3, name='Lines')
        progress_bar.setValue(100)
        QApplication.processEvents()
        progress_dialog.close()
        self.plugin.start_points_layer.events.data.connect(self.update_lines)
        self.plugin.end_points_layer.events.data.connect(self.update_lines)
        save_attributes(self.plugin, self.plugin.pkl_Path)


#------------------------- update_orientation_lines
    def update_lines(self):
        # Assuming each start point connects to an end point with the same index
        new_lines = []
        if len(self.plugin.start_points_layer.data) == len(self.plugin.end_points_layer.data):
            for start, end in zip(self.plugin.start_points_layer.data, self.plugin.end_points_layer.data):
                new_lines.append([start, end])

        self.plugin.lines_layer.data = new_lines