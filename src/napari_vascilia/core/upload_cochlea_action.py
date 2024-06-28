import matplotlib
matplotlib.use('Qt5Agg')
import os
import numpy as np
#-------------- Qui
from qtpy.QtWidgets import QMessageBox
from qtpy.QtWidgets import QApplication
from qtpy.QtWidgets import  QFileDialog
#
from .VASCilia_utils import display_images, load_attributes  # Import the utility functions


class UploadCochleaAction:
    """
    This class handles the action of uploading previously processed Cochlea datasets.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the UploadCochleaAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def execute(self):
        """
        Executes the action to upload and process previously processed Cochlea datasets.
        It checks if the analysis stage is already set and prompts the user accordingly.
        If a valid file is selected, it proceeds to load and display the file.
        """
        if self.plugin.analysis_stage is not None:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'The stack is already read, if you want to read another stack, press "Reset VASCilia" button')
            msg_box.exec_()
            return

        self.plugin.pkl_Path, _ = QFileDialog.getOpenFileName(caption="Select Analysis_state.pkl",
                                                              filter="Pickled Files (*.pkl)", directory=self.plugin.rootfolder)
        if not self.plugin.pkl_Path:
            return

        self.plugin.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()

        filename_uploaded = self.plugin.pkl_Path.split('/')[-2]
        path_tokens = self.plugin.pkl_Path.split('/')[:-2]
        if os.name == 'nt':
            path_tokens.insert(1, '\\')
            path_of_uploaded_stack = os.path.join(*path_tokens) + '\\'
        else:
            path_tokens.insert(1, '/')
            path_of_uploaded_stack = os.path.join(*path_tokens) + '/'

        load_attributes(self.plugin, self.plugin.pkl_Path)
        self.plugin.rootfolder = path_of_uploaded_stack
        self.plugin.full_stack_raw_images = os.path.join(self.plugin.rootfolder, self.plugin.filename_base,
                                                         'full_stack_raw_images')
        self.plugin.full_stack_raw_images_trimmed = os.path.join(self.plugin.rootfolder, self.plugin.filename_base,
                                                                 'full_stack_raw_images_trimmed')
        self.plugin.full_stack_rotated_images = os.path.join(self.plugin.rootfolder, self.plugin.filename_base,
                                                             'full_stack_rotated_images', 'raw_images')
        self.plugin.npy_dir = os.path.join(self.plugin.rootfolder, self.plugin.filename_base,
                                           'full_stack_rotated_images', 'prediction')
        self.plugin.obj_dir = os.path.join(self.plugin.rootfolder, self.plugin.filename_base,
                                           'full_stack_rotated_images', 'new_assignment_obj')
        self.plugin.pkl_Path = os.path.join(self.plugin.rootfolder, self.plugin.filename_base, 'Analysis_state.pkl')

        if self.plugin.filename_base != filename_uploaded:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'The stack has been renamed and that may lead to problems in the analysis, either return the name back or re-start the analysis ')
            msg_box.exec_()
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return

        #self.plugin.display_images()
        display_images(self.plugin.viewer, self.plugin.display, self.plugin.full_stack_raw_images, self.plugin.full_stack_raw_images_trimmed, self.plugin.full_stack_rotated_images, self.plugin.filename_base, self.plugin.loading_name)

        if self.plugin.analysis_stage == 4:
            self.plugin.viewer.add_labels(self.plugin.labeled_volume, name='Labeled Image')

        if self.plugin.analysis_stage == 5:
            self.plugin.viewer.add_labels(self.plugin.labeled_volume, name='Labeled Image')
            lines = [np.vstack([start, end]) for start, end in zip(self.plugin.start_points, self.plugin.end_points)]
            self.plugin.start_points_layer = self.plugin.viewer.add_points(self.plugin.start_points, size=15,
                                                                           face_color='red', name='Peak Points')
            self.plugin.end_points_layer = self.plugin.viewer.add_points(self.plugin.end_points, size=15,
                                                                         face_color='green', name='Base Points')
            self.plugin.lines_layer = self.plugin.viewer.add_shapes(lines, shape_type='path', edge_color='cyan',
                                                                    edge_width=3, name='Lines')
            self.plugin.start_points_layer.events.data.connect(self.plugin.distance_action.update_lines) #self.plugin.update_lines
            self.plugin.end_points_layer.events.data.connect(self.plugin.distance_action.update_lines)

        if self.plugin.analysis_stage == 6:
            self.plugin.viewer.add_labels(self.plugin.labeled_volume, name='Labeled Image')
            lines = [np.vstack([start, end]) for start, end in
                     zip(self.plugin.start_points_most_updated, self.plugin.end_points_most_updated)]
            self.plugin.start_points_layer = self.plugin.viewer.add_points(self.plugin.start_points_most_updated,
                                                                           size=15, face_color='red',
                                                                           name='Peak Points')
            self.plugin.end_points_layer = self.plugin.viewer.add_points(self.plugin.end_points_most_updated, size=15,
                                                                         face_color='green', name='Base Points')
            self.plugin.lines_layer = self.plugin.viewer.add_shapes(lines, shape_type='path', edge_color='cyan',
                                                                    edge_width=3, name='Lines')
            self.plugin.start_points_layer.events.data.connect(self.plugin.distance_action.update_lines)
            self.plugin.end_points_layer.events.data.connect(self.plugin.distance_action.update_lines)

        if self.plugin.clustering == 1:
            self.plugin.viewer.add_labels(self.plugin.clustered_cells, name='Clustered Cells')
            self.plugin.viewer.add_labels(self.plugin.IHC_OHC, name='IHCs vs OHCs')

        if self.plugin.gt is not None:
            self.plugin.viewer.add_labels(self.plugin.gt, name='Ground Truth')

        if self.plugin.orientation is not None:
            self.plugin.Orientation_Points_layer = self.plugin.viewer.add_points(self.plugin.orientation, size=15,
                                                                                 face_color='magenta',
                                                                                 name='Orientation Points')
            self.plugin.Orientation_Lines_layer = self.plugin.viewer.add_shapes(self.plugin.lines_with_z_swapped,
                                                                                shape_type='line', edge_color='yellow',
                                                                                name='Orientation Lines')
            self.plugin.Orientation_text_layer = self.plugin.viewer.add_points(self.plugin.text_positions, size=0.05,
                                                                               face_color='transparent',
                                                                               name='Angle Annotations')
            self.plugin.Orientation_text_layer.text = self.plugin.text_annotations
            self.plugin.Orientation_text_layer.text_color = 'lime'
            self.plugin.Orientation_text_layer.text_size = 12
            self.plugin.viewer.dims.ndisplay = 3
            self.plugin.Orientation_Points_layer.events.data.connect(self.plugin.orientation_action.update_orientation_lines)

        if self.plugin.analysis_stage >= 4:
            self.plugin.ID_layer = self.plugin.viewer.add_points(self.plugin.ID_positions_annotation, size=0.05,
                                                                 face_color='transparent', name='ID Annotations')
            self.plugin.ID_layer.text = self.plugin.id_list_annotation
            self.plugin.ID_layer.text.color = 'lime'
            self.plugin.ID_layer.text.size = 12

        print(f'Scale Factor is: {self.plugin.scale_factor}')
        self.plugin.loading_label.setText("")
        QApplication.processEvents()