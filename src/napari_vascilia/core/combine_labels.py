import matplotlib
matplotlib.use('Qt5Agg')
from skimage.measure import label, regionprops
import numpy as np
#-------------- Qui
from qtpy.QtWidgets import QMessageBox
from qtpy.QtWidgets import  QLineEdit, QHBoxLayout
from qtpy.QtWidgets import QPushButton, QWidget, QComboBox
from .VASCilia_utils import save_attributes  # Import the utility functions

class CombineAction:
    """
    This class handles the action of clustering cells using various methods.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the CellClusteringAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin


    def perform_combine_action(self, label1: str, label2: str):

        if not self.plugin.delete_allowed:  # Check if deletion is allowed
            QMessageBox.warning(None, 'Combine', 'Combine labels is not allowed in this step of analysis')
            return

        mylabel1 = int(label1)
        mylabel2 = int(label2)
        if mylabel1 not in np.unique(self.plugin.labeled_volume):
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Label Info')
            msg_box.setText('Label does not exist, please write a proper label')
            msg_box.exec_()
            return
        if mylabel2 not in np.unique(self.plugin.labeled_volume):
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Label Info')
            msg_box.setText('Label does not exist, please write a proper label')
            msg_box.exec_()
            return

        component = max(mylabel1, mylabel2)
        label_to_remove = component
        self.plugin.labeled_volume[self.plugin.labeled_volume == label_to_remove] = min(mylabel1, mylabel2)
        self.plugin.viewer.layers['Labeled Image'].data = self.plugin.labeled_volume
        self.plugin.filtered_ids.append(label_to_remove)
        self.plugin.num_components -= 1

        props = regionprops(self.plugin.labeled_volume)
        self.plugin.id_list_annotation = []
        self.plugin.ID_positions_annotation = []
        for prop in props:
            self.plugin.id_list_annotation.append(prop.label)
            # self.plugin.ID_positions_annotation.append(prop.centroid[:2])
            coords = np.where(self.plugin.labeled_volume == prop.label)
            component = np.zeros_like(self.plugin.labeled_volume)
            component[coords] = 1
            projection = np.sum(component, axis=2)
            projection[projection > 1] = 1
            y_indices, x_indices = np.where(projection == 1)
            highest_point_index = np.argmin(y_indices)
            x_highest = x_indices[highest_point_index]
            y_highest = y_indices[highest_point_index]
            self.plugin.ID_positions_annotation.append((y_highest, x_highest))
        self.plugin.ID_positions_annotation = [np.append(pos, 0) for pos in self.plugin.ID_positions_annotation]
        self.plugin.ID_positions_annotation = [pos - [10, 0, 0] for pos in self.plugin.ID_positions_annotation]
        self.plugin.ID_positions_annotation = np.array(self.plugin.ID_positions_annotation).astype(int)
        if 'ID Annotations' in self.plugin.viewer.layers:
            self.plugin.viewer.layers['ID Annotations'].data = self.plugin.ID_positions_annotation
        else:
            self.plugin.ID_layer = self.plugin.viewer.add_points(self.plugin.ID_positions_annotation, size=0.05,
                                                                 face_color='transparent',
                                                                 name='ID Annotations')

        self.plugin.ID_layer.text = self.plugin.id_list_annotation  # Assign text annotations to the points
        self.plugin.ID_layer.text.color = 'lime'
        self.plugin.ID_layer.text.size = 12
        save_attributes(self.plugin, self.plugin.pkl_Path)

    def create_combineaction_widget(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)  # Horizontal layout

        # Create the dropdown menu for selecting the clustering method
        label_input1 = QLineEdit()
        label_input1.setPlaceholderText("Enter label number")

        # Create the textbox for entering the label number
        label_input2 = QLineEdit()
        label_input2.setPlaceholderText("Enter label number")

        # Create the re-assign button
        combine_button = QPushButton("Combine labels")
        combine_button.clicked.connect(
            lambda: self.perform_combine_action(label_input1.text(), label_input2.text()))

        # Add widgets to the layout
        layout.addWidget(label_input1)
        layout.addWidget(label_input2)
        layout.addWidget(combine_button)

        return widget