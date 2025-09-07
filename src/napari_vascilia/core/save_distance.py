import numpy as np
import os
import pandas as pd
from qtpy.QtWidgets import QDialog, QFormLayout, QLineEdit, QLabel, QPushButton, QMessageBox
from qtpy.QtWidgets import QApplication
from .VASCilia_utils import save_attributes  # Import the utility functions
from .prepare_length_json import Prepare_length_json_Action

class SaveDistanceAction:
    """
    This class handles the action of saving distances for labeled volumes.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the SaveDistanceAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def execute(self):
        """
        Executes the action to save distances for labeled volumes.
        """

        def store_manual_resolution(x_res, y_res, z_res, dialog):
            try:
                # Store the physical resolution
                self.plugin.physical_resolution = (float(x_res), float(y_res), float(z_res))
                print(f"Physical resolution set to: {self.plugin.physical_resolution}")
                dialog.accept()  # Close the dialog if the input is valid
            except ValueError:
                # Handle invalid input
                dialog.accept()
                print("Invalid input for resolution. Please enter valid numerical values.")
                prompt_for_resolution()

        def prompt_for_resolution():
            # Create a dialog
            dialog = QDialog()
            dialog.setWindowTitle('Enter Physical Resolution')
            # Use a form layout
            layout = QFormLayout(dialog)
            # Create text boxes for user input
            x_input = QLineEdit(dialog)
            y_input = QLineEdit(dialog)
            z_input = QLineEdit(dialog)
            #
            # Set default values for the text boxes
            x_input.setText("0.0425")  # Example default value for X resolution in µm
            y_input.setText("0.0425")  # Example default value for Y resolution in µm
            z_input.setText("0.1099")
            # Add rows to the form layout with labels and text boxes
            layout.addRow(QLabel("X Resolution (µm):"), x_input)
            layout.addRow(QLabel("Y Resolution (µm):"), y_input)
            layout.addRow(QLabel("Z Resolution (µm):"), z_input)
            # Button for submitting the resolution
            submit_button = QPushButton('Submit', dialog)
            submit_button.clicked.connect(
                lambda: store_manual_resolution(x_input.text(), y_input.text(), z_input.text(), dialog))
            layout.addWidget(submit_button)
            # Show the dialog
            dialog.exec_()

        if self.plugin.analysis_stage < 5:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Distance Calculation')
            msg_box.setText('Press first Calculate Distance')
            msg_box.exec_()
            return

        self.plugin.loading_label.setText("<font color='red'>Save Processing..., Wait</font>")
        QApplication.processEvents()
        self.plugin.depths = []  # List to store depth of each component

        for idx, points in enumerate(self.plugin.start_points):
            self.plugin.start_points_layer.data[idx][2] = points[2]
        for idx, points in enumerate(self.plugin.end_points):
            self.plugin.end_points_layer.data[idx][2] = points[2]
        if self.plugin.physical_resolution is None:
            prompt_for_resolution()
        dx = self.plugin.physical_resolution[0]
        dy = self.plugin.physical_resolution[1]
        dz = self.plugin.physical_resolution[2]
        # Calculate distances
        self.plugin.physical_distances = []
        for sp, ep, id in zip(self.plugin.start_points_layer.data, self.plugin.end_points_layer.data, self.plugin.IDs):
            point1_phy = np.array([sp[0] * dy, sp[1] * dx, sp[2] * dz])
            point2_phy = np.array([ep[0] * dy, ep[1] * dx, ep[2] * dz])
            distance = np.linalg.norm(point1_phy - point2_phy) / self.plugin.scale_factor
            self.plugin.physical_distances.append((id, distance))
            print(f'Distance of ID {id} = {distance}')
            depth = abs(ep[2] - sp[2]) + 1
            # Get all z slices where this component exists
            # coords = np.where(self.plugin.labeled_volume == id)
            # z_slices = np.unique(coords[2])
            # original_depth = z_slices[-1] - z_slices[0] + 1
            self.plugin.depths.append(depth)

        Distance_path = os.path.join(self.plugin.rootfolder, self.plugin.filename_base, 'Distances')
        if not os.path.exists(Distance_path):
            os.makedirs(Distance_path, exist_ok=True)
        if self.plugin.clustering != 1:
            # When clustering is not done, Depth is the second column
            df = pd.DataFrame(self.plugin.physical_distances, columns=['ID', 'Distance'])
            #df['Depth'] = self.plugin.depths  # Add Depth as the second column
        else:
            # When clustering is done, Depth will be the third column
            df = pd.read_csv(os.path.join(Distance_path, 'Physical_distances.csv'))
            class_col = df.iloc[:, 2]  # Assuming the 'Class' column is the third column
            df = pd.DataFrame(self.plugin.physical_distances, columns=['ID', 'Distance'])
            df['CLass'] = class_col  # Add Class as the second column
            df['Depth'] = self.plugin.depths  # Add Depth as the third column


        df.to_csv(os.path.join(Distance_path, 'Physical_distances.csv'), index=False, sep=',')
        self.plugin.start_points_most_updated = self.plugin.start_points_layer.data
        self.plugin.end_points_most_updated = self.plugin.end_points_layer.data
        self.plugin.analysis_stage = 6

        lines = [np.vstack([start, end]) for start, end in zip(self.plugin.start_points_layer.data, self.plugin.end_points_layer.data)]
        self.plugin.viewer.layers['Peak Points'].data = self.plugin.start_points_layer.data
        self.plugin.viewer.layers['Base Points'].data = self.plugin.end_points_layer.data
        self.plugin.viewer.layers['Lines'].data = lines

        save_attributes(self.plugin, self.plugin.pkl_Path)
        #  please del or uncomment later, this is to collect the data for key point detection
        #Prepare_length_json_Action(self.plugin).execute()
        self.plugin.loading_label.setText("")
        QApplication.processEvents()
