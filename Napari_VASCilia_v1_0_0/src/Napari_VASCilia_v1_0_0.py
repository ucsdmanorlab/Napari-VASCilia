"""
Author: Yasmin Kassim
Postdoc Employee, Manor Lab, University of California San Diego
"""
import matplotlib
matplotlib.use('Qt5Agg')
import os
import json
import napari
from PyQt5.QtWidgets import QApplication
from ui_setup import create_ui


class NapariPlugin:
    """
        A class to represent the Napari plugin and manage its configuration and UI.
    """

    def __init__(self):
        """Initialize the NapariPlugin class with default settings and configuration."""
        self.load_config()
        self.setup_variables()
        self.viewer = napari.Viewer()
        self.initialize_ui()

    def load_config(self):
        """Load the configuration from the 'config.json' file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def setup_variables(self):
        """Set up the initial variables based on the loaded configuration."""
        self.rootfolder = self.config.get('rootfolder', os.path.dirname(os.path.abspath(__file__)))
        self.wsl_executable = self.config.get('wsl_executable', '')
        self.model = self.config.get('model', '')
        self.model_output_path = self.config.get('model_output_path', '')
        self.model_region_prediction = self.config.get('model_region_prediction', '')
        self.model_celltype_identification = self.config.get('model_celltype_identification', '')
        # Retrieve the variables
        self.flag_to_resize = self.config.get('flag_to_resize', False)
        self.flag_to_pad = self.config.get('flag_to_pad', False)
        self.resize_dimension = self.config.get('resize_dimension', 1200)
        self.pad_dimension = self.config.get('pad_dimension', 1500)

        # Initialize other attributes
        self.train_iter = 75000
        self.training_path = None
        self.analysis_stage = None
        self.pkl_Path = None
        self.BUTTON_WIDTH = 60
        self.BUTTON_HEIGHT = 22
        self.filename_base = None
        self.full_stack_raw_images = None
        self.full_stack_length = None
        self.full_stack_raw_images_trimmed = None
        self.full_stack_rotated_images = None
        self.physical_resolution = None
        self.format = None
        self.npy_dir = None
        self.obj_dir = None
        self.start_trim = None
        self.end_trim = None
        self.display = None
        self.labeled_volume = None
        self.filtered_ids = None
        self.num_components = None
        self.delete_allowed = True
        self.id_list_annotation = None
        self.ID_positions_annotation = None
        self.start_points = None
        self.end_points = None
        self.start_points_most_updated = None
        self.end_points_most_updated = None
        self.start_points_layer = None
        self.end_points_layer = None
        self.lines_layer = None
        self.physical_distances = None
        self.IDs = None
        self.IDtoPointsMAP = None
        self.clustering = None
        self.clustered_cells = None
        self.IHC = None
        self.IHC_OHC = None
        self.OHC = None
        self.OHC1 = None
        self.OHC2 = None
        self.OHC3 = None
        self.gt = None
        self.lines_with_z_swapped = None
        self.text_positions = None
        self.text_annotations = None
        self.id_list = None
        self.orientation = None
        self.scale_factor = 1

    def initialize_ui(self):
        """Initialize the user interface for the Napari plugin."""
        container = create_ui(self) # Create the UI container using the imported function
        self.viewer.window.add_dock_widget(container, area="right", name='Napari-VASCilia') # Add the container to the viewer
        app = QApplication([])  # Create a QApplication instance
        self.viewer.window.qt_viewer.showMaximized()  # Maximize the viewer window
        app.exec_()  # Execute the application

if __name__ == "__main__":
    plugin = NapariPlugin()  # Instantiate the NapariPlugin class
    napari.run()  # Run the Napari application
