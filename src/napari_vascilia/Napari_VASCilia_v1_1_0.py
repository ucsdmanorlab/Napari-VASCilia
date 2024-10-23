import matplotlib
matplotlib.use('Qt5Agg')
import os
import json
import napari
from qtpy.QtWidgets import QWidget
from .ui_setup import create_ui
from pathlib import Path
from qtpy.QtWidgets import QApplication


class NapariPlugin:
    """
    A class to represent the Napari plugin and manage its configuration and UI.
    """

    def __init__(self, viewer):
        """Initialize the NapariPlugin class with default settings and configuration."""
        print('initialization started')
        self.viewer = napari.current_viewer()
        self.load_config()
        self.setup_variables()
        print('setup_variables finish')

    def load_config(self):
        """Load the configuration from the 'config.json' file."""
        config_path = Path.home() / '.napari-vascilia' / 'config.json'
        if not config_path.exists():
            self.create_default_config(config_path)
            print('json file created')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def create_default_config(self, config_path):
        """Create a default configuration file if it does not exist."""
        default_config = {
            "rootfolder": str('C:/Users/..../processed_data/'),
            "wsl_executable": str('C:/Users/..../models/Train_predict_stereocilia_exe/Train_Predict_stereocilia_exe_v2'),
            "model": str('C:/Users/..../models/seg_model/stereocilia_v7/'),
            "model_output_path": str('C:/Users/..../models/new_seg_model/stereocilia_v8/'),
            "model_region_prediction": str('C:/Users/..../models/region_prediction/resnet50_best_checkpoint_resnet50_balancedclass.pth'),
            "model_celltype_identification": str('C:/Users/..../models/cell_type_identification_model/'),
            "model_ZFT_prediction": str('C:/Users/..../models/ZFT_trim_model/'),
            "model_rotation_prediction": str('C:/Users/..../models/rotation_correction_model/'),
            "flag_to_resize": False,
            "flag_to_pad": False,
            "resize_dimension": 1200,
            "pad_dimension": 1500,
            "button_width": 60,
            "button_height": 22,
        }
        print(default_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)

    def setup_variables(self):
        """Set up the initial variables based on the loaded configuration."""
        self.rootfolder = self.config.get('rootfolder', os.path.dirname(os.path.abspath(__file__)))
        self.wsl_executable = self.config.get('wsl_executable', '')
        self.model = self.config.get('model', '')
        # Convert wsl_executable and model paths for Windows OS
        if os.name == 'nt':
            if self.wsl_executable.lower().startswith('c:/') or self.wsl_executable.lower().startswith(
                    'c:\\'):
                self.wsl_executable = '/mnt/c/' + self.wsl_executable[3:].replace('\\', '/')
            if self.model.lower().startswith('c:/') or self.model.lower().startswith('c:\\'):
                self.model = '/mnt/c/' + self.model[3:].replace('\\', '/')

        self.model_output_path = self.config.get('model_output_path', '')
        self.model_region_prediction = self.config.get('model_region_prediction', '')
        self.model_celltype_identification = self.config.get('model_celltype_identification', '')
        self.model_ZFT_prediction = self.config.get('ZFT_trim_model', '')
        self.model_rotation_prediction = self.config.get('rotation_correction_model', '')
        self.flag_to_resize = self.config.get('flag_to_resize', False)
        self.flag_to_pad = self.config.get('flag_to_pad', False)
        self.resize_dimension = self.config.get('resize_dimension', 1200)
        self.pad_dimension = self.config.get('pad_dimension', 1500)
        self.BUTTON_WIDTH = self.config.get('button_width', 60)
        self.BUTTON_HEIGHT = self.config.get('button_height', 22)
        self.train_iter = 75000
        self.training_path = None
        self.analysis_stage = None
        self.pkl_Path = None
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

    def initialize_ui(self) -> QWidget:
        """Initialize the user interface for the Napari plugin."""
        print("initialize_ui called")  # Debugging print statement
        container = create_ui(self)  # Create the UI container using the imported function
        print("initialize_ui ended")  # Debugging print statement
        return container  # Return the container to be used as a QWidget

def initialize_vascilia_ui():
    plugin = NapariPlugin(napari.Viewer)
    return plugin.initialize_ui()


# class NapariPlugin:
#     """
#         A class to represent the Napari plugin and manage its configuration and UI.
#     """
#
#     def __init__(self):
#         """Initialize the NapariPlugin class with default settings and configuration."""
#         self.load_config()
#         self.setup_variables()
#         self.viewer = napari.Viewer()
#         self.initialize_ui()
#
#     def load_config(self):
#         """Load the configuration from the 'config.json' file."""
#         config_path = os.path.join(os.path.dirname(__file__), 'config.json')
#         with open(config_path, 'r') as f:
#             self.config = json.load(f)
#
#     def setup_variables(self):
#         """Set up the initial variables based on the loaded configuration."""
#         self.rootfolder = self.config.get('rootfolder', os.path.dirname(os.path.abspath(__file__)))
#         self.wsl_executable = self.config.get('wsl_executable', '')
#         self.model = self.config.get('model', '')
#         self.model_output_path = self.config.get('model_output_path', '')
#         self.model_region_prediction = self.config.get('model_region_prediction', '')
#         self.model_celltype_identification = self.config.get('model_celltype_identification', '')
#         self.model_ZFT_prediction = self.config.get('ZFT_trim_model', '')
#         self.model_rotation_prediction = self.config.get('rotation_correction_model', '')
#         # Retrieve the variables
#         self.flag_to_resize = self.config.get('flag_to_resize', False)
#         self.flag_to_pad = self.config.get('flag_to_pad', False)
#         self.resize_dimension = self.config.get('resize_dimension', 1200)
#         self.pad_dimension = self.config.get('pad_dimension', 1500)
#
#         # Initialize other attributes
#         self.train_iter = 75000
#         self.training_path = None
#         self.analysis_stage = None
#         self.pkl_Path = None
#         self.BUTTON_WIDTH = 60
#         self.BUTTON_HEIGHT = 22
#         self.filename_base = None
#         self.full_stack_raw_images = None
#         self.full_stack_length = None
#         self.full_stack_raw_images_trimmed = None
#         self.full_stack_rotated_images = None
#         self.physical_resolution = None
#         self.format = None
#         self.npy_dir = None
#         self.obj_dir = None
#         self.start_trim = None
#         self.end_trim = None
#         self.display = None
#         self.labeled_volume = None
#         self.filtered_ids = None
#         self.num_components = None
#         self.delete_allowed = True
#         self.id_list_annotation = None
#         self.ID_positions_annotation = None
#         self.start_points = None
#         self.end_points = None
#         self.start_points_most_updated = None
#         self.end_points_most_updated = None
#         self.start_points_layer = None
#         self.end_points_layer = None
#         self.lines_layer = None
#         self.physical_distances = None
#         self.IDs = None
#         self.IDtoPointsMAP = None
#         self.clustering = None
#         self.clustered_cells = None
#         self.IHC = None
#         self.IHC_OHC = None
#         self.OHC = None
#         self.OHC1 = None
#         self.OHC2 = None
#         self.OHC3 = None
#         self.gt = None
#         self.lines_with_z_swapped = None
#         self.text_positions = None
#         self.text_annotations = None
#         self.id_list = None
#         self.orientation = None
#         self.scale_factor = 1
#
#     def initialize_ui(self):
#         """Initialize the user interface for the Napari plugin."""
#         container = create_ui(self) # Create the UI container using the imported function
#         self.viewer.window.add_dock_widget(container, area="right", name='Napari-VASCilia') # Add the container to the viewer
#         app = QApplication([])  # Create a QApplication instance
#         self.viewer.window.qt_viewer.showMaximized()  # Maximize the viewer window
#         app.exec_()  # Execute the application
#
# if __name__ == "__main__":
#     plugin = NapariPlugin()  # Instantiate the NapariPlugin class
#     napari.run()  # Run the Napari application
