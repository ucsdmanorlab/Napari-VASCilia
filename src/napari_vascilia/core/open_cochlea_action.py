import cv2
from czitools import read_tools, napari_tools
#import czifile
from tifffile import imread, imwrite, TiffWriter, TiffFile
import os
import numpy as np
from readlif.reader import LifFile
from pathlib import Path
import json
#-------------- Qui
from qtpy.QtWidgets import QMessageBox
from qtpy.QtWidgets import QApplication, QDialog,  QFormLayout
from qtpy.QtWidgets import  QFileDialog, QLabel, QLineEdit
from qtpy.QtWidgets import QPushButton
from .VASCilia_utils import display_images, save_attributes, load_attributes  # Import the utility functions


class OpenCochleaAction:
    """
    This class handles the action of opening Cochlea datasets (CZI or LIF files) and preprocessing them.
        It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim

    """

    def __init__(self, plugin, batch, batch_file_path):
        """
        Initializes the OpenCochleaAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin
        self.batch = batch
        self.batch_filepath = batch_file_path

    def load_config_setup_paths(self):
        self.plugin.loading_label.setText("<font color='red'>Setup Paths..., Wait</font>")

        """Load the configuration from the 'config.json' file and verify paths."""
        config_path = Path.home() / '.napari-vascilia' / 'config.json'

        # Check if config.json exists
        if not config_path.exists():
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Configuration Error')
            msg_box.setText(
                f'The config.json file does not exist. Please create or update the config.json file according to the documentation. '
                f'This file needs to be located at {config_path} and should include all necessary paths and trained models. '
                f'Trained models can be downloaded from the links provided in the GitHub repository.'
            )
            msg_box.exec_()
            return False

        # Load config.json
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Check if all paths in the config are valid
        invalid_paths = []
        paths_to_check = {
            'rootfolder': self.config.get('rootfolder'),
            'wsl_executable': self.config.get('wsl_executable'),
            'model': self.config.get('model'),
            'model_output_path': self.config.get('model_output_path'),
            'model_region_prediction': self.config.get('model_region_prediction'),
            'model_celltype_identification': self.config.get('model_celltype_identification')
        }

        # this code check if the path starts with /mnt/c/ then it will change it to valid path and check if it exist or not
        for key, path in paths_to_check.items():
            if os.name == 'nt' and path.startswith('/mnt/c/'):
                path = Path('C:/' + path[len('/mnt/c/'):])
            else:
                path = Path(path)
            if not path.exists():
                invalid_paths.append(f"{key}: {path}")

        if invalid_paths:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Configuration Error')
            msg_box.setText(
                f'\n The following paths in config.json are invalid or do not exist, please open config.json in your config path {config_path} and update them, here are the inappropriate paths :\n' +
                '\n\n'.join(invalid_paths)
            )
            msg_box.exec_()
            return False

        # Setup the plugin variables
        self.plugin.rootfolder = paths_to_check['rootfolder'] or os.path.dirname(os.path.abspath(__file__))
        self.plugin.wsl_executable = paths_to_check['wsl_executable'] or ''
        self.plugin.model = paths_to_check['model'] or ''
        # Convert wsl_executable and model paths for Windows OS
        if os.name == 'nt':
            if self.plugin.wsl_executable.lower().startswith('c:/') or self.plugin.wsl_executable.lower().startswith(
                    'c:\\'):
                self.plugin.wsl_executable = '/mnt/c/' + self.plugin.wsl_executable[3:].replace('\\', '/')
            if self.plugin.model.lower().startswith('c:/') or self.plugin.model.lower().startswith('c:\\'):
                self.plugin.model = '/mnt/c/' + self.plugin.model[3:].replace('\\', '/')

        self.plugin.model_output_path = paths_to_check['model_output_path'] or ''
        self.plugin.model_region_prediction = paths_to_check['model_region_prediction'] or ''
        self.plugin.model_celltype_identification = paths_to_check['model_celltype_identification'] or ''
        self.flag_to_upscale = self.config.get('flag_to_upscale', False)
        self.flag_to_downscale = self.config.get('flag_to_downscale', False)
        self.flag_to_pad = self.config.get('flag_to_pad', False)

        if self.flag_to_upscale and self.flag_to_downscale:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Configuration Error')
            msg_box.setText(
                "Both 'flag_to_upscale' and 'flag_to_downscale' are set to True in config.json.\n\n"
                "Please choose only one resizing mode to avoid ambiguity."
            )
            msg_box.exec_()
            return False

        # wherever you load other flags from config
        self.plugin.force_manual_resolution = int(self.config.get('force_manual_resolution', 0))
        self.plugin.resize_dimension = self.config.get('resize_dimension', 1200)
        self.plugin.pad_dimension = self.config.get('pad_dimension', 1500)
        self.plugin.BUTTON_WIDTH = self.config.get('button_width', 60)
        self.plugin.BUTTON_HEIGHT = self.config.get('button_height', 22)

        return True

    def execute(self):
        """
            Executes the action to open and preprocess Cochlea datasets.
            It checks if the analysis stage is already set and prompts the user accordingly.
            If a valid file is selected, it proceeds to read and preprocess the file.
        """
        if not self.load_config_setup_paths():
            return

        if self.plugin.analysis_stage is not None:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'The stack is already read, if you want to read another stack, press "Reset VASCilia" button')
            msg_box.exec_()
            return

        if self.batch == 0:
            self.file_path, _ = QFileDialog.getOpenFileName()
        else:
            self.file_path = self.batch_filepath

        if self.file_path:
            if os.path.splitext(self.file_path)[1].lower() == '.czi':
                self.plugin.format = '.czi'
                self.read_czi_and_preprocess()
            elif os.path.splitext(self.file_path)[1].lower() == '.lif':
                self.plugin.format = '.lif'
                self.read_lif_and_preprocess()
            elif os.path.splitext(self.file_path)[1].lower() == '.tif':
                self.plugin.format = '.tif'
                self.read_tif_and_preprocess()
            else:
                # Show a message box if the file is not a .czi file
                QMessageBox.warning(None, 'File Selection', 'Please select a valid CZI, LIF or Tif file. Other formats are not supported')


    def resize_or_pad_image(self, image_clahe, height, width):

        new_height = height
        new_width = width

        # Handle upscaling
        if self.plugin.flag_to_upscale and (
                height < self.plugin.resize_dimension or width < self.plugin.resize_dimension):
            if height < width:
                self.plugin.scale_factor = self.plugin.resize_dimension / height
            else:
                self.plugin.scale_factor = self.plugin.resize_dimension / width
            new_height = int(height * self.plugin.scale_factor)
            new_width = int(width * self.plugin.scale_factor)
            image_clahe = cv2.resize(image_clahe, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Handle downscaling
        elif self.plugin.flag_to_downscale and (
                height > self.plugin.resize_dimension or width > self.plugin.resize_dimension):
            if height > width:
                self.plugin.scale_factor = self.plugin.resize_dimension / height
            else:
                self.plugin.scale_factor = self.plugin.resize_dimension / width
            new_height = int(height * self.plugin.scale_factor)
            new_width = int(width * self.plugin.scale_factor)
            image_clahe = cv2.resize(image_clahe, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Padding (if enabled)
        if self.plugin.flag_to_pad:
            pad_height = max(0, self.plugin.pad_dimension - new_height)
            pad_width = max(0, self.plugin.pad_dimension - new_width)
            top = pad_height // 2
            bottom = pad_height - top
            left = pad_width // 2
            right = pad_width - left
            if len(image_clahe.shape) == 3:
                image_clahe = np.pad(image_clahe, ((top, bottom), (left, right), (0, 0)), mode='constant',
                                     constant_values=0)
            else:
                image_clahe = np.pad(image_clahe, ((top, bottom), (left, right)), mode='constant', constant_values=0)

        return image_clahe

    def read_lif_and_preprocess(self):
        """
            Reads and preprocesses LIF files.
            It prompts the user for physical resolution if not available, processes the images,
            and saves the processed stack.
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
        # Read the file
        # Pre-process
        # Write the stack
        self.plugin.loading_label.setText("<font color='red'>Upload Processing..., Wait</font>")
        QApplication.processEvents()
        base_name = os.path.splitext(self.file_path)[0]
        self.plugin.filename_base = base_name.split('/')[-1].replace(' ', '')[:45]
        self.plugin.filename_base = self.plugin.filename_base.replace('(', '')
        self.plugin.filename_base = self.plugin.filename_base.replace(')', '')
        new_folder_path = os.path.join(self.plugin.rootfolder, self.plugin.filename_base)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path, exist_ok=True)
        else:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('File Selection')
            msg_box.setText(
                'This lif stack is already processed, please press the [Upload Processed Stack] buttom to upload your analysis')
            msg_box.exec_()
            # Exit the function
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return

        self.plugin.full_stack_raw_images = os.path.join(new_folder_path, 'full_stack_raw_images')
        self.plugin.full_stack_rotated_images = os.path.join(new_folder_path, 'full_stack_rotated_images', 'raw_images')

        if not os.path.exists(self.plugin.full_stack_raw_images):
            os.makedirs(self.plugin.full_stack_raw_images, exist_ok=True)

        lif_file = LifFile(self.file_path)
        images = list(lif_file.get_iter_image())
        if len(images) > 1:
            # Get the second image
            second_image = images[1]
        else:
            second_image =images[0]

        Green_stack = []
        Red_stack = []
        # This loop is to find the green stack and also the red stack to store them as seperated stacks later in the code ,
        # and it also take each i,mage , normaliza it and combine them to RGB and apply CLAHE
        for i in range(second_image.dims.z):
            Green_ch = second_image.get_frame(z=i)
            Green_stack.append(np.array(Green_ch))
            Red_ch = np.zeros((np.shape(Green_ch)[0], np.shape(Green_ch)[1]), dtype=np.uint8)
            #Red_ch = Green_ch.copy()
            Red_stack.append(Red_ch)
            Blue_ch = np.zeros((np.shape(Red_ch)[0], np.shape(Red_ch)[1]), dtype=np.uint8)
            RGB_image = np.stack([Red_ch, Green_ch, Blue_ch], axis=2)
            height, width = RGB_image.shape[:2]
            # image_8bit = RGB_image.compute()
            image_8bit = RGB_image
            #
            if np.max(image_8bit) == 0:
                print("Max value is zero, exiting loop.")
                break
            # print(i)
            image_8bit = (image_8bit - np.min(image_8bit)) / ((np.max(image_8bit) - np.min(image_8bit))) * 255

            image_8bit = np.array(image_8bit, dtype=np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # # Apply CLAHE to each channel separately
            Red_ch_clahe = clahe.apply(image_8bit[:, :, 0])
            Green_ch_clahe = clahe.apply(image_8bit[:, :, 1])
            Blue_ch_clahe = clahe.apply(image_8bit[:, :, 2])
            # Merge the channels back
            image_clahe = cv2.merge([Red_ch_clahe, Green_ch_clahe, Blue_ch_clahe])
            padded_i = str(i).zfill(4)
            file_name_towrite = self.plugin.full_stack_raw_images + '/' + self.plugin.filename_base + f'_{padded_i}.tif'

            # Check if any dimension is less than 1500
            if self.plugin.flag_to_upscale or self.plugin.flag_to_downscale or self.plugin.flag_to_pad:
                image_clahe = self.resize_or_pad_image(image_clahe, height, width)
                if i == 0:
                    new_height, new_width = image_clahe.shape[:2]
                    if (height != new_height) or (width != new_width):
                        print(f"[Resizing] Image {i}: Original = {height}x{width}, "
                              f"New = {new_height}x{new_width}, Scale Factor = {self.plugin.scale_factor:.4f}")

            imwrite(file_name_towrite, image_clahe)
        self.plugin.full_stack_length = i + 1
        self.plugin.display = None
        # ------- Find the resolution
        if getattr(self.plugin, "force_manual_resolution", 0) == 1:
            prompt_for_resolution()
        else:
            try:
                scaling_x_meters = second_image.info['scale'][0]
                scaling_y_meters = second_image.info['scale'][1]
                scaling_z_meters = second_image.info['scale'][2]
                scaling_x_micrometers = float(scaling_x_meters) / 1000.0
                scaling_y_micrometers = float(scaling_y_meters) / 1000.0
                scaling_z_micrometers = abs(float(scaling_z_meters)) / 1000.0
                self.plugin.physical_resolution = (
                    scaling_x_micrometers, scaling_y_micrometers, scaling_z_micrometers
                )
            except Exception as e:
                print(f"Error retrieving resolution: {e}")
                prompt_for_resolution()

        # self.DesAnalysisPath = new_folder_path + '/'
        print(f"Physical resolution extracted: {self.plugin.physical_resolution}")
        self.plugin.analysis_stage = 1
        self.plugin.pkl_Path = new_folder_path + '/' + 'Analysis_state.pkl'
        save_attributes(self.plugin, self.plugin.pkl_Path)
        #self.plugin.display_images()
        display_images(self.plugin.viewer, self.plugin.display, self.plugin.full_stack_raw_images,
                       self.plugin.full_stack_raw_images_trimmed, self.plugin.full_stack_rotated_images,
                       self.plugin.filename_base, self.plugin.loading_name)
        self.plugin.loading_label.setText("")
        QApplication.processEvents()

    def read_czi_and_preprocess(self):
        """
            Reads and preprocesses CZI files.
            It prompts the user for physical resolution if not available, processes the images,
            and saves the processed stack.
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
        # Read the file
        # Pre-process
        # Write the stack
        self.plugin.loading_label.setText("<font color='red'>Upload Processing..., Wait</font>")
        QApplication.processEvents()
        base_name = os.path.splitext(self.file_path)[0]
        self.plugin.filename_base = base_name.split('/')[-1].replace(' ', '')[:45]
        self.plugin.filename_base = self.plugin.filename_base.replace('(', '')
        self.plugin.filename_base = self.plugin.filename_base.replace(')', '')
        new_folder_path = os.path.join(self.plugin.rootfolder, self.plugin.filename_base)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path, exist_ok=True)
        else:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('File Selection')
            msg_box.setText(
                'This stack is already processed, please press the [Upload Processed Stack] buttom to upload your analysis')
            msg_box.exec_()
            # Exit the function
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return

        self.plugin.full_stack_raw_images = os.path.join(new_folder_path, 'full_stack_raw_images')
        self.plugin.full_stack_rotated_images = os.path.join(new_folder_path, 'full_stack_rotated_images', 'raw_images')
        # self.plugin.full_stack_raw_images = new_folder_path + '/' + 'full_stack_raw_images'
        # self.plugin.full_stack_rotated_images = new_folder_path + '/' +  'full_stack_rotated_images' + '/' +  'raw_images'

        if not os.path.exists(self.plugin.full_stack_raw_images):
            os.makedirs(self.plugin.full_stack_raw_images, exist_ok=True)

        # with czifile.CziFile(self.file_path) as czi:
        #     array6d = czi.asarray()
        array6d, mdata, dim_string6d = read_tools.read_6darray(self.file_path,
                                                               output_order="STCZYX",
                                                               use_dask=True,
                                                               chunk_zyx=False,
                                                               # T=0,
                                                               # Z=0
                                                               # S=0
                                                               # C=0
                                                               )

        array6d = array6d.compute()
        Green_stack = []
        Red_stack = []
        # This loop is to find the green stack and also the red stack to store them as seperated stacks later in the code ,
        # and it also take each i,mage , normaliza it and combine them to RGB and apply CLAHE
        for i in range(np.shape(array6d)[3]):
            Green_ch = array6d[0, 0, self.plugin.green_channel, i, :, :]
            Green_stack.append(Green_ch)
            Red_ch = array6d[0, 0, self.plugin.red_channel, i, :, :]
            Red_stack.append(Red_ch)
            if self.plugin.blue_channel == -1:
                Blue_ch = np.zeros((np.shape(Red_ch)[0], np.shape(Red_ch)[1]), dtype=np.uint8)
            else:
                Blue_ch = array6d[0, 0, self.plugin.blue_channel, i, :, :]
            RGB_image = np.stack([Red_ch, Green_ch, Blue_ch], axis=2)
            height, width = RGB_image.shape[:2]
            image_8bit = RGB_image
            if np.max(image_8bit) == 0:
                print("Max value is zero, exiting loop.")
                break
            image_8bit = (image_8bit - np.min(image_8bit)) / ((np.max(image_8bit) - np.min(image_8bit))) * 255
            image_8bit = np.array(image_8bit, dtype=np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # # Apply CLAHE to each channel separately
            Red_ch_clahe = clahe.apply(image_8bit[:, :, 0])
            Green_ch_clahe = clahe.apply(image_8bit[:, :, 1])
            Blue_ch_clahe = clahe.apply(image_8bit[:, :, 2])
            # Merge the channels back
            image_clahe = cv2.merge([Red_ch_clahe, Green_ch_clahe, Blue_ch_clahe])
            padded_i = str(i).zfill(4)
            file_name_towrite = self.plugin.full_stack_raw_images + '/' + self.plugin.filename_base + f'_{padded_i}.tif'
            # Check if any dimension is less than 1500
            if self.plugin.flag_to_upscale or self.plugin.flag_to_downscale or self.plugin.flag_to_pad:
                image_clahe = self.resize_or_pad_image(image_clahe, height, width)
                if i == 0:
                    new_height, new_width = image_clahe.shape[:2]
                    if (height != new_height) or (width != new_width):
                        print(f"[Resizing] Image {i}: Original = {height}x{width}, "
                              f"New = {new_height}x{new_width}, Scale Factor = {self.plugin.scale_factor:.4f}")
            imwrite(file_name_towrite, image_clahe)
        self.plugin.full_stack_length = i + 1
        self.plugin.display = None
        #---------------------------------- Find Resolution ----------------------
        if getattr(self.plugin, "force_manual_resolution", 0) == 1:
            prompt_for_resolution()
        else:
            try:
                scaling_x_meters = mdata.channelinfo.czisource[
                    'ImageDocument'].Metadata.Experiment.ExperimentBlocks.AcquisitionBlock.AcquisitionModeSetup.ScalingX
                scaling_y_meters = mdata.channelinfo.czisource[
                    'ImageDocument'].Metadata.Experiment.ExperimentBlocks.AcquisitionBlock.AcquisitionModeSetup.ScalingY
                scaling_z_meters = mdata.channelinfo.czisource[
                    'ImageDocument'].Metadata.Experiment.ExperimentBlocks.AcquisitionBlock.AcquisitionModeSetup.ScalingZ
                scaling_x_micrometers = float(scaling_x_meters) * 1e6
                scaling_y_micrometers = float(scaling_y_meters) * 1e6
                scaling_z_micrometers = float(scaling_z_meters) * 1e6
                self.plugin.physical_resolution = (
                    scaling_x_micrometers, scaling_y_micrometers, scaling_z_micrometers
                )
            except Exception as e:
                print(f"Error retrieving resolution: {e}")
                prompt_for_resolution()

        #--------------------------------------------------------------------------------------
        #self.DesAnalysisPath = new_folder_path + '/'
        print(f"Physical resolution extracted: {self.plugin.physical_resolution}")
        self.plugin.analysis_stage = 1
        self.plugin.pkl_Path = new_folder_path + '/'  + 'Analysis_state.pkl'
        save_attributes(self.plugin, self.plugin.pkl_Path)
        #self.plugin.display_images()
        display_images(self.plugin.viewer, self.plugin.display, self.plugin.full_stack_raw_images,
                       self.plugin.full_stack_raw_images_trimmed, self.plugin.full_stack_rotated_images,
                       self.plugin.filename_base, self.plugin.loading_name)
        self.plugin.loading_label.setText("")
        QApplication.processEvents()

    #------------- Tif Reader --------------------------------------------------------------------------

    def read_tif_and_preprocess(self):
        """
            Reads and preprocesses tif files.
            It prompts the user for physical resolution if not available, processes the images,
            and saves the processed stack.
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
        # Read the file
        # Pre-process
        # Write the stack
        self.plugin.loading_label.setText("<font color='red'>Upload Processing..., Wait</font>")
        QApplication.processEvents()
        base_name = os.path.splitext(self.file_path)[0]
        self.plugin.filename_base = base_name.split('/')[-1].replace(' ', '')[:45]
        self.plugin.filename_base = self.plugin.filename_base.replace('(', '')
        self.plugin.filename_base = self.plugin.filename_base.replace(')', '')
        new_folder_path = os.path.join(self.plugin.rootfolder, self.plugin.filename_base)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path, exist_ok=True)
        else:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('File Selection')
            msg_box.setText(
                'This stack is already processed, please press the [Upload Processed Stack] buttom to upload your analysis')
            msg_box.exec_()
            # Exit the function
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return

        self.plugin.full_stack_raw_images = os.path.join(new_folder_path, 'full_stack_raw_images')
        self.plugin.full_stack_rotated_images = os.path.join(new_folder_path, 'full_stack_rotated_images', 'raw_images')

        if not os.path.exists(self.plugin.full_stack_raw_images):
            os.makedirs(self.plugin.full_stack_raw_images, exist_ok=True)

        im = imread(self.file_path)
        Green_stack = []
        Red_stack = []

        for i in range(im.shape[0]):
            if im.ndim == 4:  # 4D: (frames, channels, height, width)
                Green_ch = im[i, self.plugin.green_channel, :, :]
                Red_ch = im[i, self.plugin.red_channel, :, :]
            else:  # Assume 3D: (frames, height, width) with single channel
                Green_ch = im[i, :, :]
                Red_ch = np.zeros((Green_ch.shape[0], Green_ch.shape[1]), dtype=np.uint8)

            if self.plugin.blue_channel == -1:
                Blue_ch = np.zeros((Green_ch.shape[0], Green_ch.shape[1]), dtype=np.uint8)
            else:
                Blue_ch = im[i, self.plugin.blue_channel, :, :]

            Green_stack.append(np.array(Green_ch))
            Red_stack.append(np.array(Red_ch))
            RGB_image = np.stack([Red_ch, Green_ch, Blue_ch], axis=2)
            height, width = RGB_image.shape[:2]
            image_8bit = (RGB_image - np.min(RGB_image)) / ((np.max(RGB_image) - np.min(RGB_image))) * 255
            image_8bit = np.array(image_8bit, dtype=np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            Red_ch_clahe = clahe.apply(image_8bit[:, :, 0])
            Green_ch_clahe = clahe.apply(image_8bit[:, :, 1])
            Blue_ch_clahe = clahe.apply(image_8bit[:, :, 2])
            image_clahe = cv2.merge([Red_ch_clahe, Green_ch_clahe, Blue_ch_clahe])
            padded_i = str(i).zfill(4)
            file_name_towrite = self.plugin.full_stack_raw_images + '/' + self.plugin.filename_base + f'_{padded_i}.tif'
            # Check if any dimension is less than 1500
            if self.plugin.flag_to_upscale or self.plugin.flag_to_downscale or self.plugin.flag_to_pad:
                image_clahe = self.resize_or_pad_image(image_clahe, height, width)
                if i == 0:
                    new_height, new_width = image_clahe.shape[:2]
                    if (height != new_height) or (width != new_width):
                        print(f"[Resizing] Image {i}: Original = {height}x{width}, "
                              f"New = {new_height}x{new_width}, Scale Factor = {self.plugin.scale_factor:.4f}")
            imwrite(file_name_towrite, image_clahe)

        self.plugin.full_stack_length = i + 1
        self.plugin.display = None
        # ---------------------------------- Find Resolution ----------------------
        if getattr(self.plugin, "force_manual_resolution", 0) == 1:
            prompt_for_resolution()
        else:
            try:
                with TiffFile(self.file_path) as tif:
                    page = tif.pages[0]
                    tags = page.tags

                    x_res_tag = tags.get('XResolution')
                    y_res_tag = tags.get('YResolution')
                    res_unit_tag = tags.get('ResolutionUnit')

                    if x_res_tag and y_res_tag:
                        x_res_value = x_res_tag.value[0] / x_res_tag.value[1]
                        y_res_value = y_res_tag.value[0] / y_res_tag.value[1]
                        res_unit = res_unit_tag.value if res_unit_tag else 1
                        if res_unit == 2:
                            factor = 25_400  # µm per inch
                        elif res_unit == 3:
                            factor = 10_000  # µm per cm
                        else:
                            factor = 1  # assume already µm/pixel
                        x_res_micron = factor / x_res_value
                        y_res_micron = factor / y_res_value
                    else:
                        raise ValueError("XResolution or YResolution tags not found")

                    description = page.description
                    z_res_micron = None
                    if description:
                        for entry in description.split('\n'):
                            if entry.startswith('spacing='):
                                z_res_micron = float(entry.split('=')[1])
                                break

                    if z_res_micron is None:
                        raise ValueError("Z resolution (spacing) not found in ImageJ metadata.")

                    self.plugin.physical_resolution = (x_res_micron, y_res_micron, z_res_micron)
                    print(f"Physical resolution extracted: {self.plugin.physical_resolution}")
            except Exception as e:
                print(f"Error retrieving resolution: {e}")
                prompt_for_resolution()

        #--------------------------------------------------------------------------------------
        #self.DesAnalysisPath = new_folder_path + '/'
        self.plugin.analysis_stage = 1
        self.plugin.pkl_Path = new_folder_path + '/'  + 'Analysis_state.pkl'
        save_attributes(self.plugin, self.plugin.pkl_Path)
        #self.plugin.display_images()
        display_images(self.plugin.viewer, self.plugin.display, self.plugin.full_stack_raw_images,
                       self.plugin.full_stack_raw_images_trimmed, self.plugin.full_stack_rotated_images,
                       self.plugin.filename_base, self.plugin.loading_name)
        self.plugin.loading_label.setText("")
        QApplication.processEvents()