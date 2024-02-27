# This code is fully implemented by Dr. Yasmin Kassim at Manor Lab/ Cell and Development Biology Department

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Qt5Agg')
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import os
import napari
import imageio
from skimage.io import imsave, imread
from scipy.ndimage import label, find_objects, binary_fill_holes, sum as ndi_sum
import cv2
import shutil
from magicgui import magicgui
import pandas as pd
import subprocess
import re
from qtpy.QtCore import QTimer
from czitools import read_tools, napari_tools
import czifile
from scipy.ndimage import rotate
from tifffile import imwrite
import pickle
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import csv
from PIL import Image
import time
# Region classification
import torch
from torchvision import transforms
from torchvision.models import resnet50
import os
from PIL import Image
from torch import nn
from collections import Counter
import numpy as np
#-------------- Qui
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QApplication, QProgressBar, QPushButton, QVBoxLayout, QDesktopWidget, QDialog, QSlider, QFormLayout
from qtpy.QtWidgets import  QFileDialog, QLabel, QLineEdit, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QSize
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

class NapariPlugin:
    def __init__(self):
        # Initialize attributes (previously global variables)
        self.rootfolder = os.path.dirname(os.path.abspath(__file__))
        #self.wsl_executable = '/mnt/c/Users/Yasmin/pycharm_projects/Detecron2/Executable_v1/Predict_stereocilia_exe_v1/Predict_stereocilia_exe_v1'
        self.wsl_executable = '/mnt/c/Users/Yasmin/pycharm_projects/Detecron2/Executable_v1/Train_predict_stereocilia_exe/Train_Predict_stereocilia_exe_v1'
        self.model = '/mnt/c/Users/Yasmin/pycharm_projects/Detecron2/Executable_v1/Train_predict_stereocilia_exe/stereocilia_v3/'
        self.model_output_path = 'C:/Users/Yasmin/pycharm_projects/Detecron2/Executable_v1/Train_predict_stereocilia_exe/stereocilia_v4/'
        self.model_region_prediction = 'C:/Users/Yasmin/pycharm_projects/Torch_classification/Three_class_classification_yk/Exp1_2D_threeclasses/resnet50_best_checkpoint_resnet50_balancedclass.pth'
        self.train_iter = 50000
        self.training_path = None

        self.analysis_stage = None
        self.pkl_Path = None
        # Napari buttoms width and height
        self.BUTTON_WIDTH = 60
        self.BUTTON_HEIGHT = 28

        self.file_path = None   # This is file path from QFileDialog.getOpenFileName()
        self.filename_base = None  # file name
        self.full_stack_raw_images = None
        self.full_stack_length = None
        self.full_stack_raw_images_trimmed = None
        self.full_stack_rotated_images = None
        self.physical_resolution = None
        #self.DesAnalysisPath = None

        self.npy_dir = None     # This is for npy files after we run the prediction
        self.obj_dir = None     # This is for new_assignment_obj after we run the prediction, this will be used by the tracking algorithm

        self.start_trim = None  # Which slice to start trimming
        self.end_trim = None    # Which slice to end trimming
        self.display = None     # to control which frames to display, None  to display full stack, 1 for full_stack_raw_images_trimmed, And 2 for full_stack_rotated_images
        self.labeled_volume = None   # This is the labled volume that will be added through self.viewer.add_labels, and I generated it after prediction and tracking
        self.filtered_ids = None     # The objects that have been filtered when depth < 3
        self.num_components = None   # Number of compoenents in the labeled volume

        self.start_points = None
        self.end_points = None
        self.start_points_most_updated = None
        self.end_points_most_updated = None
        self.start_points_layer = None  # for distance calculationself.start_points_layer.data, self.end_points_layer.data
        self.end_points_layer = None    # for distance calculation
        self.lines_layer = None        # for distance calculation
        self.physical_distances = None
        self.IDs = None
        self.IDtoPointsMAP = None     # to map which cc ID corresponds to which starting point, ending point and line
        self.clustering = None  # to inform the sotware that the clustering is done, mainly for delete buttom

        self.clustered_cells = None
        self.IHC = None
        self.IHC_OHC = None
        self.OHC = None
        self.OHC1 = None
        self.OHC2 = None
        self.OHC3 = None
        #
        self.gt = None

        self.viewer = napari.Viewer()
        self.initialize_ui()

    def save_attributes(self, filename):
        # Specify the attributes to save
        attributes_to_save = {
            'rootfolder': self.rootfolder,
            'wsl_executable': self.wsl_executable,
            'model': self.model,
            'analysis_stage': self.analysis_stage,
            'pkl_Path': self.pkl_Path,
            'BUTTON_WIDTH': self.BUTTON_WIDTH,
            'BUTTON_HEIGHT': self.BUTTON_HEIGHT,
            'file_path': self.file_path,
            'filename_base': self.filename_base,
            'full_stack_raw_images': self.full_stack_raw_images,
            'full_stack_length': self.full_stack_length,
            'full_stack_raw_images_trimmed': self.full_stack_raw_images_trimmed,
            'full_stack_rotated_images': self.full_stack_rotated_images,
            'physical_resolution': self.physical_resolution,
            #'DesAnalysisPath': self.DesAnalysisPath,
            'npy_dir': self.npy_dir,
            'obj_dir': self.obj_dir,
            'start_trim': self.start_trim,
            'end_trim': self.end_trim,
            'display': self.display,
            'labeled_volume': self.labeled_volume,
            'filtered_ids': self.filtered_ids,
            'num_components': self.num_components,
            'physical_distances': self.physical_distances,
            'start_points_most_updated': self.start_points_most_updated,
            'end_points_most_updated': self.end_points_most_updated,
            'start_points': self.start_points,
            'end_points': self.end_points,
            'IDs': self.IDs,
            'IDtoPointsMAP': self.IDtoPointsMAP,
            'Clustering_state': self.clustering,
            'clustered_cells': self.clustered_cells,
            'IHC': self.IHC,
            'IHC_OHC': self.IHC_OHC,
            'OHC': self.OHC,
            'OHC1': self.OHC1,
            'OHC2': self.OHC2,
            'OHC3': self.OHC3,
            'gt': self.gt

        }
        # Save the specified attributes
        with open(filename, 'wb') as file:
            pickle.dump(attributes_to_save, file)

    def load_attributes(self, filename):
        # Load the saved attributes
        with open(filename, 'rb') as file:
            loaded_attributes = pickle.load(file)

        # Update the class attributes with the loaded values
        self.rootfolder = loaded_attributes.get('rootfolder', None)
        self.wsl_executable = loaded_attributes.get('wsl_executable', None)
        self.model = loaded_attributes.get('model', None)
        self.analysis_stage = loaded_attributes.get('analysis_stage', None)
        self.pkl_Path = loaded_attributes.get('pkl_Path', None)
        self.BUTTON_WIDTH = loaded_attributes.get('BUTTON_WIDTH', None)
        self.BUTTON_HEIGHT = loaded_attributes.get('BUTTON_HEIGHT', None)
        self.file_path = loaded_attributes.get('file_path', None)
        self.filename_base = loaded_attributes.get('filename_base', None)
        self.full_stack_raw_images = loaded_attributes.get('full_stack_raw_images', None)
        self.full_stack_length = loaded_attributes.get('full_stack_length', None)
        self.full_stack_raw_images_trimmed = loaded_attributes.get('full_stack_raw_images_trimmed', None)
        self.full_stack_rotated_images = loaded_attributes.get('full_stack_rotated_images', None)
        self.physical_resolution = loaded_attributes.get('physical_resolution', None)
        self.npy_dir = loaded_attributes.get('npy_dir', None)
        self.obj_dir = loaded_attributes.get('obj_dir', None)
        self.start_trim = loaded_attributes.get('start_trim', None)
        self.end_trim = loaded_attributes.get('end_trim', None)
        self.display = loaded_attributes.get('display', None)
        self.labeled_volume = loaded_attributes.get('labeled_volume', None)
        self.filtered_ids = loaded_attributes.get('filtered_ids', None)
        self.num_components = loaded_attributes.get('num_components', None)
        self.physical_distances = loaded_attributes.get('physical_distances', None)
        self.start_points_most_updated = loaded_attributes.get('start_points_most_updated', None) #
        self.end_points_most_updated = loaded_attributes.get('end_points_most_updated', None)
        self.start_points = loaded_attributes.get('start_points', None)  #
        self.end_points = loaded_attributes.get('end_points', None)
        self.IDs = loaded_attributes.get('IDs', None)
        self.IDtoPointsMAP = loaded_attributes.get('IDtoPointsMAP', None)
        self.clustering = loaded_attributes.get('Clustering_state', None)
        self.clustered_cells = loaded_attributes.get('clustered_cells', None)
        self.IHC = loaded_attributes.get('IHC', None)
        self.IHC_OHC = loaded_attributes.get('IHC_OHC', None)
        self.OHC = loaded_attributes.get('OHC', None)
        self.OHC1 = loaded_attributes.get('OHC1', None)
        self.OHC2 = loaded_attributes.get('OHC2', None)
        self.OHC3 = loaded_attributes.get('OHC3', None)
        self.gt = loaded_attributes.get('gt', None)


    def initialize_ui(self):

        container = QWidget()
        layout = QVBoxLayout(container)
        # Load the logo image
        layout.setContentsMargins(0, 0, 0, 0)
        logo_path = './VASCilia_logo1.png'  # Update this to the path of your logo file
        logo_pixmap = QPixmap(logo_path)
        # Resize the logo
        logo_size = QSize(125, 75)  # Set this to your desired dimensions
        scaled_logo_pixmap = logo_pixmap.scaled(logo_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Create a QLabel to display the logo
        logo_label = QLabel()
        logo_label.setPixmap(scaled_logo_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)  # Center align the logo if desired
        # Add the logo label to the layout
        logo_label.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(logo_label)

        buttons_info = [
            ("Open CZI Cochlea Files and Preprocess", self.open_czi_cochlea_files),
            ("Upload Processed CZI Stack", self.upload_czi_cochlea_files),
            ("Trim Full Stack", self.trim_cochlea_files),
            ("Rotate", self.Rotate_stack),
            ("Segment with 3DCiliaSeg", self.segment_with_3DCiliaSeg),
            ("Reconstruct and Visualize", self.visualize_segmentation_Detecron2_with_tracking)

        ]

        for text, func in buttons_info:
            button = QPushButton(text)
            button.clicked.connect(func)
            button.setMinimumSize(self.BUTTON_WIDTH, self.BUTTON_HEIGHT)
            layout.addWidget(button)

        # Add the filter_component_widget to the layout 'Delete buttom'
        self.filter_component_widget = self.create_filter_component_widget()
        layout.addWidget(self.filter_component_widget.native)

        # List your buttons and connect them to functions
        buttons_info = [
            ("Calculate Measurements", self.calculate_measurements),
            ("Calculate Distance", self.calculate_distance),
            ("Save Distance", self.save_distance)
        ]

        # Create buttons and add them to the layout
        for text, func in buttons_info:
            button = QPushButton(text)
            button.clicked.connect(func)
            button.setMinimumSize(self.BUTTON_WIDTH, self.BUTTON_HEIGHT)
            layout.addWidget(button)

        #
        self.clustering_widget_choice = self.create_clustering_widget()
        layout.addWidget(self.clustering_widget_choice.native)

        buttons_info = [
            ("Find IHCs and OHCs", self.find_IHC_OHC),
            #("Compute Orientation", self.calculate_orientation),
            ("Compute Protein Intensity", self.Compute_Protein_Intensity),
            ("Predict region", self.predict_region)
        ]


        # Create buttons and add them to the layout
        for text, func in buttons_info:
            button = QPushButton(text)
            button.clicked.connect(func)
            button.setMinimumSize(self.BUTTON_WIDTH, self.BUTTON_HEIGHT)
            layout.addWidget(button)

        # Add a separator label with text
        separator_label = QLabel("------- Training Section ------")
        separator_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(separator_label)

        buttons_info = [
            ("Create/Save Ground Truth", self.creategt),
            ("Generate Ground Truth Masks", self.savemasks),
            ("Display Stored Ground Truth", self.display_stored_gt),
            ("Copy Segmentation Masks to Ground Truth", self.copymasks),
            ("Move Ground Truth to Training Folder", self.move_gt),
            ("Check Training Data", self.check_training_data),
            ("Train New Model for 3DCiliaSeg", self.train_cilia),
            ("Reset VASCilia", self.reset_button),
            ("Exit VASCilia", self.exit_button)
        ]

        # Create buttons and add them to the layout
        for text, func in buttons_info:
            button = QPushButton(text)
            button.clicked.connect(func)
            button.setMinimumSize(self.BUTTON_WIDTH, self.BUTTON_HEIGHT)
            layout.addWidget(button)

        self.loading_label = QLabel("")
        layout.addWidget(self.loading_label)
        #
        self.loading_name = QLabel("")
        layout.addWidget(self.loading_name)
        # Add the container as a dock widget to the viewer
        self.viewer.window.add_dock_widget(container, area="right", name='Napari-VASCilia')
        app = QApplication([])
        self.viewer.window.qt_viewer.showMaximized()
        app.exec_()

###---------------------- Open CZI buttom code --------------------------------------------------
    def open_czi_cochlea_files(self):
        if self.analysis_stage is not None:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'The stack is already read, if you want to read another stack, press "Reset VASCilia" buttom')
            msg_box.exec_()
            return
        self.file_path, _ = QFileDialog.getOpenFileName()
        if self.file_path:
            if os.path.splitext(self.file_path)[1].lower() == '.czi':
                self.read_czi_and_preprocess()

            else:
                # Show a message box if the file is not a .czi file
                QMessageBox.warning(None, 'File Selection', 'Please select a valid CZI file.')


    def read_czi_and_preprocess(self):
        # Read the file
        # Pre-process
        # Write the stack
        self.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()
        dst_folder = './'
        base_name = os.path.splitext(self.file_path)[0]
        self.filename_base = base_name.split('/')[-1].replace(' ', '')[:45]
        new_folder_path = os.path.join(dst_folder, self.filename_base)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path, exist_ok=True)
        else:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('File Selection')
            msg_box.setText(
                'This CZI stack is already processed, please press the [Upload Processed CZI Stack] buttom to upload your analysis')
            msg_box.exec_()
            # Exit the function
            return

        self.full_stack_raw_images = os.path.join(new_folder_path, 'full_stack_raw_images')
        self.full_stack_rotated_images = os.path.join(new_folder_path, 'full_stack_rotated_images', 'raw_images')

        if not os.path.exists(self.full_stack_raw_images):
            os.makedirs(self.full_stack_raw_images, exist_ok=True)

        with czifile.CziFile(self.file_path) as czi:
            array6d = czi.asarray()
        array6d, mdata, dim_string6d = read_tools.read_6darray(self.file_path,
                                                               output_order="STCZYX",
                                                               use_dask=True,
                                                               chunk_zyx=False,
                                                               # T=0,
                                                               # Z=0
                                                               # S=0
                                                               # C=0
                                                               )

        # # Store the physical resolution
        # scaling_x_meters  = mdata.channelinfo.czisource['ImageDocument'].Metadata.Experiment.ExperimentBlocks.AcquisitionBlock.AcquisitionModeSetup.ScalingX
        # scaling_y_meters = mdata.channelinfo.czisource['ImageDocument'].Metadata.Experiment.ExperimentBlocks.AcquisitionBlock.AcquisitionModeSetup.ScalingY
        # scaling_z_meters = mdata.channelinfo.czisource['ImageDocument'].Metadata.Experiment.ExperimentBlocks.AcquisitionBlock.AcquisitionModeSetup.ScalingZ
        # scaling_x_micrometers = float(scaling_x_meters) * 1e6
        # scaling_y_micrometers = float(scaling_y_meters) * 1e6
        # scaling_z_micrometers = float(scaling_z_meters) * 1e6
        # self.physical_resolution = (scaling_x_micrometers, scaling_y_micrometers, scaling_z_micrometers)
        # print(self.physical_resolution)
        array6d = array6d.compute()
        Green_stack = []
        Red_stack = []
        # This loop is to find the green stack and also the red stack to store them as seperated stacks later in the code ,
        # and it also take each i,mage , normaliza it and combine them to RGB and apply CLAHE
        for i in range(np.shape(array6d)[3]):
            Green_ch = array6d[0, 0, 0, i, :, :]
            Green_stack.append(Green_ch)
            Red_ch = array6d[0, 0, 1, i, :, :]
            Red_stack.append(Red_ch)
            Blue_ch = np.zeros((np.shape(Red_ch)[0], np.shape(Red_ch)[1]), dtype=np.uint8)
            RGB_image = np.stack([Red_ch, Green_ch, Blue_ch], axis=2)
            #image_8bit = RGB_image.compute()
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
            file_name_towrite = self.full_stack_raw_images + '/' + self.filename_base + f'_{padded_i}.tif'
            imwrite(file_name_towrite, image_clahe)
        self.full_stack_length = i + 1
        self.display = None
        #self.DesAnalysisPath = new_folder_path + '/'
        self.analysis_stage = 1
        self.pkl_Path = new_folder_path + '/'  + 'Analysis_state.pkl'
        self.save_attributes(self.pkl_Path)
        self.display_images()
        self.loading_label.setText("")
        QApplication.processEvents()

    def upload_czi_cochlea_files(self):
        if self.analysis_stage is not None:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'The stack is already read, if you want to read another stack, press "Reset VASCilia" buttom')
            msg_box.exec_()
            return
        self.pkl_Path, _ = QFileDialog.getOpenFileName(caption="Select Analysis_state.pkl", filter="Pickled Files (*.pkl)")
        if not self.pkl_Path:
            return
        self.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()
        self.load_attributes(self.pkl_Path)
        self.display_images()
        if self.analysis_stage == 4:
            self.viewer.add_labels(self.labeled_volume, name='Labeled Image')
        if self.analysis_stage == 5:
            self.viewer.add_labels(self.labeled_volume, name='Labeled Image')
            lines = [np.vstack([start, end]) for start, end in zip(self.start_points, self.end_points)]
            self.start_points_layer = self.viewer.add_points(self.start_points, size=15, face_color='red',
                                                             name='Peak Points')
            self.end_points_layer = self.viewer.add_points(self.end_points, size=15, face_color='green',
                                                           name='Base Points')
            self.lines_layer = self.viewer.add_shapes(lines, shape_type='path', edge_color='cyan', edge_width=3,
                                                 name='Lines')
            self.start_points_layer.events.data.connect(self.update_lines)
            self.end_points_layer.events.data.connect(self.update_lines)
        if self.analysis_stage == 6:
            self.viewer.add_labels(self.labeled_volume, name='Labeled Image')
            lines = [np.vstack([start, end]) for start, end in zip(self.start_points_most_updated, self.end_points_most_updated)]
            self.start_points_layer = self.viewer.add_points(self.start_points_most_updated, size=15, face_color='red',
                                                             name='Peak Points')
            self.end_points_layer = self.viewer.add_points(self.end_points_most_updated, size=15, face_color='green',
                                                           name='Base Points')
            self.lines_layer = self.viewer.add_shapes(lines, shape_type='path', edge_color='cyan', edge_width=3,
                                                 name='Lines')
            self.start_points_layer.events.data.connect(self.update_lines)
            self.end_points_layer.events.data.connect(self.update_lines)
        if self.clustering == 1:
            self.viewer.add_labels(self.clustered_cells, name='Clustered Cells')
            self.viewer.add_labels(self.IHC_OHC, name='IHCs vs OHCs')
        if self.gt is not None:
            self.viewer.add_labels(self.gt, name='Ground Truth')

        self.loading_label.setText("")
        QApplication.processEvents()

    def display_images(self):
        #full_stack_rotated_images
        images = []
        red_images = []
        if self.display == None:
            display_path = self.full_stack_raw_images
        elif self.display == 1:
            display_path = self.full_stack_raw_images_trimmed
        elif self.display == 2:
            display_path = self.full_stack_rotated_images
        rawim_files = sorted(
            [os.path.join(display_path, f) for f in os.listdir(display_path) if
             f.endswith('.tif')])  # Change '.png' if you're using a different format

        # Read each 2D mask and stack them
        for rawim_file in rawim_files:
            im = imread(rawim_file)
            red_images.append(im[:, :, 0])
            images.append(im[:, :, 1])

        red_3d = np.stack(red_images, axis=-1)
        im_3d = np.stack(images, axis=-1)

        if 'Original Volume' in self.viewer.layers:
            self.viewer.layers['Original Volume'].data = im_3d
            self.viewer.layers['Protein Volume'].data = red_3d
        else:
            self.viewer.add_image(im_3d, name='Original Volume', colormap='green', blending='additive')
            self.viewer.add_image(red_3d, name='Protein Volume', colormap='red', blending='additive')
        self.viewer.dims.order = (2, 0, 1)
        self.loading_name.setText(self.filename_base)
###------------------------ Open Code Ends---------------------------------------------------------------
###------------------------ Trim Code starts---------------------------------------------------------------

    def trim_cochlea_files(self):

        if self.analysis_stage >= 2 :
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'The stack is already trimmed, if you want to change the trim decision, delete the current folder and restart the analysis')
            msg_box.exec_()
            return
        dialog = QDialog()
        dialog.setWindowTitle("Trim Cochlea Files")

        layout = QVBoxLayout()

        # Start number input
        start_label = QLabel("Start No:")
        start_input = QLineEdit()
        layout.addWidget(start_label)
        layout.addWidget(start_input)

        # End number input
        end_label = QLabel("End No:")
        end_input = QLineEdit()
        layout.addWidget(end_label)
        layout.addWidget(end_input)

        # OK and Cancel buttons
        buttons = QWidget()
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        buttons.setLayout(button_layout)

        layout.addWidget(buttons)
        dialog.setLayout(layout)

        def copy_files_in_range():
            total_digits = 4  # Assuming 4 digits for file numbering

            # Ensure destination path exists
            if not os.path.exists(self.full_stack_raw_images_trimmed):
                os.makedirs(self.full_stack_raw_images_trimmed)
            else:
                shutil.rmtree(self.full_stack_raw_images_trimmed)
                os.makedirs(self.full_stack_raw_images_trimmed)

            # Iterate over the range of file numbers
            for i in range(self.start_trim, self.end_trim + 1):
                padded_num = str(i).zfill(4)
                file_name_to_copy = f'{self.filename_base}_{padded_num}.tif'
                source_file = os.path.join(self.full_stack_raw_images, file_name_to_copy)
                dest_file = os.path.join(self.full_stack_raw_images_trimmed, file_name_to_copy)

                # Copy file if it exists
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
                else:
                    print(f"File not found: {source_file}")

        def process_input(start_no, end_no):
            # global full_stack_raw_images_trimmed
            try:
                self.start_trim = int(start_no)
                self.end_trim = int(end_no)
                if self.start_trim < self.end_trim <= self.full_stack_length:
                    self.full_stack_raw_images_trimmed = os.path.dirname(self.full_stack_raw_images)
                    self.full_stack_raw_images_trimmed = os.path.join(self.full_stack_raw_images_trimmed,
                                                                      'full_stack_raw_images_trimmed')
                    if not os.path.exists(self.full_stack_raw_images_trimmed):
                        os.makedirs(self.full_stack_raw_images_trimmed, exist_ok=True)
                    copy_files_in_range()
                    self.display = 1
                    self.analysis_stage = 2
                    self.save_attributes(self.pkl_Path)
                    self.display_images()

            except ValueError as e:
                print(f"Input Error: {e}")
            dialog.close()

        # Connect button signals
        ok_button.clicked.connect(lambda: process_input(start_input.text(), end_input.text()))
        cancel_button.clicked.connect(dialog.close)
        dialog.exec_()

###------------------------ Trim Code Ends---------------------------------------------------------------
###------------------------ Rotate Code starts-----------------------------------------------------------

    def Rotate_stack(self):
        if self.analysis_stage >= 3 :
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'The stack is already rotated, if you want to change the rotating decision, delete the current folder and restart the analysis')
            msg_box.exec_()
            return

        # global middle_image
        # global full_stack_raw_images_trimmed
        # global full_stack_rotated_images
        image_files = [f for f in sorted(os.listdir(self.full_stack_raw_images_trimmed)) if
                       f.endswith('.tif')]  # Adjust the extension if needed
        # Sort the list of images
        image_files.sort()
        # Calculate the index of the middle image
        middle_index = len(image_files) // 2
        # Get the name of the middle image
        middle_image_name = image_files[middle_index] if image_files else None
        middle_image = imread(os.path.join(self.full_stack_raw_images_trimmed, middle_image_name))
        # Create a pop-up window
        popup = QDialog()
        layout = QVBoxLayout(popup)

        def numpy_array_to_qpixmap(array):
            if array.dtype == np.uint8:
                if array.ndim == 2:
                    q_img = QImage(array, array.shape[1], array.shape[0], array.strides[0], QImage.Format_Grayscale8)
                elif array.ndim == 3 and array.shape[2] == 3:
                    q_img = QImage(array, array.shape[1], array.shape[0], array.strides[0], QImage.Format_RGB888)
                else:
                    raise ValueError("Unsupported array shape for QPixmap conversion.")
            else:
                raise ValueError("Unsupported array data type for QPixmap conversion.")
            return QPixmap.fromImage(q_img)

        # Add an image display label
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)  # Center-align the label
        pixmap = numpy_array_to_qpixmap(middle_image)
        desired_size = QSize(300, 300)  # You can adjust this to the size you want
        pixmap = pixmap.scaled(desired_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        layout.addWidget(label)

        # Slider for rotation
        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.setMinimum(-180)
        slider.setMaximum(180)
        layout.addWidget(slider)

        # Update the displayed image when the slider is moved
        def update_image(value):
            rotated = rotate(middle_image, value, reshape=False)
            pixmap = numpy_array_to_qpixmap(rotated)
            desired_size = QSize(300, 300)  # You can adjust this to the size you want
            pixmap = pixmap.scaled(desired_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)

        slider.valueChanged.connect(update_image)

        # OK and Cancel buttons
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        layout.addWidget(ok_button)
        layout.addWidget(cancel_button)

        # Define what happens when the OK button is clicked
        def on_ok():
            self.loading_label.setText("<font color='red'>Processing..., Wait</font>")
            QApplication.processEvents()
            if not os.path.exists(self.full_stack_rotated_images):
                os.makedirs(self.full_stack_rotated_images)
            else:
                shutil.rmtree(self.full_stack_rotated_images)
                os.makedirs(self.full_stack_rotated_images)
            angle = slider.value()
            # Apply the rotation to the entire stack
            images = []
            rawim_files = sorted(
                [os.path.join(self.full_stack_raw_images_trimmed, f) for f in os.listdir(self.full_stack_raw_images_trimmed) if
                 f.endswith('.tif')])  # Change '.png' if you're using a different format

            # Read each 2D mask and stack them
            for rawim_file in rawim_files:
                im = imread(rawim_file)
                im = rotate(im, angle, reshape=True)
                imwrite(os.path.join(self.full_stack_rotated_images, os.path.basename(rawim_file)), im)
            # Close the pop-up
            popup.close()
            self.display = 2
            self.analysis_stage = 3
            self.save_attributes(self.pkl_Path)
            self.display_images()
            self.loading_label.setText("")
            QApplication.processEvents()

        ok_button.clicked.connect(on_ok)
        cancel_button.clicked.connect(popup.close)
        popup.show()


###------------------------ Rotate Code Ends---------------------------------------------------------------
###------------------------ train Code starts---------------------------------------------------------------
    def check_training_data(self):
        self.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()
        def folder_content_check(dir_path):
            dir_contents = os.listdir(dir_path)
            for item in dir_contents:
                if item == 'Train' or item == 'Val':
                    continue
                else:
                    QMessageBox.warning(None, 'Folder Selection', 'Please select a folder that has only "Train" and "Val" folders')
                    return 0

        def check_masks(dir_path, png_files):
            for filename in png_files:
                with Image.open(os.path.join(dir_path, filename)) as Img:
                    Img = np.array(Img)
                    mask_labels = np.unique(Img)
                    if mask_labels.max() > 0:
                        continue
                    else:
                        print(filename)
                        return 0

                #
        if not os.path.exists(self.model_output_path):
            os.makedirs(self.model_output_path)
        else:
            dirfiles = os.listdir(self.model_output_path)
            if dirfiles != []:
                QMessageBox.warning(None, 'Output Model Error',
                                            'Output Model folder needs to be empty')
                self.loading_label.setText("")
                QApplication.processEvents()
                return

        dir_path = QFileDialog.getExistingDirectory()
        check = folder_content_check(dir_path)
        if check == 0:
            self.loading_label.setText("")
            QApplication.processEvents()
            return
        # Check if tif files are the same number as png files
        tif_files = [filename for filename in os.listdir(os.path.join(dir_path,'Train')) if filename.endswith('.tif')]
        png_files = [filename for filename in os.listdir(os.path.join(dir_path,'Train')) if filename.endswith('.png')]
        if len(tif_files) != len(png_files):
            QMessageBox.warning(None, 'Files Issues',
                                'Train Tif files and corresponding masks are not equal')
            self.loading_label.setText("")
            QApplication.processEvents()
            return
        # Check if each png file has only ID's from 1 to n, and also check if each png file has the same name as .tif file
        set1 = set([filename[:-3] for filename in tif_files])
        set2 = set([filename[:-3] for filename in png_files])
        if set1 != set2:
            QMessageBox.warning(None, 'Files Issues',
                                    'Train File names are not the same, each .tif file should have the same name as .png file')
            self.loading_label.setText("")
            QApplication.processEvents()
            return
        Train_mask_check = check_masks(os.path.join(dir_path, 'Train'), png_files)
        if Train_mask_check == 0:
            QMessageBox.warning(None, 'Files Issues',
                                'There is at least one mask in Train that does not have labels')
            self.loading_label.setText("")
            QApplication.processEvents()
            return
        # Val
        tif_files = [filename for filename in os.listdir(os.path.join(dir_path, 'Val')) if filename.endswith('.tif')]
        png_files = [filename for filename in os.listdir(os.path.join(dir_path, 'Val')) if filename.endswith('.png')]
        if len(tif_files) != len(png_files):
            QMessageBox.warning(None, 'Files Issues',
                                'Val Tif files and corresponding masks are not equal')
            self.loading_label.setText("")
            QApplication.processEvents()
            return
        # Check if each png file has only ID's from 1 to n, and also check if each png file has the same name as .tif file
        set1 = set([filename[:-3] for filename in tif_files])
        set2 = set([filename[:-3] for filename in png_files])
        if set1 != set2:
            QMessageBox.warning(None, 'Files Issues',
                                'Val File names are not the same, each .tif file should have the same name as .png file')
            self.loading_label.setText("")
            QApplication.processEvents()
            return
        Test_mask_check = check_masks(os.path.join(dir_path, 'Val'), png_files)
        if Test_mask_check == 0:
            QMessageBox.warning(None, 'Files Issues',
                                'There is at least one mask in Val that does not have labels')
            self.loading_label.setText("")
            QApplication.processEvents()
            return

        self.training_path = dir_path + '/'
        QMessageBox.warning(None, 'Files Check Complete',
                            'Congratulations: Click train buttom')
        self.loading_label.setText("")
        QApplication.processEvents()


    def train_cilia(self):
        if self.training_path == None:
            QMessageBox.warning(None, 'Check Training Data',
                                'Click first "Check Training Data" buttom')
            return

        currentfolder = os.path.join(self.training_path)  # this path will not be considered here because it is segmentation task but we need to have it
        currentfolder = currentfolder.replace(':', '').replace('\\', '/')
        currentfolder = '/mnt/' + currentfolder.lower()
        currentfolder = os.path.dirname(currentfolder) + '/'
        # folder_path, path to training data
        trainfolder = os.path.join(self.training_path)
        trainfolder = trainfolder.replace(':', '').replace('\\', '/')
        trainfolder = '/mnt/' + trainfolder.lower()
        trainfolder = os.path.dirname(trainfolder) + '/'
        #
        output_model_path = self.model_output_path
        output_model_path = output_model_path.replace(':', '').replace('\\', '/')
        output_model_path = '/mnt/' + output_model_path.lower()
        output_model_path = os.path.dirname(output_model_path) + '/'

        command = f'wsl {self.wsl_executable} --train_predict {0} --folder_path {trainfolder}  --model_output_path {output_model_path} --iterations {self.train_iter} --rootfolder {currentfolder} --model {self.model}  --threshold {0.7}'

        # Configuring the time
        total_iterations = self.train_iter
        time_per_1000_iterations = 10 * 60  # 10 minutes in seconds
        total_time = (total_iterations / 1000) * time_per_1000_iterations

        # Create and configure the progress dialog
        progress_dialog = QDialog()
        progress_dialog.setWindowFlags(progress_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        progress_dialog.setWindowTitle('Training in progress, wait....')
        progress_dialog.setFixedSize(300, 100)
        layout = QVBoxLayout()
        progress_bar = QProgressBar(progress_dialog)
        layout.addWidget(progress_bar)
        progress_dialog.setLayout(layout)

        def center_widget_on_screen(widget):
            frame_geometry = widget.frameGeometry()
            screen_center = QDesktopWidget().availableGeometry().center()
            frame_geometry.moveCenter(screen_center)
            widget.move(frame_geometry.topLeft())

        center_widget_on_screen(progress_dialog)
        progress_dialog.show()
        progress_bar.setMaximum(100)  # Set the maximum value

        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   bufsize=1, universal_newlines=True)

        start_time = time.time()

        def update_progress_bar():
            line = process.stdout.readline()
            if line:
                print(line)
                # Check if the process outputs "done"
                if "done" in line:
                    progress_bar.setValue(100)
                    timer.stop()
                    process.stdout.close()
                    process.stderr.close()
                    progress_dialog.close()
                    return

            elapsed_time = time.time() - start_time
            progress = (elapsed_time / total_time) * 95  # Cap the progress at 95 until "done" is read
            progress_bar.setValue(min(int(progress), 95))
            QApplication.processEvents()

            if process.poll() is not None and progress_bar.value() != 100:
                # If the process has ended but "done" wasn't read
                print("Process ended unexpectedly.")
                timer.stop()
                process.stdout.close()
                process.stderr.close()
                progress_dialog.close()

        # Set up a timer to periodically update the progress
        timer = QTimer()
        timer.timeout.connect(update_progress_bar)
        timer.start(150)  # Update every 30 seconds

    ###------------------------ train Code ends---------------------------------------------------------------
###------------------------ Segment Code starts------------------------------------------------------------

    def segment_with_3DCiliaSeg(self):
        if self.analysis_stage < 3:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'Please press "Rotate" buttom')
            msg_box.exec_()
            return
        if self.analysis_stage >= 4:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'The stack is already segmented')
            msg_box.exec_()
            return
        currentfolder = os.path.join(self.rootfolder, self.full_stack_rotated_images.strip('./'))
        currentfolder = currentfolder.replace(':', '').replace('\\', '/')
        currentfolder = '/mnt/' + currentfolder.lower()
        currentfolder = os.path.dirname(currentfolder) + '/'

        # You don't need these for prediction but we need to put them in the command
        # folder_path, path to training data
        trainfolder = os.path.join(self.rootfolder)
        trainfolder = trainfolder.replace(':', '').replace('\\', '/')
        trainfolder = '/mnt/' + trainfolder.lower()
        trainfolder = os.path.dirname(trainfolder) + '/'
        #
        output_model_path = self.model_output_path
        output_model_path = output_model_path.replace(':', '').replace('\\', '/')
        output_model_path = '/mnt/' + output_model_path.lower()
        output_model_path = os.path.dirname(output_model_path) + '/'
        # Construct the command to run in WSL
        #command = f'wsl {self.wsl_executable} --rootfolder {currentfolder} --model {self.model}'
        command = f'wsl {self.wsl_executable} --train_predict {1} --folder_path {trainfolder}  --model_output_path {output_model_path} --iterations {self.train_iter} --rootfolder {currentfolder} --model {self.model}  --threshold {0.7}'

        # Create and configure the progress dialog
        progress_dialog = QDialog()
        progress_dialog.setWindowFlags(progress_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        progress_dialog.setWindowTitle('Segmentation in progress, wait....')
        progress_dialog.setFixedSize(300, 100)
        layout = QVBoxLayout()
        progress_bar = QProgressBar(progress_dialog)
        layout.addWidget(progress_bar)
        progress_dialog.setLayout(layout)

        def center_widget_on_screen(widget):
            frame_geometry = widget.frameGeometry()
            screen_center = QDesktopWidget().availableGeometry().center()
            frame_geometry.moveCenter(screen_center)
            widget.move(frame_geometry.topLeft())

        center_widget_on_screen(progress_dialog)
        progress_dialog.show()
        progress_bar.setMaximum(100)  # Set the maximum value

        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   bufsize=1, universal_newlines=True)

        def update_progress_bar():
            line = process.stdout.readline()
            if line:
                print(line)  # Optional: print the output line (remove if not needed)
                match = re.search(r"##(\d+(?:\.\d+)?)%", line)
                if match:
                    progress = float(match.group(1))
                    progress_bar.setValue(int(round(progress)))  # Convert to integer and round
                    # Process GUI events to update the progress bar
                    QApplication.processEvents()
            else:
                # Check if the subprocess has finished
                if process.poll() is not None:
                    # Close the progress dialog when done
                    timer.stop()
                    process.stdout.close()
                    process.stderr.close()
                    progress_dialog.close()

        # Set up a timer to periodically check the output
        timer = QTimer()
        timer.timeout.connect(update_progress_bar)
        timer.start(150)  # Check every 100 milliseconds

###------------------------ Segment Code ends------------------------------------------------------------
###------------------------ Track and Visualize Code starts------------------------------------------------------------
    def visualize_segmentation_Detecron2_with_tracking(self):
        self.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()
        def overlap_with_previous(component, previous_mask):
            """Returns the label of the overlap from the previous mask, 0 if none."""
            overlap = np.bincount(previous_mask[component].flatten())
            overlap[0] = 0  # ignore the background
            if overlap.max() > 300:
                return overlap.argmax()
            else:
                return 0

        self.npy_dir = os.path.dirname(self.full_stack_rotated_images) + '/prediction/'  # This is for prediction after wsl run
        self.obj_dir = os.path.dirname(
            self.full_stack_rotated_images) + '/new_assignment_obj/'  # This is for visulization after the tracking algorithm runs

        npy_files = [f for f in os.listdir(self.npy_dir) if f.endswith('.npy')]
        # -- new_assignment_obj
        if not os.path.exists(self.obj_dir):
            os.makedirs(self.obj_dir)

        # Assuming your masks are in a sorted list called mask_paths
        previous_mask = None
        latest_label = 0

        for im_file in npy_files:
            file_path = os.path.join(self.npy_dir, os.path.basename(im_file))
            # Load the .npy file
            data = np.load(file_path, allow_pickle=True).item()
            # Extract the bounding boxes, scores, and masks
            boxes = data['boxes']
            scores = data['scores']
            masks = data['masks']
            labeled_mask = np.zeros_like(masks[0], dtype=np.int32)
            for i, mask in enumerate(masks):
                labeled_mask[mask > 0] = i + 1
            num_features = i + 1
            temp = labeled_mask.copy()
            # If this isn't the first mask
            if previous_mask is not None:
                for i in range(1, num_features + 1):
                    component = (labeled_mask == i)
                    # plt.imshow(component)
                    # plt.show()
                    # Check overlap with previous mask
                    overlap_label = overlap_with_previous(component, previous_mask)

                    if overlap_label:
                        temp[component] = overlap_label
                    else:
                        latest_label += 1
                        temp[component] = latest_label

            else:  # if this is the first mask
                latest_label = num_features

            imageio.imwrite(self.obj_dir + os.path.basename(im_file).replace('.npy', '.png'), temp.astype(np.uint8))

            # Update the previous mask for the next iteration
            previous_mask = temp
        # -------------------------------------------------
        masks = []
        newmask_files = sorted([os.path.join(self.obj_dir, f) for f in os.listdir(self.obj_dir) if
                                f.endswith('.png')])  # Change '.png' if you're using a different format
        # Read each 2D mask and stack them
        for maskfile in newmask_files:
            mask = imread(maskfile)
            masks.append(mask)

        self.labeled_volume = np.stack(masks, axis=-1)  # This stacks along the third dimension

        self.filtered_ids = []
        regions = find_objects(self.labeled_volume)

        for i in range(1, len(regions) + 1):
            region_slice = regions[i - 1]  # Get the slice for the i-th region
            depth = region_slice[2].stop - region_slice[2].start  # Calculate depth

            if depth <= 3:
                self.filtered_ids.append(i)
                self.labeled_volume[self.labeled_volume == i] = 0  # Filter out the component by setting it to zero

        print(f"Filtered IDs: {self.filtered_ids}")
        self.num_components = len(np.unique(self.labeled_volume)) - 1  # Subtract 1 to exclude background

        #
        self.analysis_stage = 4
        self.save_attributes(self.pkl_Path)
        self.viewer.add_labels(self.labeled_volume, name='Labeled Image')
        #
        self.viewer.layers['Original Volume'].colormap = 'gray'
        self.viewer.layers['Protein Volume'].visible = not self.viewer.layers['Protein Volume'].visible
        #
        self.loading_label.setText("")
        QApplication.processEvents()


###------------------------ Track and Visualize Code ends------------------------------------------------------------
###------------------------ Delete Code starts-----------------------------------------------------------------------
    def create_filter_component_widget(self):
        @magicgui(call_button="Delete Label")
        def _widget(component: int):
            self.delete_label(component)

        return _widget

    def delete_label(self, component: int):
        self.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()
        label_to_remove = component
        self.labeled_volume[self.labeled_volume == label_to_remove] = 0
        self.viewer.layers['Labeled Image'].data = self.labeled_volume
        self.filtered_ids.append(label_to_remove)
        self.num_components -= 1
        #--------- here we check if clustering is performed because that means I have new mask to be considered in dleteing the component self.IHC,self.IHC_OHC, self.OHC1, self.OHC2, self.OHC3, self.OHC, ...

        if self.clustering == 1:
            coordinates_to_remove = np.argwhere(self.labeled_volume == label_to_remove)  # this is if self.clustering == 1
            mask = np.zeros_like(self.IHC, dtype = bool)
            mask[tuple(coordinates_to_remove.T)] = True
            self.IHC[mask] = 0
            self.OHC[mask] = 0
            self.IHC_OHC[mask] = 0
            self.OHC1[mask] = 0
            self.OHC2[mask] = 0
            self.OHC3[mask] = 0
            self.clustered_cells[mask] = 0
            self.viewer.layers['Clustered Cells'].data = self.clustered_cells
            layer_name = 'IHCs vs OHCs'
            if layer_name in self.viewer.layers:
                self.viewer.layers['IHCs vs OHCs'].data = self.IHC_OHC
            # for coordinates in coordinates_to_remove:
            #     self.IHC[tuple(coordinates)] = 0
            #     self.IHC_OHC[tuple(coordinates)] = 0
            #     self.OHC[tuple(coordinates)] = 0
            #     self.OHC1[tuple(coordinates)] = 0
            #     self.OHC2[tuple(coordinates)] = 0
            #     self.OHC3[tuple(coordinates)] = 0
        #-------------------------------- Here handle the case when calculate distance is pressed before save distances
        if self.analysis_stage == 5:
            for idpoints,idcc in self.IDtoPointsMAP:
                if idcc == component:
                    myidpoints = idpoints
                    break
            templist = list(self.start_points)
            del templist[myidpoints]
            self.start_points = np.array(templist)
            templist = list(self.end_points)
            del templist[myidpoints]
            self.end_points = np.array(templist)
            templist = list(self.IDs)
            del templist[myidpoints]
            self.IDs = np.array(templist)
            #---
            self.viewer.layers['Peak Points'].data = self.start_points
            self.viewer.layers['Base Points'].data = self.end_points
            self.start_points_layer.data = self.start_points
            self.end_points_layer.data = self.end_points
            new_lines = []
            if len(self.start_points_layer.data) == len(self.end_points_layer.data):
                for start, end in zip(self.start_points_layer.data, self.end_points_layer.data):
                    new_lines.append([start, end])
            self.viewer.layers['Lines'].data = new_lines
            IDtoPointsMAP_list = []
            tempid = 0
            for cc in range(self.num_components + 1 + len(self.filtered_ids)):
                if cc == 0 or cc in self.filtered_ids:
                    continue
                IDtoPointsMAP_list.append((tempid, cc))
                tempid = tempid + 1
            self.IDtoPointsMAP = tuple(IDtoPointsMAP_list)
        # -------------------------------- Here handle the case after save distances
        if self.analysis_stage == 6:
            for idpoints,idcc in self.IDtoPointsMAP:
                if idcc == component:
                    myidpoints = idpoints
                    break
            # WE need to update those lists when a connected component is deleted
            templist = list(self.start_points_most_updated)
            del templist[myidpoints]
            self.start_points_most_updated = np.array(templist)
            templist = list(self.end_points_most_updated)
            del templist[myidpoints]
            self.end_points_most_updated = np.array(templist)
            templist = list(self.start_points)
            del templist[myidpoints]
            self.start_points = np.array(templist)
            templist = list(self.end_points)
            del templist[myidpoints]
            self.end_points = np.array(templist)
            templist = list(self.IDs)
            del templist[myidpoints]
            self.IDs = np.array(templist)
            #------------------------------------------------------------------------------
            self.viewer.layers['Peak Points'].data = self.start_points_most_updated
            self.viewer.layers['Base Points'].data = self.end_points_most_updated
            self.start_points_layer.data = self.start_points_most_updated
            self.end_points_layer.data = self.end_points_most_updated
            new_lines = []
            if len(self.start_points_layer.data) == len(self.end_points_layer.data):
                for start, end in zip(self.start_points_layer.data, self.end_points_layer.data):
                    new_lines.append([start, end])
            self.viewer.layers['Lines'].data = new_lines
            IDtoPointsMAP_list = []
            tempid = 0
            for cc in range(self.num_components + 1 + len(self.filtered_ids)):
                if cc == 0 or cc in self.filtered_ids:
                    continue
                IDtoPointsMAP_list.append((tempid, cc))
                tempid = tempid + 1
            self.IDtoPointsMAP = tuple(IDtoPointsMAP_list)

        self.save_attributes(self.pkl_Path)
        self.loading_label.setText("")


###------------------------ Delete Code ends-------------------------------------------------------------------
###------------------------ Measurments Code starts------------------------------------------------------------
    def calculate_measurements(self):

        measurements_dir = self.rootfolder + '/' + self.filename_base + '/measurements/'

        if not os.path.exists(measurements_dir):
            os.makedirs(measurements_dir)

        props = regionprops(self.labeled_volume)
        measurements_list = []

        for prop in props:
            label = prop.label
            volume = prop.area  # in voxels
            centroid = prop.centroid

            # Create a dictionary for each label's properties and append it to the list
            measurements_list.append({
                'Label': label,
                'Volume (voxels)': volume,
                'Centroid (y, x, z)': centroid
            })

        # Convert the list of dictionaries to a DataFrame and export to CSV
        global_prop = props
        df = pd.DataFrame(measurements_list)
        df.to_csv(os.path.join(measurements_dir + "measurements.csv"), index=False)

    ###------------------------ Measurments Code ends------------------------------------------------------------
    def calculate_orientation(self):
        msg = QMessageBox()
        msg.setWindowTitle("Information")
        msg.setText("Future versions")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()
        # z_middle_slice = self.labeled_volume.shape[2]//2
        # middle_layer = self.labeled_volume[:,:,z_middle_slice]
        #
        # def calculate_angle(pt1, pt2):
        #     angle_rad = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        #     angle_deg = np.degrees(angle_rad)
        #     return angle_deg
        #
        # contour_ids = np.unique(middle_layer[middle_layer!=0])
        # points_data = []
        # text_data = []
        # for ccid in contour_ids:
        #     img = np.uint8(middle_layer == ccid) * 255
        #     contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     if len(contours) > 0 and len(contours[0]) >= 5:
        #         # Fit an ellipse to the first contour (the segmented object)
        #         ellipse = cv2.fitEllipse(contours[0])
        #
        #     if ccid == 14:
        #         print('done')
        #     # Extract the angle
        #     (x, y), (MA, ma), angle = ellipse
        #     # Convert to angle relative to horizontal axis
        #     angle_relative_to_horizontal = np.abs(90 - angle)
        #     print(f"The orientation angle of the stereocilia bundle {ccid} is: {angle_relative_to_horizontal} degrees")
            #skeleton = skeletonize(img)

            # for contour in contours:
            #     # Identify extremal points
            #     leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
            #     rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
            #     topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            #     bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
            # y_coords, x_coords = np.where(skeleton)
            # # Find leftmost and rightmost points
            # leftmost_point = [y_coords[np.argmin(x_coords)], np.min(x_coords)]
            # rightmost_point = [y_coords[np.argmax(x_coords)], np.max(x_coords)]
            #
            # angle = calculate_angle(leftmost_point, rightmost_point)
            #     text_data.append(f"{angle:.2f}°")
            #     points_data.append(topmost)


        #props = regionprops(middle_layer)
        # # Prepare the data for the Points layer

        #
        # Convert orientations to degrees and ensure they are within 0-360 range
        # for prop in props:
        #     #orientation_degrees = np.degrees(prop.orientation) % 360
        #     text_data.append(f"{orientation_degrees:.2f}°")
        #     points_data.append(prop.centroid[::-1])  # Reversing to get (x,y) coordinates for Napari

        # Create a Points layer with the orientation text
        #self.viewer.add_points(np.array(points_data), size=5, face_color='red', name='Orientation Points', text=text_data)
        # points_3D = [(y, x, 0) for x, y in points_data]
        # self.viewer.add_points(np.array(points_3D),
        #                        size=0,  # Makes points effectively invisible
        #                        face_color='transparent',  # No color for points
        #                        name='Orientation Points',
        #                        text=text_data)  # Red color for text
    def Compute_Protein_Intensity(self):

        def plot_responces(label3D, image_3d, celltype, barcolor, min_intensity, max_intensity, max_mean_intensity):
            props = regionprops(label3D, intensity_image=image_3d)
            # Initialize lists to store mean and total intensities
            mean_intensities = []
            total_intensities = []
            labels = []
            # Collect mean and total intensity for each region
            for region in props:
                labels.append(region.label)
                mean_intensities.append(region.mean_intensity)
                total_intensities.append(region.intensity_image.sum())

            # CSV file
            # Now, write the collected data to a CSV file
            with open(intensity_dir + '/' + celltype + '/' + 'region_intensities.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(['Region ID', 'Mean Intensity', 'Total Intensity'])

                # Write the data
                for label, mean_intensity, total_intensity in zip(labels, mean_intensities, total_intensities):
                    writer.writerow([label, mean_intensity, total_intensity])

            if  celltype == 'Allcells':
                total_intensities = np.array(total_intensities)
                min_intensity = total_intensities.min()
                max_intensity = total_intensities.max()
                total_intensities = (total_intensities - min_intensity) / (max_intensity - min_intensity)
                mean_intensities = np.array(mean_intensities)
                max_mean_intensity = mean_intensities.max()
            else:
                total_intensities = np.array(total_intensities)
                total_intensities = (total_intensities - min_intensity) / (max_intensity - min_intensity)


            plt.figure(figsize=(12, 6))
            plt.bar(labels, mean_intensities, color=barcolor)
            plt.title('Mean Intensity for Each Hair Cell')
            plt.xlabel('Stereocilia Bundle')
            plt.ylabel('Mean Intensity')
            plt.ylim(0, max_mean_intensity)
            plt.xticks(labels)
            plt.tight_layout()
            plt.savefig(intensity_dir + '/' + celltype + '/' + 'mean_intensity_per_cell.png', dpi=300)

            plt.figure(figsize=(12, 6))
            plt.bar(labels, total_intensities, color=barcolor)
            plt.title('Total Intensity for Each Hair Cell')
            plt.xlabel('Stereocilia Bundle')
            plt.ylabel('Total Intensity')
            plt.xticks(labels)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(intensity_dir + '/' + celltype + '/' + 'total_intensity_per_cell.png', dpi=300)
            return min_intensity, max_intensity, max_mean_intensity


        intensity_dir = self.rootfolder + '/' + self.filename_base + '/Protein_responce/'

        if not os.path.exists(intensity_dir + '/' + 'Allcells'):
            os.makedirs(intensity_dir + '/' + 'Allcells')

        if not os.path.exists(intensity_dir + '/' + 'IHCs'):
            os.makedirs(intensity_dir + '/' + 'IHCs')

        if not os.path.exists(intensity_dir + '/' + 'OHCs'):
            os.makedirs(intensity_dir + '/' + 'OHCs')

        if not os.path.exists(intensity_dir + '/' + 'OHC1'):
            os.makedirs(intensity_dir + '/' + 'OHC1')

        if not os.path.exists(intensity_dir + '/' + 'OHC2'):
            os.makedirs(intensity_dir + '/' + 'OHC2')

        if not os.path.exists(intensity_dir + '/' + 'OHC3'):
            os.makedirs(intensity_dir + '/' + 'OHC3')

        image_files = [f for f in os.listdir(self.full_stack_rotated_images) if
                       f.endswith('.tif')]  # Adjust the extension based on your image files
        image_3d = np.zeros(self.labeled_volume.shape, dtype = np.uint8)
        for idx, image in enumerate(image_files):
            im = imread(os.path.join(self.full_stack_rotated_images,image))
            redch = im[:,:,0]
            image_3d[:,:,idx] = redch

        [min_intensity, max_intensity, max_mean_intensity] = plot_responces(self.labeled_volume, image_3d, 'Allcells','green', 0, 0, 0)
        [_,_,_] = plot_responces(self.IHC, image_3d, 'IHCs', 'yellow',min_intensity, max_intensity, max_mean_intensity)
        [_,_,_] = plot_responces(self.OHC, image_3d, 'OHCs', 'red', min_intensity, max_intensity, max_mean_intensity)
        [_, _, _] = plot_responces(self.OHC1, image_3d, 'OHC1', 'skyblue', min_intensity, max_intensity, max_mean_intensity)
        [_, _, _] = plot_responces(self.OHC2, image_3d, 'OHC2', 'lightgreen', min_intensity, max_intensity, max_mean_intensity)
        [_, _, _] = plot_responces(self.OHC3, image_3d, 'OHC3', 'thistle', min_intensity, max_intensity, max_mean_intensity)

        print('done')
    # Create a magicgui widget for the clustering function
    def create_clustering_widget(self):
        @magicgui(call_button='Perform Cell Clustering', method={'choices': ['GMM', 'KMeans']})
        def clustering_widget(method: str):
            self.perform_clustering(method)
        return clustering_widget

    def perform_clustering(self, method: str):
        self.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()
        if self.analysis_stage < 5:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Distance Calculation')
            msg_box.setText(
                'Press first Calculate Distance')
            msg_box.exec_()
            # Exit the function
            return
        self.viewer.layers['Lines'].visible = False
        self.viewer.layers['Base Points'].visible = False
        self.viewer.layers['Peak Points'].visible = False
        self.viewer.layers['Labeled Image'].visible = False
        layer_name = 'IHCs vs OHCs'
        if layer_name in self.viewer.layers:
            self.viewer.layers['IHCs vs OHCs'].visible = False

        if method == 'GMM':
            Basepoints_y = [point[0] for point in self.end_points]  # Extracting the y-values
            lebel_list = []
            for ((point_lbl, reallable), val) in zip(self.IDtoPointsMAP, Basepoints_y):
                lebel_list.append({
                    'Label': reallable,
                    'Centroid (y, x)': val})
            Basepoints_y_array = np.array(Basepoints_y).reshape(-1, 1)

            gmm = GaussianMixture(n_components=4, random_state=0)
            gmm.fit(Basepoints_y_array)
            labels = gmm.predict(Basepoints_y_array)
            cluster_centers = gmm.means_

        elif method == 'KMeans':
            props = regionprops(self.labeled_volume)
            centroids = []
            lebel_list = []
            for prop in props:
                label = prop.label
                centroid = prop.centroid[0]

                # Create a dictionary for each label's properties and append it to the list
                lebel_list.append({
                    'Label': label,
                    'Centroid (y, x)': centroid
                })
                centroids.append(centroid)
            centroids = np.array(centroids).reshape(-1, 1)
            kmeans = KMeans(n_clusters=4, random_state=0)
            kmeans.fit(centroids)
            labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

        # Prepare a list of labeled centroids to return
        labeled_centroids = [
            {"Label": lbl_dict["Label"], "Centroid": lbl_dict["Centroid (y, x)"], "Cluster": int(cluster_label)}
            for lbl_dict, cluster_label in zip(lebel_list, labels)]

        # Step 1: Create a mapping from original labels to cluster IDs
        label_to_cluster = {item['Label']: item['Cluster'] for item in labeled_centroids}

        # Step 2: Create self.labeled_volume_clustered by replacing labels with cluster IDs
        self.labeled_volume_clustered = np.copy(self.labeled_volume)

        # Replace each label in the volume with its corresponding cluster ID
        for original_label, cluster_id in label_to_cluster.items():
            self.labeled_volume_clustered[self.labeled_volume == original_label] = cluster_id + 9
        #
        #
        ihc_cluster = np.argmax(cluster_centers[:, 0])
        for item in labeled_centroids:
            item['Cell Type'] = 'IHC' if item['Cluster'] == ihc_cluster else 'OHC'
        self.IHC_OHC = np.copy(self.labeled_volume_clustered)
        for label_info in labeled_centroids:
            label_val = 9 if label_info['Cell Type'] == 'IHC' else 10  # Example: 9 for IHC, 10 for OHC
            original_label = label_info['Label']
            self.IHC_OHC[self.labeled_volume == original_label] = label_val
        self.IHC = np.zeros_like(self.IHC_OHC)
        self.OHC = np.zeros_like(self.IHC_OHC)
        self.IHC[self.IHC_OHC == 9] = 1
        self.OHC[self.IHC_OHC == 10] = 1
        self.IHC = self.IHC * self.labeled_volume
        self.OHC = self.OHC * self.labeled_volume
        # identify OHC1
        cluster_centers[ihc_cluster] = 0
        ohc1_cluster = np.argmax(cluster_centers[:, 0])
        for item in labeled_centroids:
            item['Cell Type'] = 'OHC1' if item['Cluster'] == ohc1_cluster else 'any'
        self.OHC1 = np.copy(self.labeled_volume_clustered)
        for label_info in labeled_centroids:
            label_val = 10 if label_info['Cell Type'] == 'OHC1' else 11  # Example: 9 for IHC, 10 for OHC
            original_label = label_info['Label']
            self.OHC1[self.labeled_volume == original_label] = label_val
        temp = self.OHC1
        self.OHC1[temp == 10] = 1
        self.OHC1[temp == 11] = 0
        self.OHC1 = self.OHC1 * self.labeled_volume
        # identify OHC2
        cluster_centers[ohc1_cluster] = 0
        ohc2_cluster = np.argmax(cluster_centers[:, 0])
        for item in labeled_centroids:
            item['Cell Type'] = 'OHC2' if item['Cluster'] == ohc2_cluster else 'any'
        self.OHC2 = np.copy(self.labeled_volume_clustered)
        for label_info in labeled_centroids:
            label_val = 11 if label_info['Cell Type'] == 'OHC2' else 12  # Example: 9 for IHC, 10 for OHC
            original_label = label_info['Label']
            self.OHC2[self.labeled_volume == original_label] = label_val
        temp = self.OHC2
        self.OHC2[temp == 11] = 1
        self.OHC2[temp == 12] = 0
        self.OHC2 = self.OHC2 * self.labeled_volume
        # identify OHC3
        cluster_centers[ohc2_cluster] = 0
        ohc3_cluster = np.argmax(cluster_centers[:, 0])
        for item in labeled_centroids:
            item['Cell Type'] = 'OHC3' if item['Cluster'] == ohc3_cluster else 'any'
        self.OHC3 = np.copy(self.labeled_volume_clustered)
        for label_info in labeled_centroids:
            label_val = 12 if label_info['Cell Type'] == 'OHC3' else 13  # Example: 9 for IHC, 10 for OHC
            original_label = label_info['Label']
            self.OHC3[self.labeled_volume == original_label] = label_val
        temp = self.OHC3
        self.OHC3[temp == 12] = 1
        self.OHC3[temp == 13] = 0
        self.OHC3 = self.OHC3 * self.labeled_volume
        #
        self.clustered_cells = np.copy(self.labeled_volume) # to visualize it with colors
        self.clustered_cells[self.OHC3 > 0] = 28
        self.clustered_cells[self.OHC2 > 0] = 37
        self.clustered_cells[self.OHC1 > 0] = 16
        self.clustered_cells[self.IHC > 0] = 9

        # This variable to let the software knows that clustering is done, mainly for delete buttom
        self.clustering = 1
        self.save_attributes(self.pkl_Path)

        layer_name = 'Clustered Cells'
        if layer_name in self.viewer.layers:
            self.viewer.layers['Clustered Cells'].data = self.clustered_cells
            self.viewer.layers['Clustered Cells'].visible = True
        else:
            self.viewer.add_labels(self.clustered_cells, name='Clustered Cells')
        self.loading_label.setText("")
        QApplication.processEvents()

    def find_IHC_OHC(self):
        self.viewer.layers['Clustered Cells'].visible = False
        layer_name = 'IHCs vs OHCs'
        if layer_name in self.viewer.layers:
            self.viewer.layers['IHCs vs OHCs'].data = self.IHC_OHC
            self.viewer.layers['IHCs vs OHCs'].visible = True
        else:
            self.viewer.add_labels(self.IHC_OHC, name='IHCs vs OHCs')
            # self.viewer.add_labels(self.OHC1, name='OHC1')
            # self.viewer.layers['OHC1'].visible = False
            # self.viewer.add_labels(self.OHC2, name='OHC2')
            # self.viewer.layers['OHC2'].visible = False
            # self.viewer.add_labels(self.OHC3, name='OHC3')
            # self.viewer.layers['OHC3'].visible = False
            # self.viewer.add_labels(self.IHC, name='IHC')
            # self.viewer.layers['IHC'].visible = False
            # self.viewer.add_labels(self.OHC, name='OHC')
            # self.viewer.layers['OHC'].visible = False

    def save_distance(self):
        if self.analysis_stage < 5:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Distance Calculation')
            msg_box.setText(
                'Press first Calculate Distance')
            msg_box.exec_()
            # Exit the function
            return
        for idx, points in enumerate(self.start_points):
            self.start_points_layer.data[idx][2] = points[2]
        for idx, points in enumerate(self.end_points):
            self.end_points_layer.data[idx][2] = points[2]

        dx = self.physical_resolution[0]
        dy = self.physical_resolution[1]
        dz = self.physical_resolution[2]
        # Calculate distances
        self.physical_distances = []
        for sp, ep, id in zip(self.start_points_layer.data, self.end_points_layer.data, self.IDs):
            point1_phy = np.array([sp[0] * dy, sp[1] * dx, sp[2] * dz])
            point2_phy = np.array([ep[0] * dy, ep[1] * dx, ep[2] * dz])
            distance = np.linalg.norm(point1_phy - point2_phy)
            self.physical_distances.append((id, distance))
            print(f'Distance of ID {id} = {distance}')
        # dst_folder = './'
        # base_name = os.path.splitext(self.file_path)[0]
        # self.filename_base = base_name.split('/')[-1].replace(' ', '')
        #new_folder_path = os.path.join(dst_folder, self.filename_base)
        df = pd.DataFrame(self.physical_distances, columns=['ID', 'Distance'])
        #Distance_path = new_folder_path + '/' + 'Distances'
        Distance_path = self.rootfolder + '/' + self.filename_base + '/Distances/'
        if not os.path.exists(Distance_path):
            os.makedirs(Distance_path, exist_ok=True)
        df.to_csv(Distance_path + '/' + 'Physical_distances.csv', index=False, sep=',')
        self.start_points_most_updated = self.start_points_layer.data
        self.end_points_most_updated = self.end_points_layer.data
        self.analysis_stage = 6
        #--- to solve the adding layers problem during changing the peak and base points
        lines = [np.vstack([start, end]) for start, end in zip(self.start_points_layer.data, self.end_points_layer.data)]
        self.viewer.layers['Peak Points'].data = self.start_points_layer.data
        self.viewer.layers['Base Points'].data = self.end_points_layer.data
        self.viewer.layers['Lines'].data = lines
        #-------------------------------------------------------------------------------
        self.save_attributes(self.pkl_Path)

    def update_lines(self, event=None):
        # Assuming each start point connects to an end point with the same index
        new_lines = []
        # Here I'm letting only x,y to be change by the user and keep the algorithmic z
        # for idx, points in enumerate(start_points):
        #     self.start_points_layer.data[idx][2] = points[2]
        # for idx, points in enumerate(end_points):
        #     self.end_points_layer.data[idx][2] = points[2]
        if len(self.start_points_layer.data) == len(self.end_points_layer.data):
            for start, end in zip(self.start_points_layer.data, self.end_points_layer.data):
                new_lines.append([start, end])

        self.lines_layer.data = new_lines

    def calculate_distance(self):
        def store_manual_resolution(x_res, y_res, z_res, dialog):
            try:
                # Store the physical resolution
                self.physical_resolution = (float(x_res), float(y_res), float(z_res))
                print(f"Physical resolution set to: {self.physical_resolution}")
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

        def find_resolution():
            try:
                with czifile.CziFile(self.file_path) as czi:
                    array6d = czi.asarray()
                _, mdata, _ = read_tools.read_6darray(self.file_path,
                                                                    output_order="STCZYX",
                                                                    use_dask=True,
                                                                    chunk_zyx=False,
                                                                    # T=0,
                                                                    # Z=0
                                                                    # S=0
                                                                    # C=0
                                                                    )

                # Store the physical resolution
                scaling_x_meters = mdata.channelinfo.czisource[
                    'ImageDocument'].Metadata.Experiment.ExperimentBlocks.AcquisitionBlock.AcquisitionModeSetup.ScalingX
                scaling_y_meters = mdata.channelinfo.czisource[
                    'ImageDocument'].Metadata.Experiment.ExperimentBlocks.AcquisitionBlock.AcquisitionModeSetup.ScalingY
                scaling_z_meters = mdata.channelinfo.czisource[
                    'ImageDocument'].Metadata.Experiment.ExperimentBlocks.AcquisitionBlock.AcquisitionModeSetup.ScalingZ
                scaling_x_micrometers = float(scaling_x_meters) * 1e6
                scaling_y_micrometers = float(scaling_y_meters) * 1e6
                scaling_z_micrometers = float(scaling_z_meters) * 1e6
                self.physical_resolution = (scaling_x_micrometers, scaling_y_micrometers, scaling_z_micrometers)
            except Exception as e:
                # If an error occurred, prompt the user to input the resolution manually
                print(f"Error retrieving resolution: {e}")
                prompt_for_resolution()

        def find_centroid(binary_image):
            # Make sure the image is binary (0s and 1s)
            assert np.isin(binary_image, [0, 1]).all(), "Image should be binary"

            # Find the indices of all object (1) pixels
            object_pixels = np.argwhere(binary_image == 1)

            # Calculate the centroid coordinates
            centroid_x = int(np.mean(object_pixels[:, 0]))
            centroid_y = int(np.mean(object_pixels[:, 1]))
            return centroid_x, centroid_y

        def center_widget_on_screen(widget):
            frame_geometry = widget.frameGeometry()
            screen_center = QDesktopWidget().availableGeometry().center()
            frame_geometry.moveCenter(screen_center)
            widget.move(frame_geometry.topLeft())

        if self.analysis_stage == 5 or self.analysis_stage == 6:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Distance Calculation')
            msg_box.setText(
                'Distances are already calculated')
            msg_box.exec_()
            # Exit the function
            return
        #-- finding resolution
        find_resolution()
        progress_dialog = QDialog()
        progress_dialog.setWindowTitle('Distance Calculation Progress')
        progress_dialog.setFixedSize(300, 100)
        layout = QVBoxLayout()
        progress_bar = QProgressBar(progress_dialog)
        layout.addWidget(progress_bar)
        progress_dialog.setLayout(layout)
        center_widget_on_screen(progress_dialog)
        progress_dialog.show()
        # Update the progress bar
        progress_bar.setMaximum(100)  # Set the maximum value
        progress_per_item = 100 / (self.num_components + 1 + len(self.filtered_ids))
        print(f'filtered_ids={self.filtered_ids}')
        self.start_points = []
        self.end_points = []
        distances = []
        self.IDs = []
        print(len(self.filtered_ids))
        IDtoPointsMAP_list = []
        tempid = 0
        for cc in range(self.num_components + 1 + len(self.filtered_ids)):
            if cc == 0 or cc in self.filtered_ids:
                #print(f'skip{cc}')
                continue

            self.IDs.append(cc)
            IDtoPointsMAP_list.append((tempid,cc))   # I need this to know which start,end,line corresponds to which connected componenet
            tempid = tempid + 1
            coords = np.where(self.labeled_volume == cc)
            component = np.zeros_like(self.labeled_volume)
            component[coords] = 1
            # Assume the volume is your 3D binary volume
            # Sum the binary volume along the z-axis to get a 2D projection
            projection = np.sum(component, axis=2)
            projection[projection > 1] = 1
            y_indices, x_indices = np.where(projection == 1)
            highest_point_index = np.argmin(y_indices)
            x_highest = x_indices[highest_point_index]
            y_highest = y_indices[highest_point_index]

            # To find the z coordinate(s) of the highest point in the 3D volume
            z_highest = np.where(component[y_highest, x_highest, :] == 1)[0]
            self.start_points.append([y_highest, x_highest, z_highest.item(0)])
            centroid = find_centroid(projection)

            # Find the bottom-most boundary point starting from the centroid
            for y in range(centroid[0], projection.shape[0]):
                if projection[y, centroid[1]] == 0:
                    bottom_y = y - 1
                    break
            # Coordinates of the 2D point you found in the previous step
            x_2d, y_2d = centroid[1], bottom_y

            for z in range(component.shape[2]):
                if np.any(component[:, :, z] != 0):  # Check if there's any label in the slice
                    break

            self.end_points.append([y_2d, x_2d, z])

            current_progress = (cc + 1) * progress_per_item
            progress_bar.setValue(int(current_progress))
            # Process GUI events to update the progress bar
            QApplication.processEvents()

        self.IDtoPointsMAP = tuple(IDtoPointsMAP_list)
        self.analysis_stage = 5
        lines = [np.vstack([start, end]) for start, end in zip(self.start_points, self.end_points)]
        if self.analysis_stage == 6:
            self.viewer.layers['Peak Points'].data = self.start_points
            self.viewer.layers['Base Points'].data = self.end_points
            self.viewer.layers['Lines'].data = lines
        else:
            self.start_points_layer = self.viewer.add_points(self.start_points, size=15, face_color='red', name='Peak Points')
            self.end_points_layer = self.viewer.add_points(self.end_points, size=15, face_color='green', name='Base Points')
            self.lines_layer = self.viewer.add_shapes(lines, shape_type='path', edge_color='cyan', edge_width=3, name='Lines')
        progress_bar.setValue(100)
        QApplication.processEvents()
        progress_dialog.close()
        # Attach event listeners to the start and end points layers
        self.start_points_layer.events.data.connect(self.update_lines)
        self.end_points_layer.events.data.connect(self.update_lines)
        self.save_attributes(self.pkl_Path)

    def predict_region(self):
        def predict_image(image_path, model, transform):
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            return predicted.item()

        def select_images_around_middle(mouse_folder, num_images='all'):
            """
            Selects a specified number of images around the middle of a stack.

            Parameters:
            - mouse_folder: Path to the folder containing images.
            - num_images: Number of images to select. Can be an odd integer or 'all'.
                          If the number is greater than the available images, returns all images.
                          If 'all', returns all images.

            Returns:
            - A list of paths to the selected images.
            """
            image_files = [os.path.join(mouse_folder, f) for f in os.listdir(mouse_folder) if f.endswith('.tif')]
            total_images = len(image_files)

            # If num_images is 'all' or larger than the total, return all images
            if num_images == 'all' or num_images >= total_images:
                return image_files

            # Ensure num_images is odd to symmetrically select around the middle
            num_images = max(1, min(num_images, total_images))  # Ensure within bounds
            if num_images % 2 == 0:
                num_images += 1  # Adjust to ensure odd number

            half_window = num_images // 2
            middle_index = total_images // 2

            start_index = max(0, middle_index - half_window)
            end_index = min(total_images, middle_index + half_window + 1)

            # Adjust the window in case of boundary issues
            if start_index == 0:
                end_index = min(num_images, total_images)
            elif end_index == total_images:
                start_index = max(0, total_images - num_images)

            median_images = image_files[start_index:end_index]

            return median_images

        def evaluate_accuracy_per_mouse(root_dir, model, transform, num_images_for_decision):

            median_images = select_images_around_middle(root_dir, num_images=num_images_for_decision)

            votes = []
            for image_path in median_images:
                prediction = predict_image(image_path, model, transform)
                votes.append(prediction)

            # Majority vote
            most_common, num_most_common = Counter(votes).most_common(1)[0]
            if most_common == 0:
                predicted_class = 'APEX'
            elif most_common == 1:
                predicted_class = 'BASE'
            elif most_common == 2:
                predicted_class = 'MIDDLE'
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Region Prediction')
            msg_box.setText(
                f"Region predicted as {predicted_class} ")
            msg_box.exec_()
            print(most_common)

        self.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the trained model checkpoint
        checkpoint = torch.load(self.model_region_prediction)
        # Load the trained model
        model = resnet50(pretrained=False)  # Initialize the model
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)  # Adjust the final layer based on the number of classes
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()  # Set the model to evaluation mode
        # Define transformations for the input image
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Evaluate accuracy per mouse
        evaluate_accuracy_per_mouse(self.full_stack_rotated_images,
                                    model, val_transform,  num_images_for_decision=13)
        self.loading_label.setText("")
        QApplication.processEvents()

    def creategt(self):
        self.loading_label.setText("<font color='red'>Process in progress..., Wait</font>")
        QApplication.processEvents()
        # Get the shape of the original layer
        original_volume_layer = self.viewer.layers['Original Volume']
        original_shape = original_volume_layer.data.shape
        layer_name = 'Ground Truth'
        if layer_name in self.viewer.layers:
            self.viewer.layers['Ground Truth'].visible = True
            self.save_attributes(self.pkl_Path)
        else:
            # Create an empty array with the same shape
            self.gt = np.zeros(original_shape, dtype=np.uint8)
            self.viewer.add_labels(self.gt, name='Ground Truth')
        self.save_attributes(self.pkl_Path)
        self.loading_label.setText("")
        QApplication.processEvents()
    def savemasks(self):

        def replace_values(mask):
            unique_labels = np.unique(mask[mask != 0])
            counter = 1
            newmask = np.zeros_like(mask)
            for item in unique_labels:
                labeled_area = mask == item
                newmask[labeled_area] = counter
                counter = counter + 1
            return (newmask)
        # filter out any component < 250
        # Make sure all values between 0 and 255
        # if there are more than one region with the same label, then the code will keep just the largest one and delete others
        self.loading_label.setText("<font color='red'>Generating GT masks in progress..., Wait</font>")
        QApplication.processEvents()
        #
        flag = 0
        # dst_folder = './'
        # base_name = os.path.splitext(self.file_path)[0]
        # self.filename_base = base_name.split('/')[-1].replace(' ', '')
        # new_folder_path = os.path.join(dst_folder, self.filename_base)
        gt_path = self.rootfolder + '/' + self.filename_base + '/Ground_Truth/'
        if not os.path.exists(gt_path):
            os.makedirs(gt_path, exist_ok=True)
        filenames = os.listdir(self.full_stack_rotated_images)
        z = np.shape(self.gt)[2]
        # Loop over all the layers
        for i in range(z):
            mask = np.array(Image.fromarray(self.gt[:,:,i]))
            unique_labels = np.unique(mask[mask != 0])
            if np.all((unique_labels >= 0) & (unique_labels <= 255)):
                filled_mask = mask.copy()
                for cc in unique_labels:
                    component_mask = mask == cc  # Boolean mask for the component
                    true_labels_count = np.sum(component_mask)
                    if true_labels_count < 500:
                        filled_mask[component_mask] = 0
                        continue
                    filled_component = binary_fill_holes(component_mask)
                    #
                    labeled_array_filled_component, num_features = label(filled_component)
                    if num_features > 1:  # Check if there are more than two separate components
                        # Calculate the size of each component
                        component_sizes = ndi_sum(filled_component, labeled_array_filled_component,
                                                  range(1, num_features + 1))
                        # Find the label of the largest component
                        largest_component_label = np.argmax(component_sizes) + 1
                        # Keep only the largest component
                        filled_component = labeled_array_filled_component == largest_component_label
                        del_area_component = (labeled_array_filled_component != largest_component_label) & (
                                    labeled_array_filled_component != 0)
                        filled_mask[del_area_component] = 0
                        filled_mask[filled_component] = cc  # Set largest component
                    else:
                    #
                        filled_pixels = component_mask != filled_component
                        filled_mask[filled_pixels] = cc

                maskpath = os.path.join(gt_path, filenames[i][:-3] + 'png')
                filled_mask = replace_values(filled_mask)
                filled_mask = Image.fromarray(filled_mask)
                filled_mask.save(maskpath)
            else:
                flag = 1

        if flag == 1:
            self.loading_label.setText("")
            QApplication.processEvents()
            msg_box = QMessageBox()
            msg_box.setWindowTitle('GT Masks Info')
            msg_box.setText(
                'Some GT annotation has lables with values more than 255 which is not acceptable')
            msg_box.exec_()
        self.loading_label.setText("")
        QApplication.processEvents()

    def copymasks(self):
        self.loading_label.setText("<font color='red'>Copying masks in progress..., Wait</font>")
        QApplication.processEvents()

        gt_path = self.rootfolder + '/' + self.filename_base + '/Ground_Truth/'
        if os.path.exists(gt_path):
            msg_box = QMessageBox()
            msg_box.setWindowTitle('GT Masks Info')
            msg_box.setText(
                'Ground Truth layer is already created, delete the Ground Truth folder if you want to restart from segmentation mask')
            msg_box.exec_()
            self.loading_label.setText("")
            QApplication.processEvents()
            return

        if self.analysis_stage < 4:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('GT Masks Info')
            msg_box.setText(
                'Sorry, you need to segment and reconstruct first')
            msg_box.exec_()
            self.loading_label.setText("")
            QApplication.processEvents()
            return

        # segmentation_path = self.rootfolder + '/' + self.filename_base + '/full_stack_rotated_images/' + 'new_assignment_obj/'
        # segimages = []
        # rawim_files = sorted(
        #     [os.path.join(segmentation_path, f) for f in os.listdir(segmentation_path) if
        #      f.endswith('.png')])  # Change '.png' if you're using a different format
        #
        # # Read each 2D mask and stack them
        # for rawim_file in rawim_files:
        #     im = imread(rawim_file)
        #     segimages.append(im)

        #segimages_3d = np.stack(segimages, axis=-1)
        self.gt = np.copy(self.labeled_volume)
        if 'Ground Truth' in self.viewer.layers:
            self.viewer.layers['Ground Truth'].data = self.gt
        else:
            self.viewer.add_labels(self.gt, name='Ground Truth')

        #self.gt = segimages_3d
        self.loading_label.setText("")
        QApplication.processEvents()

    def display_stored_gt(self):
        self.loading_label.setText("<font color='red'>Saving masks in progress..., Wait</font>")
        QApplication.processEvents()
        # dst_folder = './'
        # base_name = os.path.splitext(self.file_path)[0]
        # self.filename_base = base_name.split('/')[-1].replace(' ', '')
        # new_folder_path = os.path.join(dst_folder, self.filename_base)
        gt_path = self.rootfolder + '/' + self.filename_base + '/Ground_Truth/'
        if not os.path.exists(gt_path):
            msg_box = QMessageBox()
            msg_box.setWindowTitle('GT Masks Info')
            msg_box.setText(
                'Please annotate Ground Truth and save them, then click display')
            msg_box.exec_()
        gtimages = []
        rawim_files = sorted(
            [os.path.join(gt_path, f) for f in os.listdir(gt_path) if
             f.endswith('.png')])  # Change '.png' if you're using a different format

        # Read each 2D mask and stack them
        for rawim_file in rawim_files:
            im = imread(rawim_file)
            gtimages.append(im)

        gtimages_3d = np.stack(gtimages, axis=-1)
        if 'Stored Ground Truth' in self.viewer.layers:
            self.viewer.layers['Stored Ground Truth'].data = gtimages_3d
        else:
            self.viewer.add_labels(gtimages_3d, name='Stored Ground Truth')

        self.loading_label.setText("")
        QApplication.processEvents()

    def move_gt(self):
        dir_path = QFileDialog.getExistingDirectory()
        if not dir_path:
            return
        self.loading_label.setText("<font color='red'>Moving masks in progress..., Wait</font>")
        QApplication.processEvents()
        # dst_folder = './'
        # base_name = os.path.splitext(self.file_path)[0]
        # self.filename_base = base_name.split('/')[-1].replace(' ', '')
        # new_folder_path = os.path.join(dst_folder, self.filename_base)
        gt_path = self.rootfolder + '/' + self.filename_base + '/Ground_Truth/'
        if not os.path.exists(gt_path):
            msg_box = QMessageBox()
            msg_box.setWindowTitle('GT Masks Info')
            msg_box.setText(
                'Please annotate Ground Truth and save them, then click move buttom')
            msg_box.exec_()

        rawim_files = sorted(
            [(gt_path + '/' + f) for f in os.listdir(gt_path) if
             f.endswith('.png')])  # Change '.png' if you're using a different format

        for rawim_file in rawim_files:
            im = imread(rawim_file)
            unique_ids = np.unique(im[im!=0])
            raw_file_name = rawim_file.split('/')[-1].replace('.png','.tif')
            source_file_im = os.path.join(self.full_stack_rotated_images, raw_file_name)
            if unique_ids.size > 0:
                shutil.copy2(rawim_file, dir_path)
                shutil.copy2(source_file_im, dir_path)
        self.loading_label.setText("")
        QApplication.processEvents()

    def exit_button(self):
        while len(self.viewer.layers) > 0:
            self.viewer.layers.pop(0)
        self.viewer.window.close()

    def reset_button(self):
        self.loading_name.setText("")
        self.loading_label.setText("")
        QApplication.processEvents()
        while len(self.viewer.layers) > 0:
            self.viewer.layers.pop(0)
        self.train_iter = 50000
        self.training_path = None
        self.analysis_stage = None
        self.pkl_Path = None
        self.file_path = None  # This is file path from QFileDialog.getOpenFileName()
        self.filename_base = None  # file name
        self.full_stack_raw_images = None
        self.full_stack_length = None
        self.full_stack_raw_images_trimmed = None
        self.full_stack_rotated_images = None
        self.physical_resolution = None
        self.npy_dir = None  # This is for npy files after we run the prediction
        self.obj_dir = None  # This is for new_assignment_obj after we run the prediction, this will be used by the tracking algorithm
        self.start_trim = None  # Which slice to start trimming
        self.end_trim = None  # Which slice to end trimming
        self.display = None  # to control which frames to display, None  to display full stack, 1 for full_stack_raw_images_trimmed, And 2 for full_stack_rotated_images
        self.labeled_volume = None  # This is the labled volume that will be added through self.viewer.add_labels, and I generated it after prediction and tracking
        self.filtered_ids = None  # The objects that have been filtered when depth < 3
        self.num_components = None  # Number of compoenents in the labeled volume
        self.start_points = None
        self.end_points = None
        self.start_points_most_updated = None
        self.end_points_most_updated = None
        self.start_points_layer = None  # for distance calculationself.start_points_layer.data, self.end_points_layer.data
        self.end_points_layer = None  # for distance calculation
        self.lines_layer = None  # for distance calculation
        self.physical_distances = None
        self.IDs = None
        self.IDtoPointsMAP = None  # to map which cc ID corresponds to which starting point, ending point and line
        self.clustering = None  # to inform the sotware that the clustering is done, mainly for delete buttom
        self.clustered_cells = None
        self.IHC = None
        self.IHC_OHC = None
        self.OHC = None
        self.OHC1 = None
        self.OHC2 = None
        self.OHC3 = None
        self.gt = None



# This ensures that the following code only runs if the script is the main program
if __name__ == "__main__":
    # Instantiate the plugin
    plugin = NapariPlugin()
    napari.run()