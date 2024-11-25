import matplotlib
matplotlib.use('Qt5Agg')
from skimage.io import imread
import shutil
from scipy.ndimage import rotate
from tifffile import imwrite
import os
#-------------- Qui
from qtpy.QtWidgets import QMessageBox
from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout, QDialog, QSlider
from .VASCilia_utils import display_images, save_attributes  # Import the utility functions
from .trim_AI import Trim_AI_prediction  # Import the class from trim_AI.py
from .rotate_AI import Rotate_AI_prediction
from .segment_cochlea_action import SegmentCochleaAction
from .visualize_track_action import VisualizeTrackAction
from .calculate_measurements import CalculateMeasurementsAction
from .calculate_distance import CalculateDistanceAction
from .save_distance import SaveDistanceAction
from .identify_celltype_action import CellClusteringAction
from .compute_signal_action import ComputeSignalAction
from .predict_tonotopic_region import PredictRegionAction
from .compute_orientation_action import ComputeOrientationAction
from .commute_training_action import commutetraining
from .reset_exit_action import reset_exit
from .visualize_track_SORT import VisualizeTrackActionSORT
from .prepare_length_json import Prepare_length_json_Action

class BatchCochleaAction:
    """
    This class handles the action of batch processing.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def execute(self):


        if self.plugin.analysis_stage >= 2:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'The stack is already partially of fully analyzed, if you want to run batch analysis, you need to press the button before Trimming action ')
            msg_box.exec_()
            return

        def copy_files_in_range():
            total_digits = 4  # Assuming 4 digits for file numbering

            # Ensure destination path exists
            if not os.path.exists(self.plugin.full_stack_raw_images_trimmed):
                os.makedirs(self.plugin.full_stack_raw_images_trimmed)
            else:
                shutil.rmtree(self.plugin.full_stack_raw_images_trimmed)
                os.makedirs(self.plugin.full_stack_raw_images_trimmed)

            # Iterate over the range of file numbers
            for i in range(self.plugin.start_trim, self.plugin.end_trim + 1):
                padded_num = str(i).zfill(4)
                file_name_to_copy = f'{self.plugin.filename_base}_{padded_num}.tif'
                source_file = os.path.join(self.plugin.full_stack_raw_images, file_name_to_copy)
                dest_file = os.path.join(self.plugin.full_stack_raw_images_trimmed, file_name_to_copy)

                # Copy file if it exists
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
                else:
                    print(f"File not found: {source_file}")

        def process_input(start_no, end_no):

            self.plugin.start_trim = int(start_no)
            self.plugin.end_trim = int(end_no)

            if self.plugin.start_trim == self.plugin.end_trim:
                self.plugin.end_trim = self.plugin.end_trim + 1


            if self.plugin.start_trim < self.plugin.end_trim <= self.plugin.full_stack_length:
                self.plugin.full_stack_raw_images_trimmed = os.path.dirname(self.plugin.full_stack_raw_images)
                self.plugin.full_stack_raw_images_trimmed = os.path.join(self.plugin.full_stack_raw_images_trimmed, 'full_stack_raw_images_trimmed')
                if not os.path.exists(self.plugin.full_stack_raw_images_trimmed):
                    os.makedirs(self.plugin.full_stack_raw_images_trimmed, exist_ok=True)
                copy_files_in_range()
                self.plugin.display = 1
                self.plugin.analysis_stage = 2
                save_attributes(self.plugin, self.plugin.pkl_Path)
                display_images(self.plugin.viewer, self.plugin.display, self.plugin.full_stack_raw_images,
                                   self.plugin.full_stack_raw_images_trimmed, self.plugin.full_stack_rotated_images,
                                   self.plugin.filename_base, self.plugin.loading_name)


        ### AI sugggestion for start and end index
        self.plugin.loading_label.setText("<font color='red'>Trim Processing..., Wait</font>")
        QApplication.processEvents()
        trim_ai = Trim_AI_prediction(self.plugin)
        start_index, end_index = trim_ai.execute()
        # Connect button signals
        if start_index == 'None' or end_index == 'None':
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'Trimming AI did not find any valid frame to process')
            msg_box.exec_()
            return
        process_input(start_index, end_index)

        ## ------ Rotate the stack is started
        self.plugin.loading_label.setText("<font color='red'>Rotate Processing..., Wait</font>")
        QApplication.processEvents()
        image_files = [f for f in sorted(os.listdir(self.plugin.full_stack_raw_images_trimmed)) if f.endswith('.tif')]
        image_files.sort()
        middle_index = len(image_files) // 2
        middle_image_name = image_files[middle_index] if image_files else None
        middle_image = imread(os.path.join(self.plugin.full_stack_raw_images_trimmed, middle_image_name))
        # This code is related to the rotate_AI proediction------------
        rotate_ai = Rotate_AI_prediction(self.plugin)
        angle_to_rotate_ai = rotate_ai.execute()

        if not os.path.exists(self.plugin.full_stack_rotated_images):
            os.makedirs(self.plugin.full_stack_rotated_images)
        else:
            shutil.rmtree(self.plugin.full_stack_rotated_images)
            os.makedirs(self.plugin.full_stack_rotated_images)
        angle = angle_to_rotate_ai

        rawim_files = sorted(
            [os.path.join(self.plugin.full_stack_raw_images_trimmed, f) for f in
             os.listdir(self.plugin.full_stack_raw_images_trimmed) if
             f.endswith('.tif')])

        for rawim_file in rawim_files:
            im = imread(rawim_file)
            im = rotate(im, angle, reshape=True)
            imwrite(os.path.join(self.plugin.full_stack_rotated_images, os.path.basename(rawim_file)), im)

        file_path_angle = os.path.dirname(self.plugin.full_stack_rotated_images)
        with open(os.path.join(file_path_angle, 'angle.txt'), 'w') as file:
            file.write(str(angle))
        self.plugin.display = 2
        self.plugin.analysis_stage = 3
        save_attributes(self.plugin, self.plugin.pkl_Path)
        display_images(self.plugin.viewer, self.plugin.display, self.plugin.full_stack_raw_images,
                       self.plugin.full_stack_raw_images_trimmed, self.plugin.full_stack_rotated_images,
                       self.plugin.filename_base, self.plugin.loading_name)
        self.plugin.loading_label.setText("")
        QApplication.processEvents()

        # Visualization and Tracking
        self.plugin.loading_label.setText("<font color='red'>Segmentation Processing..., Wait</font>")
        QApplication.processEvents()
        SegmentCochleaAction(self.plugin).execute()
        #VisualizeTrackAction(self.plugin).execute()
        VisualizeTrackActionSORT(self.plugin).execute()

        # Calculate measurments and save the 3D label
        CalculateMeasurementsAction(self.plugin).execute()

        # Calculate the length from tip to bottom
        self.plugin.distance_action.execute()
        SaveDistanceAction(self.plugin).execute()

        #  please del or uncomment later, this is to collect the data for key point detection
        #Prepare_length_json_Action(self.plugin).execute()

        # Identify cell types
        method = 'Deep Learning'
        CellClusteringAction(self.plugin).perform_clustering(method)
        CellClusteringAction(self.plugin).find_IHC_OHC()

        # Compute protein intensity
        ComputeSignalAction(self.plugin).compute_protein_intensity()

        # Compute orientation
        method = 'Height_only'
        ComputeOrientationAction(self.plugin).calculate_orientation(method)
