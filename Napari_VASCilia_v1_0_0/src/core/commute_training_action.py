import matplotlib
matplotlib.use('Qt5Agg')
from skimage.measure import label
from skimage.io import imread
from scipy.ndimage import label, binary_fill_holes, sum as ndi_sum
from scipy.ndimage import binary_dilation, generate_binary_structure
import shutil
import subprocess
from qtpy.QtCore import QTimer
import time
import os
from PIL import Image
import numpy as np
#-------------- Qui
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QApplication, QProgressBar, QVBoxLayout, QDesktopWidget, QDialog
from qtpy.QtWidgets import  QFileDialog
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import QVBoxLayout
from .VASCilia_utils import save_attributes  # Import the utility functions


class commutetraining:
    """
    This class handles the action of creating GT, save and tran them.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the commute training with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def creategt(self):
        self.plugin.loading_label.setText("<font color='red'>Process in progress..., Wait</font>")
        QApplication.processEvents()
        # Get the shape of the original layer
        original_volume_layer = self.plugin.viewer.layers['Original Volume']
        original_shape = original_volume_layer.data.shape
        layer_name = 'Ground Truth'
        if layer_name in self.plugin.viewer.layers:
            self.plugin.viewer.layers['Ground Truth'].visible = True
            save_attributes(self.plugin, self.plugin.pkl_Path)
        else:
            # Create an empty array with the same shape
            self.plugin.gt = np.zeros(original_shape, dtype=np.uint8)
            self.plugin.viewer.add_labels(self.plugin.gt, name='Ground Truth')
        save_attributes(self.plugin, self.plugin.pkl_Path)
        self.plugin.loading_label.setText("")
        QApplication.processEvents()


    def savemasks(self):

        def set_touching_boundaries_to_zero(labeled_mask):
            # Generate a binary structure for dilation
            labeled_mask = np.array(labeled_mask)
            # Generate a binary structure for dilation.
            structure = generate_binary_structure(labeled_mask.ndim,2)  # Adjust 'ndim' to match your mask's dimensionality
            # Dilate the entire labeled mask. This expands all labels.
            dilated_mask = binary_dilation(labeled_mask > 0, structure=structure)
            # Create an empty mask to store the locations where dilated labels touch other labels.
            touching_boundaries_global = np.zeros_like(labeled_mask, dtype=bool)
            # Iterate through each unique label in the labeled mask.
            for label in np.unique(labeled_mask):
                if label == 0: continue  # Skip the background
                # Create a mask for the current label.
                label_mask = (labeled_mask == label)
                # Dilate the current label's mask.
                dilated_label_mask = binary_dilation(label_mask, structure=structure)
                # Find where the dilated mask of the current label intersects with other labels.
                touching_areas = dilated_label_mask & (dilated_mask & ~label_mask)
                # Update the global touching boundaries mask.
                touching_boundaries_global |= touching_areas
            # Apply the touching boundaries mask to set touching pixels to zero
            labeled_mask[touching_boundaries_global] = 0
            return Image.fromarray(labeled_mask)

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
        self.plugin.loading_label.setText("<font color='red'>Generating GT masks in progress..., Wait</font>")
        QApplication.processEvents()
        #
        flag = 0
        gt_path = self.plugin.rootfolder + '/' + self.plugin.filename_base + '/Ground_Truth/'
        if not os.path.exists(gt_path):
            os.makedirs(gt_path, exist_ok=True)
        filenames = os.listdir(self.plugin.full_stack_rotated_images)
        z = np.shape(self.plugin.gt)[2]
        # Loop over all the layers
        for i in range(z):
            mask = np.array(Image.fromarray(self.plugin.gt[:,:,i]))
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
                ##--- check touching boundaries-------------
                touching_components_mask = set_touching_boundaries_to_zero(filled_mask)
                ##------------------------------------------
                touching_components_mask.save(maskpath)
            else:
                flag = 1

        if flag == 1:
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            msg_box = QMessageBox()
            msg_box.setWindowTitle('GT Masks Info')
            msg_box.setText(
                'Some GT annotation has lables with values more than 255 which is not acceptable')
            msg_box.exec_()
        self.plugin.loading_label.setText("")
        QApplication.processEvents()


    def display_stored_gt(self):
        self.plugin.loading_label.setText("<font color='red'>Saving masks in progress..., Wait</font>")
        QApplication.processEvents()
        gt_path = self.plugin.rootfolder + '/' + self.plugin.filename_base + '/Ground_Truth/'
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
        if 'Stored Ground Truth' in self.plugin.viewer.layers:
            self.plugin.viewer.layers['Stored Ground Truth'].data = gtimages_3d
        else:
            self.plugin.viewer.add_labels(gtimages_3d, name='Stored Ground Truth')

        self.plugin.loading_label.setText("")
        QApplication.processEvents()

    def copymasks(self):
        self.plugin.loading_label.setText("<font color='red'>Copying masks in progress..., Wait</font>")
        QApplication.processEvents()

        gt_path = self.plugin.rootfolder + '/' + self.plugin.filename_base + '/Ground_Truth/'
        if os.path.exists(gt_path):
            msg_box = QMessageBox()
            msg_box.setWindowTitle('GT Masks Info')
            msg_box.setText(
                'Ground Truth layer is already created, delete the Ground Truth folder if you want to restart from segmentation mask')
            msg_box.exec_()
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return

        if self.plugin.analysis_stage < 4:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('GT Masks Info')
            msg_box.setText(
                'Sorry, you need to segment and reconstruct first')
            msg_box.exec_()
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return

        self.plugin.gt = np.copy(self.plugin.labeled_volume)
        if 'Ground Truth' in self.plugin.viewer.layers:
            self.plugin.viewer.layers['Ground Truth'].data = self.plugin.gt
        else:
            self.plugin.viewer.add_labels(self.plugin.gt, name='Ground Truth')

        self.plugin.loading_label.setText("")
        QApplication.processEvents()

    def move_gt(self):
        dir_path = QFileDialog.getExistingDirectory()
        if not dir_path:
            return
        self.plugin.loading_label.setText("<font color='red'>Moving masks in progress..., Wait</font>")
        QApplication.processEvents()
        gt_path = self.plugin.rootfolder + '/' + self.plugin.filename_base + '/Ground_Truth/'
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
            source_file_im = os.path.join(self.plugin.full_stack_rotated_images, raw_file_name)
            if unique_ids.size > 0:
                shutil.copy2(rawim_file, dir_path)
                shutil.copy2(source_file_im, dir_path)
        self.plugin.loading_label.setText("")
        QApplication.processEvents()


    def check_training_data(self):
        self.plugin.loading_label.setText("<font color='red'>Processing..., Wait</font>")
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
        if not os.path.exists(self.plugin.model_output_path):
            os.makedirs(self.plugin.model_output_path)
        else:
            dirfiles = os.listdir(self.plugin.model_output_path)
            if dirfiles != []:
                QMessageBox.warning(None, 'Output Model Error',
                                            'Output Model folder needs to be empty')
                self.plugin.loading_label.setText("")
                QApplication.processEvents()
                return

        dir_path = QFileDialog.getExistingDirectory()
        check = folder_content_check(dir_path)
        if check == 0:
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return
        # Check if tif files are the same number as png files
        tif_files = [filename for filename in os.listdir(os.path.join(dir_path,'Train')) if filename.endswith('.tif')]
        png_files = [filename for filename in os.listdir(os.path.join(dir_path,'Train')) if filename.endswith('.png')]
        if len(tif_files) != len(png_files):
            QMessageBox.warning(None, 'Files Issues',
                                'Train Tif files and corresponding masks are not equal')
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return
        # Check if each png file has only ID's from 1 to n, and also check if each png file has the same name as .tif file
        set1 = set([filename[:-3] for filename in tif_files])
        set2 = set([filename[:-3] for filename in png_files])
        if set1 != set2:
            QMessageBox.warning(None, 'Files Issues',
                                    'Train File names are not the same, each .tif file should have the same name as .png file')
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return
        Train_mask_check = check_masks(os.path.join(dir_path, 'Train'), png_files)
        if Train_mask_check == 0:
            QMessageBox.warning(None, 'Files Issues',
                                'There is at least one mask in Train that does not have labels')
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return
        # Val
        tif_files = [filename for filename in os.listdir(os.path.join(dir_path, 'Val')) if filename.endswith('.tif')]
        png_files = [filename for filename in os.listdir(os.path.join(dir_path, 'Val')) if filename.endswith('.png')]
        if len(tif_files) != len(png_files):
            QMessageBox.warning(None, 'Files Issues',
                                'Val Tif files and corresponding masks are not equal')
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return
        # Check if each png file has only ID's from 1 to n, and also check if each png file has the same name as .tif file
        set1 = set([filename[:-3] for filename in tif_files])
        set2 = set([filename[:-3] for filename in png_files])
        if set1 != set2:
            QMessageBox.warning(None, 'Files Issues',
                                'Val File names are not the same, each .tif file should have the same name as .png file')
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return
        Test_mask_check = check_masks(os.path.join(dir_path, 'Val'), png_files)
        if Test_mask_check == 0:
            QMessageBox.warning(None, 'Files Issues',
                                'There is at least one mask in Val that does not have labels')
            self.plugin.loading_label.setText("")
            QApplication.processEvents()
            return

        self.plugin.training_path = dir_path + '/'
        QMessageBox.warning(None, 'Files Check Complete',
                            'Congratulations: Click train buttom')
        self.plugin.loading_label.setText("")
        QApplication.processEvents()


    def train_cilia(self):
        if self.plugin.training_path == None:
            QMessageBox.warning(None, 'Check Training Data',
                                'Click first "Check Training Data" buttom')
            return

        currentfolder = os.path.join(self.plugin.training_path)  # this path will not be considered here because it is segmentation task but we need to have it
        currentfolder = currentfolder.replace(':', '').replace('\\', '/')
        currentfolder = '/mnt/' + currentfolder.lower()
        currentfolder = os.path.dirname(currentfolder) + '/'
        # folder_path, path to training data
        trainfolder = os.path.join(self.plugin.training_path)
        trainfolder = trainfolder.replace(':', '').replace('\\', '/')
        trainfolder = '/mnt/' + trainfolder.lower()
        trainfolder = os.path.dirname(trainfolder) + '/'
        #
        output_model_path = self.plugin.model_output_path
        output_model_path = output_model_path.replace(':', '').replace('\\', '/')
        output_model_path = '/mnt/' + output_model_path.lower()
        output_model_path = os.path.dirname(output_model_path) + '/'

        command = f'wsl {self.plugin.wsl_executable} --train_predict {0} --folder_path {trainfolder}  --model_output_path {output_model_path} --iterations {self.plugin.train_iter} --rootfolder {currentfolder} --model {self.plugin.model}  --threshold {0.7}'

        # Configuring the time
        total_iterations = self.plugin.train_iter
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
