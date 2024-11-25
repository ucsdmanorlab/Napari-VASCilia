import matplotlib
matplotlib.use('Qt5Agg')
from skimage.io import imread
import shutil
from scipy.ndimage import rotate
from tifffile import imwrite
import os
import numpy as np
#-------------- Qui
from qtpy.QtWidgets import QMessageBox
from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout, QDialog, QSlider
from qtpy.QtWidgets import  QLabel
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtCore import Qt, QSize
from qtpy.QtWidgets import QPushButton, QVBoxLayout
from .VASCilia_utils import display_images, save_attributes  # Import the utility functions
from . rotate_AI import Rotate_AI_prediction

class RotateCochleaAction:
    """
    This class handles the action of rotating Cochlea stacks.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the RotateCochleaAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin
        self.popup = None  # Keep a reference to the dialog

    def execute(self):
        """
        Executes the action to rotate Cochlea stacks.
        It prompts the user for the rotation angle and processes the files accordingly.
        """
        if self.plugin.analysis_stage >= 3:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText(
                'The stack is already rotated, if you want to change the rotating decision, delete the current folder and restart the analysis')
            msg_box.exec_()
            return

        image_files = [f for f in sorted(os.listdir(self.plugin.full_stack_raw_images_trimmed)) if f.endswith('.tif')]
        image_files.sort()
        middle_index = len(image_files) // 2
        middle_image_name = image_files[middle_index] if image_files else None
        middle_image = imread(os.path.join(self.plugin.full_stack_raw_images_trimmed, middle_image_name))
        # This code is related to the rotate_AI proediction------------
        rotate_ai = Rotate_AI_prediction(self.plugin)
        angle_to_rotate_ai = rotate_ai.execute()
        middle_image = rotate(middle_image, angle_to_rotate_ai, reshape=True)
        ##-------------------------------------------------------------
        self.popup = QDialog()  # Changed from local variable to class attribute
        layout = QVBoxLayout(self.popup)

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

        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        pixmap = numpy_array_to_qpixmap(middle_image)
        desired_size = QSize(300, 300)
        pixmap = pixmap.scaled(desired_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        layout.addWidget(label)

        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.setMinimum(-180)
        slider.setMaximum(180)
        emoji_label = QLabel("🤖 🧠 🤖 AI Companion Rotation 🤖 🧠 🤖")  # Emojis as text
        layout.addWidget(emoji_label, alignment=Qt.AlignHCenter)
        layout.addWidget(slider)

        def update_image(value):
            rotated = rotate(middle_image, value, reshape=False)
            pixmap = numpy_array_to_qpixmap(rotated)
            pixmap = pixmap.scaled(desired_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)

        slider.valueChanged.connect(update_image)

        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        layout.addWidget(ok_button)
        layout.addWidget(cancel_button)

        def on_ok():
            self.plugin.loading_label.setText("<font color='red'>Processing..., Wait</font>")
            QApplication.processEvents()
            if not os.path.exists(self.plugin.full_stack_rotated_images):
                os.makedirs(self.plugin.full_stack_rotated_images)
            else:
                shutil.rmtree(self.plugin.full_stack_rotated_images)
                os.makedirs(self.plugin.full_stack_rotated_images)
            angle = slider.value() + angle_to_rotate_ai
            # This line is also related to AI rotation model
            # if angle == 0:
            #     angle = angle_to_rotate_ai
            rawim_files = sorted(
                [os.path.join(self.plugin.full_stack_raw_images_trimmed, f) for f in os.listdir(self.plugin.full_stack_raw_images_trimmed) if
                 f.endswith('.tif')])

            for rawim_file in rawim_files:
                im = imread(rawim_file)
                im = rotate(im, angle, reshape=True)
                imwrite(os.path.join(self.plugin.full_stack_rotated_images, os.path.basename(rawim_file)), im)
            self.popup.close()
            file_path_angle = os.path.dirname(self.plugin.full_stack_rotated_images)
            with open(os.path.join(file_path_angle, 'angle.txt'), 'w') as file:
                file.write(str(angle))
            self.plugin.rot_angle = angle
            self.plugin.display = 2
            self.plugin.analysis_stage = 3
            save_attributes(self.plugin, self.plugin.pkl_Path)
            #self.plugin.display_images()
            display_images(self.plugin.viewer, self.plugin.display, self.plugin.full_stack_raw_images,
                           self.plugin.full_stack_raw_images_trimmed, self.plugin.full_stack_rotated_images,
                           self.plugin.filename_base, self.plugin.loading_name)
            self.plugin.loading_label.setText("")
            QApplication.processEvents()

        ok_button.clicked.connect(on_ok)
        cancel_button.clicked.connect(self.popup.close)
        #self.popup.show()  # Ensure the dialog is shown
        self.popup.exec_()