import matplotlib
matplotlib.use('Qt5Agg')
import shutil
import os
#-------------- Qui
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QDialog
from qtpy.QtWidgets import  QLabel, QLineEdit, QHBoxLayout
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from .VASCilia_utils import display_images, save_attributes  # Import the utility functions

class TrimCochleaAction:
    """
    This class handles the action of trimming Cochlea stacks.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the TrimCochleaAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def execute(self):
        """
        Executes the action to trim Cochlea stacks.
        It prompts the user for the range of stacks to trim and processes the files accordingly.
        """
        if self.plugin.analysis_stage >= 2:
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
            try:
                self.plugin.start_trim = int(start_no)
                self.plugin.end_trim = int(end_no)
                if self.plugin.start_trim < self.plugin.end_trim <= self.plugin.full_stack_length:
                    self.plugin.full_stack_raw_images_trimmed = os.path.dirname(self.plugin.full_stack_raw_images)
                    self.plugin.full_stack_raw_images_trimmed = os.path.join(self.plugin.full_stack_raw_images_trimmed, 'full_stack_raw_images_trimmed')
                    if not os.path.exists(self.plugin.full_stack_raw_images_trimmed):
                        os.makedirs(self.plugin.full_stack_raw_images_trimmed, exist_ok=True)
                    copy_files_in_range()
                    self.plugin.display = 1
                    self.plugin.analysis_stage = 2
                    save_attributes(self.plugin, self.plugin.pkl_Path)
                    #self.plugin.display_images()
                    display_images(self.plugin.viewer, self.plugin.display, self.plugin.full_stack_raw_images,
                                   self.plugin.full_stack_raw_images_trimmed, self.plugin.full_stack_rotated_images,
                                   self.plugin.filename_base, self.plugin.loading_name)

            except ValueError as e:
                print(f"Input Error: {e}")
            dialog.close()

        # Connect button signals
        ok_button.clicked.connect(lambda: process_input(start_input.text(), end_input.text()))
        cancel_button.clicked.connect(dialog.close)
        dialog.exec_()