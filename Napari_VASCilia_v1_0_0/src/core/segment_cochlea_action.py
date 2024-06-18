import os
import re
import subprocess
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QMessageBox, QProgressBar, QApplication, QDesktopWidget
from PyQt5.QtCore import QTimer, Qt

class SegmentCochleaAction:
    """
    This class handles the action of segmenting Cochlea stacks.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the SegmentCochleaAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def execute(self):
        """
        Executes the action to segment Cochlea stacks.
        It runs a segmentation command and shows a progress dialog.
        """
        if self.plugin.analysis_stage < 3:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText('Please press "Rotate" button')
            msg_box.exec_()
            return
        if self.plugin.analysis_stage >= 4:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Analysis Details')
            msg_box.setText('The stack is already segmented')
            msg_box.exec_()
            return

        currentfolder = os.path.join(self.plugin.rootfolder, self.plugin.full_stack_rotated_images.strip('./'))
        currentfolder = currentfolder.replace(':', '').replace('\\', '/')
        currentfolder = '/mnt/' + currentfolder.lower()
        currentfolder = os.path.dirname(currentfolder) + '/'

        trainfolder = os.path.join(self.plugin.rootfolder)
        trainfolder = trainfolder.replace(':', '').replace('\\', '/')
        trainfolder = '/mnt/' + trainfolder.lower()
        trainfolder = os.path.dirname(trainfolder) + '/'

        output_model_path = self.plugin.model_output_path
        output_model_path = output_model_path.replace(':', '').replace('\\', '/')
        output_model_path = '/mnt/' + output_model_path.lower()
        output_model_path = os.path.dirname(output_model_path) + '/'

        command = f'wsl {self.plugin.wsl_executable} --train_predict 1 --folder_path {trainfolder} --model_output_path {output_model_path} --iterations {self.plugin.train_iter} --rootfolder {currentfolder} --model {self.plugin.model} --threshold 0.7'

        progress_dialog = QDialog()
        progress_dialog.setModal(True)
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
        progress_bar.setMaximum(100)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   bufsize=1, universal_newlines=True)

        def update_progress_bar():
            line = process.stdout.readline()
            if line:
                print(line)
                match = re.search(r"##(\d+(?:\.\d+)?)%", line)
                if match:
                    progress = float(match.group(1))
                    progress_bar.setValue(int(round(progress)))
                    QApplication.processEvents()
            else:
                if process.poll() is not None:
                    timer.stop()
                    process.stdout.close()
                    process.stderr.close()
                    progress_dialog.close()

        timer = QTimer()
        timer.timeout.connect(update_progress_bar)
        timer.start(150)

        progress_dialog.exec_()
