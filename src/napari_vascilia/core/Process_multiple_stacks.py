import matplotlib
matplotlib.use('Qt5Agg')
import os
import pandas as pd
import time
#-------------- Qui
from .open_cochlea_action import OpenCochleaAction
from .reset_exit_action import reset_exit
from .batch_action import BatchCochleaAction
from qtpy.QtWidgets import QMessageBox


class BatchCochleaAction_multi_stacks:
    """
    This class handles the action of processing multiple stacks.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the batch processing action with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def execute(self):
        """
        Executes the action of processing multiple stacks.
        """
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Analysis Details')
        msg_box.setText(
            'Are you sure you want to do multi-stack processing? Please make sure that you have already set up the file_names_for_batch_processing.csv file. If so, click OK to continue.'
        )

        # Add OK and Cancel buttons
        msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        # Execute the message box and capture the response
        response = msg_box.exec_()

        if response == QMessageBox.Ok:

            #reset_exit(self.plugin).reset_button()
            # Read the Excel file
            file_path_for_batch_processing = os.path.join(self.plugin.rootfolder, 'file_names_for_batch_processing.csv')
            df = pd.read_csv(file_path_for_batch_processing) #, header=None
            start_time = time.time()
            # Open a log file to write invalid paths
            with open('invalid_paths_batch_processing', 'w') as log_file:
                # Loop through the first column
                for index, file_path in df.iloc[:, 0].items():
                    # Check if the path is a string and ends with .czi, .lif, or .tif
                    if isinstance(file_path, str) and file_path.lower().endswith(('.czi', '.lif', '.tif')):
                        # Check if the file exists
                        if os.path.exists(file_path):
                            file_path = file_path.strip().replace("\\", "/")
                            base_name = os.path.splitext(file_path)[0]
                            filename_base = base_name.split('/')[-1].replace(' ', '')[:45]
                            filename_base = filename_base.replace('(', '')
                            filename_base = filename_base.replace(')', '')
                            new_folder_path = os.path.join(self.plugin.rootfolder, filename_base)

                            if not os.path.exists(new_folder_path):
                                OpenCochleaAction(self.plugin, batch = 1, batch_file_path = file_path).execute()
                                BatchCochleaAction(self.plugin).execute()
                            time.sleep(2)
                            reset_exit(self.plugin).reset_button()
                            self.plugin.viewer.layers.clear()

                        else:
                            log_file.write(f"Invalid path: {file_path} (File does not exist)\n")
                    else:
                        log_file.write(f"Invalid path: {file_path} (Incorrect file extension or format)\n")

            end_time = time.time()
            time_elapsed = end_time - start_time
            print(f'Elapsed time doing the analysis equal {time_elapsed}')
            #reset_exit(self.plugin).exit_button()
