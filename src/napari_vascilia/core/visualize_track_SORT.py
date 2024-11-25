import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import imageio
from skimage.io import imread
from scipy.ndimage import find_objects
import numpy as np
import os
# from .sort import Sort  # Import the SORT tracking class
# from torchvision.ops import nms  # Import NMS from torchvision
import subprocess
#-------------- Qui
from qtpy.QtWidgets import QApplication
from .VASCilia_utils import save_attributes  # Import the utility functions

class VisualizeTrackActionSORT:
    """
    This class handles the action of visualizing and tracking Cochlea segmentations.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the VisualizeTrackAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def execute(self):
        """
        Executes the action to visualize and track Cochlea segmentations.
        """

        def run_sort_executable(npy_dir, obj_dir, nms_threshold=0.4):
            """
            Run the SORT tracking executable with the provided input and output paths and NMS threshold.

            Args:
                npy_dir (str): Path to the folder containing .npy files (detections).
                obj_dir (str): Path to the folder where output masks will be stored.
                nms_threshold (float): The IoU threshold for Non-Maximum Suppression.
            """

            # Ensure the output directory exists
            if not os.path.exists(obj_dir):
                os.makedirs(obj_dir)

            # Path to your executable
            executable_path = os.path.abspath("./src/napari_vascilia/core/track_me_SORT_v3_exe.exe")
            print(executable_path)
            # Create the command to call the executable
            command = [
                executable_path,
                npy_dir,
                obj_dir,
                str(nms_threshold)  # Convert threshold to a string
            ]

            try:
                # Run the executable with arguments
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Print the output of the executable
                print("Executable output:", result.stdout.decode())
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running executable: {e.stderr.decode()}")


        self.plugin.loading_label.setText("<font color='red'>Track Processing..., Wait</font>")
        QApplication.processEvents()

        self.plugin.npy_dir = os.path.dirname(self.plugin.full_stack_rotated_images) + '/prediction/'
        self.plugin.obj_dir = os.path.dirname(self.plugin.full_stack_rotated_images) + '/new_assignment_obj/'

        # Run the function with the specified paths
        run_sort_executable(self.plugin.npy_dir, self.plugin.obj_dir, nms_threshold=0.5)
        plt.close('all')

        # npy_files = [f for f in os.listdir(self.plugin.npy_dir) if f.endswith('.npy')]
        # if not os.path.exists(self.plugin.obj_dir):
        #     os.makedirs(self.plugin.obj_dir)
        #
        # # Initialize SORT tracker
        # tracker = Sort()
        #
        # # Get list of detection files
        # for im_file in npy_files:
        #     # Load detections for the current frame
        #     file_path = os.path.join(self.plugin.npy_dir, os.path.basename(im_file))
        #     data = np.load(file_path, allow_pickle=True).item()
        #
        #     # Extract the bounding boxes, scores, and masks
        #     boxes = data['boxes']
        #     scores = data['scores']
        #     masks = data['masks']
        #
        #     # Convert to tensors and apply NMS as previously
        #     boxes_tensor = torch.from_numpy(boxes)
        #     scores_tensor = torch.from_numpy(scores)
        #
        #     iou_threshold = 0.4
        #     indices = nms(boxes_tensor, scores_tensor, iou_threshold)
        #
        #     # Convert indices to a set for faster membership checking
        #     kept_indices = set(indices.cpu().numpy())
        #     # Now filter out the masks that correspond to boxes that were not selected by NMS
        #     filtered_masks = [mask for i, mask in enumerate(masks) if i in kept_indices]
        #     # Convert filtered_masks to a numpy array if necessary
        #     masks = np.array(filtered_masks)
        #
        #     boxes = boxes_tensor[indices].cpu().numpy()
        #     scores = scores_tensor[indices].cpu().numpy()
        #
        #     # Without NMS, we use all detections directly
        #     # Assign track IDs
        #     output_mask = np.zeros_like(masks[0], dtype=np.uint8)
        #
        #     # Update tracker with current detections
        #     tracks = tracker.update(np.hstack((boxes, scores[:, np.newaxis])))
        #
        #     for track in tracks:
        #         track_id = int(track[4])  # Track ID
        #         x1, y1, x2, y2 = track[0], track[1], track[2], track[3]  # Track bounding box coordinates
        #         target_box = np.array([x1, y1, x2, y2])
        #
        #         # matches = np.all(np.floor(boxes) == np.floor(target_box), axis = 1)
        #         distances = np.sum(np.abs(boxes - target_box), axis=1)
        #         best_index = np.argmin(distances)
        #         mask = np.squeeze(masks[best_index, :, :])
        #         output_mask[mask] = track_id
        #
        #     file_name = os.path.join(self.plugin.obj_dir, os.path.basename(im_file).replace('.npy', '.png'))
        #     file_name = os.path.normpath(file_name)
        #     if len(file_name) > 255:
        #         raise ValueError(f"Path length exceeds the limit: {file_name}")
        #     imageio.imwrite(file_name, output_mask.astype(np.uint8))

        masks = []
        newmask_files = sorted([os.path.join(self.plugin.obj_dir, f) for f in os.listdir(self.plugin.obj_dir) if f.endswith('.png')])
        for maskfile in newmask_files:
            mask = imread(maskfile)
            masks.append(mask)

        self.plugin.labeled_volume = np.stack(masks, axis=-1)
        self.plugin.filtered_ids = []
        regions = find_objects(self.plugin.labeled_volume)

        # This code to filter out lables that has only one segmented region
        for i in range(1, len(regions) + 1):
            region_slice = regions[i - 1]
            depth = region_slice[2].stop - region_slice[2].start
            if depth <= 1:
                self.plugin.filtered_ids.append(i)
                self.plugin.labeled_volume[self.plugin.labeled_volume == i] = 0

        # This code is to filter out the components that are near the boundaries
        # from skimage.transform import rotate
        # rotated_labeled_volume = rotate(self.plugin.labeled_volume, -self.plugin.rot_angle, order=0, preserve_range=True).astype(int)
        # regions = find_objects(rotated_labeled_volume)
        # for i in range(1, len(regions) + 1):
        #     region_slice = regions[i - 1]
        #     # Check if the region touches any edge of the rotated volume
        #     if (
        #             region_slice[0].start == 0 or region_slice[0].stop == rotated_labeled_volume.shape[
        #         0] or  # Top or bottom edges
        #             region_slice[1].start == 0 or region_slice[1].stop == rotated_labeled_volume.shape[
        #         1]
        #     ):
        #         # Mark for deletion in the original labeled volume
        #         self.plugin.filtered_ids.append(i)
        #         self.plugin.labeled_volume[self.plugin.labeled_volume == i] = 0


        self.plugin.num_components = len(np.unique(self.plugin.labeled_volume)) - 1

        self.plugin.viewer.add_labels(self.plugin.labeled_volume, name='Labeled Image')
        props = regionprops(self.plugin.labeled_volume)
        self.plugin.id_list_annotation = []
        self.plugin.ID_positions_annotation = []
        for prop in props:
            self.plugin.id_list_annotation.append(prop.label)
            # self.plugin.ID_positions_annotation.append(prop.centroid[:2])
            coords = np.where(self.plugin.labeled_volume == prop.label)
            component = np.zeros_like(self.plugin.labeled_volume)
            component[coords] = 1
            projection = np.sum(component, axis=2)
            projection[projection > 1] = 1
            y_indices, x_indices = np.where(projection == 1)
            highest_point_index = np.argmin(y_indices)
            x_highest = x_indices[highest_point_index]
            y_highest = y_indices[highest_point_index]
            self.plugin.ID_positions_annotation.append((y_highest, x_highest))
        self.plugin.ID_positions_annotation = [np.append(pos, 0) for pos in self.plugin.ID_positions_annotation]
        self.plugin.ID_positions_annotation = [pos - [10, 0, 0] for pos in self.plugin.ID_positions_annotation]
        self.plugin.ID_positions_annotation = np.array(self.plugin.ID_positions_annotation).astype(int)
        if 'ID Annotations' in self.plugin.viewer.layers:
            self.plugin.viewer.layers['ID Annotations'].data = self.plugin.ID_positions_annotation
        else:
            self.plugin.ID_layer = self.plugin.viewer.add_points(self.plugin.ID_positions_annotation, size=0.05,
                                                                 face_color='transparent',
                                                                 name='ID Annotations')

        self.plugin.ID_layer.text = self.plugin.id_list_annotation
        self.plugin.ID_layer.text.color = 'lime'
        self.plugin.ID_layer.text.size = 12
        self.plugin.viewer.dims.ndisplay = 3
        self.plugin.analysis_stage = 4
        save_attributes(self.plugin, self.plugin.pkl_Path)
        self.plugin.viewer.layers['Original Volume'].colormap = 'gray'
        self.plugin.viewer.layers['Protein Volume'].visible = not self.plugin.viewer.layers['Protein Volume'].visible
        self.plugin.loading_label.setText("")
        QApplication.processEvents()