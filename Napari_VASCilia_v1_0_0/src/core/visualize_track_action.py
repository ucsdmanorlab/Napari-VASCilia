import matplotlib
matplotlib.use('Qt5Agg')
from skimage.measure import label, regionprops
import imageio
from skimage.io import imread
from scipy.ndimage import find_objects
import os
import numpy as np
#-------------- Qui
from PyQt5.QtWidgets import QApplication
from .VASCilia_utils import save_attributes  # Import the utility functions

class VisualizeTrackAction:
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
        self.plugin.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()

        def overlap_with_previous(component, previous_mask):
            """Returns the label of the overlap from the previous mask, 0 if none."""
            overlap = np.bincount(previous_mask[component].flatten())
            overlap[0] = 0  # ignore the background
            if overlap.max() > 300:
                return overlap.argmax()
            else:
                return 0

        self.plugin.npy_dir = os.path.dirname(self.plugin.full_stack_rotated_images) + '/prediction/'
        self.plugin.obj_dir = os.path.dirname(self.plugin.full_stack_rotated_images) + '/new_assignment_obj/'

        npy_files = [f for f in os.listdir(self.plugin.npy_dir) if f.endswith('.npy')]
        if not os.path.exists(self.plugin.obj_dir):
            os.makedirs(self.plugin.obj_dir)

        previous_mask = None
        latest_label = 0

        for im_file in npy_files:
            file_path = os.path.join(self.plugin.npy_dir, os.path.basename(im_file))
            data = np.load(file_path, allow_pickle=True).item()
            boxes = data['boxes']
            scores = data['scores']
            masks = data['masks']
            labeled_mask = np.zeros_like(masks[0], dtype=np.int32)
            for i, mask in enumerate(masks):
                labeled_mask[mask > 0] = i + 1
            num_features = i + 1
            temp = labeled_mask.copy()
            if previous_mask is not None:
                for i in range(1, num_features + 1):
                    component = (labeled_mask == i)
                    overlap_label = overlap_with_previous(component, previous_mask)
                    if overlap_label:
                        temp[component] = overlap_label
                    else:
                        latest_label += 1
                        temp[component] = latest_label
            else:
                latest_label = num_features

            imageio.imwrite(os.path.join(self.plugin.obj_dir, os.path.basename(im_file).replace('.npy', '.png')), temp.astype(np.uint8))
            previous_mask = temp

        masks = []
        newmask_files = sorted([os.path.join(self.plugin.obj_dir, f) for f in os.listdir(self.plugin.obj_dir) if f.endswith('.png')])
        for maskfile in newmask_files:
            mask = imread(maskfile)
            masks.append(mask)

        self.plugin.labeled_volume = np.stack(masks, axis=-1)
        self.plugin.filtered_ids = []
        regions = find_objects(self.plugin.labeled_volume)

        for i in range(1, len(regions) + 1):
            region_slice = regions[i - 1]
            depth = region_slice[2].stop - region_slice[2].start
            if depth < 1:
                self.plugin.filtered_ids.append(i)
                self.plugin.labeled_volume[self.plugin.labeled_volume == i] = 0

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