import numpy as np
from skimage.measure import regionprops
from PyQt5.QtWidgets import QMessageBox, QApplication
from magicgui import magicgui
from .VASCilia_utils import save_attributes  # Import the utility functions

class DeleteAction:
    """
    This class handles the action of deleting labels in the Cochlea segmentations.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the DeleteAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def create_filter_component_widget(self):
        @magicgui(call_button="Delete Label")
        def _widget(component: int):
            if self.plugin.delete_allowed:  # Check if deletion is allowed
                self.delete_label(component)
            else:
                QMessageBox.warning(None, 'Delete', 'Deletion is not allowed in this step of analysis')

        return _widget

    def delete_label(self, component: int):
        self.plugin.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()
        label_to_remove = component
        self.plugin.labeled_volume[self.plugin.labeled_volume == label_to_remove] = 0
        self.plugin.viewer.layers['Labeled Image'].data = self.plugin.labeled_volume
        self.plugin.filtered_ids.append(label_to_remove)
        self.plugin.num_components -= 1

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

        self.plugin.ID_layer.text = self.plugin.id_list_annotation  # Assign text annotations to the points
        self.plugin.ID_layer.text.color = 'lime'
        self.plugin.ID_layer.text.size = 12

        if self.plugin.clustering == 1:
            coordinates_to_remove = np.argwhere(self.plugin.labeled_volume == label_to_remove)
            mask = np.zeros_like(self.plugin.IHC, dtype=bool)
            mask[tuple(coordinates_to_remove.T)] = True
            self.plugin.IHC[mask] = 0
            self.plugin.OHC[mask] = 0
            self.plugin.IHC_OHC[mask] = 0
            self.plugin.OHC1[mask] = 0
            self.plugin.OHC2[mask] = 0
            self.plugin.OHC3[mask] = 0
            self.plugin.clustered_cells[mask] = 0
            self.plugin.viewer.layers['Clustered Cells'].data = self.plugin.clustered_cells
            layer_name = 'IHCs vs OHCs'
            if layer_name in self.plugin.viewer.layers:
                self.plugin.viewer.layers[layer_name].data = self.plugin.IHC_OHC

        if self.plugin.analysis_stage == 5:
            for idpoints, idcc in self.plugin.IDtoPointsMAP:
                if idcc == component:
                    myidpoints = idpoints
                    break
            templist = list(self.plugin.start_points)
            del templist[myidpoints]
            self.plugin.start_points = np.array(templist)
            templist = list(self.plugin.end_points)
            del templist[myidpoints]
            self.plugin.end_points = np.array(templist)
            templist = list(self.plugin.IDs)
            del templist[myidpoints]
            self.plugin.IDs = np.array(templist)
            self.plugin.viewer.layers['Peak Points'].data = self.plugin.start_points
            self.plugin.viewer.layers['Base Points'].data = self.plugin.end_points
            self.plugin.start_points_layer.data = self.plugin.start_points
            self.plugin.end_points_layer.data = self.plugin.end_points
            new_lines = []
            if len(self.plugin.start_points_layer.data) == len(self.plugin.end_points_layer.data):
                for start, end in zip(self.plugin.start_points_layer.data, self.plugin.end_points_layer.data):
                    new_lines.append([start, end])
            self.plugin.viewer.layers['Lines'].data = new_lines
            IDtoPointsMAP_list = []
            tempid = 0
            for cc in range(self.plugin.num_components + 1 + len(self.plugin.filtered_ids)):
                if cc == 0 or cc in self.plugin.filtered_ids:
                    continue
                IDtoPointsMAP_list.append((tempid, cc))
                tempid += 1
            self.plugin.IDtoPointsMAP = tuple(IDtoPointsMAP_list)

        if self.plugin.analysis_stage == 6:
            for idpoints, idcc in self.plugin.IDtoPointsMAP:
                if idcc == component:
                    myidpoints = idpoints
                    break
            templist = list(self.plugin.start_points_most_updated)
            del templist[myidpoints]
            self.plugin.start_points_most_updated = np.array(templist)
            templist = list(self.plugin.end_points_most_updated)
            del templist[myidpoints]
            self.plugin.end_points_most_updated = np.array(templist)
            templist = list(self.plugin.start_points)
            del templist[myidpoints]
            self.plugin.start_points = np.array(templist)
            templist = list(self.plugin.end_points)
            del templist[myidpoints]
            self.plugin.end_points = np.array(templist)
            templist = list(self.plugin.IDs)
            del templist[myidpoints]
            self.plugin.IDs = np.array(templist)
            self.plugin.viewer.layers['Peak Points'].data = self.plugin.start_points_most_updated
            self.plugin.viewer.layers['Base Points'].data = self.plugin.end_points_most_updated
            self.plugin.start_points_layer.data = self.plugin.start_points_most_updated
            self.plugin.end_points_layer.data = self.plugin.end_points_most_updated
            new_lines = []
            if len(self.plugin.start_points_layer.data) == len(self.plugin.end_points_layer.data):
                for start, end in zip(self.plugin.start_points_layer.data, self.plugin.end_points_layer.data):
                    new_lines.append([start, end])
            self.plugin.viewer.layers['Lines'].data = new_lines
            IDtoPointsMAP_list = []
            tempid = 0
            for cc in range(self.plugin.num_components + 1 + len(self.plugin.filtered_ids)):
                if cc == 0 or cc in self.plugin.filtered_ids:
                    continue
                IDtoPointsMAP_list.append((tempid, cc))
                tempid += 1
            self.plugin.IDtoPointsMAP = tuple(IDtoPointsMAP_list)

        save_attributes(self.plugin,self.plugin.pkl_Path)
        self.plugin.loading_label.setText("")