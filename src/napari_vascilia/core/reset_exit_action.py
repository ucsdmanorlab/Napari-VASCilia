from qtpy.QtWidgets import QApplication


class reset_exit:
    """
    This class handles the action of resetting or exiting VASCilia.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the commute of resetting and exitting VASCilia with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin


    def exit_button(self):
        while len(self.plugin.viewer.layers) > 0:
            self.plugin.viewer.layers.pop(0)
        self.plugin.viewer.window.close()

    def reset_button(self):
        self.plugin.loading_name.setText("")
        self.plugin.loading_label.setText("")
        QApplication.processEvents()
        while len(self.plugin.viewer.layers) > 0:
            self.plugin.viewer.layers.pop(0)
        self.plugin.train_iter = 50000
        self.plugin.training_path = None
        self.plugin.analysis_stage = None
        self.plugin.pkl_Path = None
        self.plugin.filename_base = None  # file name
        self.plugin.full_stack_raw_images = None
        self.plugin.full_stack_length = None
        self.plugin.full_stack_raw_images_trimmed = None
        self.plugin.full_stack_rotated_images = None
        self.plugin.physical_resolution = None
        self.plugin.npy_dir = None  # This is for npy files after we run the prediction
        self.plugin.obj_dir = None  # This is for new_assignment_obj after we run the prediction, this will be used by the tracking algorithm
        self.plugin.start_trim = None  # Which slice to start trimming
        self.plugin.end_trim = None  # Which slice to end trimming
        self.plugin.display = None  # to control which frames to display, None  to display full stack, 1 for full_stack_raw_images_trimmed, And 2 for full_stack_rotated_images
        self.plugin.labeled_volume = None  # This is the labled volume that will be added through self.viewer.add_labels, and I generated it after prediction and tracking
        self.plugin.filtered_ids = None  # The objects that have been filtered when depth < 3
        self.plugin.num_components = None  # Number of compoenents in the labeled volume
        self.plugin.start_points = None
        self.plugin.end_points = None
        self.plugin.start_points_most_updated = None
        self.plugin.end_points_most_updated = None
        self.plugin.start_points_layer = None  # for distance calculationself.start_points_layer.data, self.end_points_layer.data
        self.plugin.end_points_layer = None  # for distance calculation
        self.plugin.lines_layer = None  # for distance calculation
        self.plugin.physical_distances = None
        self.plugin.IDs = None
        self.plugin.IDtoPointsMAP = None  # to map which cc ID corresponds to which starting point, ending point and line
        self.plugin.clustering = None  # to inform the sotware that the clustering is done, mainly for delete buttom
        self.plugin.clustered_cells = None
        self.plugin.IHC = None
        self.plugin.IHC_OHC = None
        self.plugin.OHC = None
        self.plugin.OHC1 = None
        self.plugin.OHC2 = None
        self.plugin.OHC3 = None
        self.plugin.gt = None
        self.plugin.lines_with_z_swapped = None
        self.plugin.text_positions = None
        self.plugin.text_annotations = None
        self.plugin.id_list = None
        self.plugin.orientation = None
        self.plugin.delete_allowed = True
        self.plugin.id_list_annotation = None  # This is for label annotation (label ID)
        self.plugin.ID_positions_annotation = None  # This is for label annotation (label ID)
        self.scale_factor = 1

