from qtpy.QtWidgets import QApplication

class reset_exit:
    def __init__(self, plugin):
        self.plugin = plugin

    def exit_button(self):
        try:
            self.plugin.viewer.layers.clear()
        except Exception:
            pass
        try:
            if hasattr(self.plugin.viewer, "window") and self.plugin.viewer.window is not None:
                self.plugin.viewer.window.close()
        except Exception:
            pass

    def reset_button(self):
        # Clear status text
        try:
            self.plugin.loading_name.setText("")
            self.plugin.loading_label.setText("")
        except Exception:
            pass
        QApplication.processEvents()

        # Stop any running slice animation (best effort)
        try:
            if hasattr(self.plugin.viewer, "window") and self.plugin.viewer.window is not None:
                qt_viewer = self.plugin.viewer.window._qt_viewer
                anim = getattr(qt_viewer.dims, "_animation_thread", None)
                if anim and anim.isRunning():
                    anim.quit()
                    anim.wait()
        except Exception as e:
            print(f"[Animation stop warning] {e}")

        # Remove layers and reset the view
        try:
            self.plugin.viewer.layers.clear()
        except Exception as e:
            print(f"[Layer removal error] {e}")

        try:
            self.plugin.viewer.dims.ndisplay = 3
            self.plugin.viewer.reset_view()
        except Exception as e:
            print(f"[Viewer reset warning] {e}")

        # Reset the plugin's state (note: reset attributes on self.plugin)
        self.plugin.train_iter = 75000
        self.plugin.training_path = None
        self.plugin.analysis_stage = None
        self.plugin.pkl_Path = None
        self.plugin.filename_base = None
        self.plugin.full_stack_raw_images = None
        self.plugin.full_stack_length = None
        self.plugin.full_stack_raw_images_trimmed = None
        self.plugin.full_stack_rotated_images = None
        self.plugin.physical_resolution = None
        self.plugin.format = None
        self.plugin.npy_dir = None
        self.plugin.obj_dir = None
        self.plugin.start_trim = None
        self.plugin.end_trim = None
        self.plugin.display = None
        self.plugin.labeled_volume = None
        self.plugin.filtered_ids = None
        self.plugin.num_components = None
        self.plugin.delete_allowed = True
        self.plugin.id_list_annotation = None
        self.plugin.ID_positions_annotation = None
        self.plugin.start_points = None
        self.plugin.end_points = None
        self.plugin.start_points_most_updated = None
        self.plugin.end_points_most_updated = None
        self.plugin.start_points_layer = None
        self.plugin.end_points_layer = None
        self.plugin.lines_layer = None
        self.plugin.physical_distances = None
        self.plugin.IDs = None
        self.plugin.IDtoPointsMAP = None
        self.plugin.clustering = None
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
        self.plugin.scale_factor = 1
        self.plugin.start_end_points_properties = None
        self.plugin.rot_angle = 0


        QApplication.processEvents()
        print("✅ Plugin reset complete.")
