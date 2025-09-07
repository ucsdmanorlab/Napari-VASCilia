import os
import pandas as pd
from skimage.measure import regionprops
import numpy as np
from qtpy.QtWidgets import QApplication
from tifffile import imread
import json


class CalculateMeasurementsAction:
    """
    This class handles the action of calculating measurements for labeled volumes.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the CalculateMeasurementsAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def execute(self):
        """
        Executes the action to calculate measurements for labeled volumes.
        """
        self.plugin.loading_label.setText("<font color='red'>Measurements Processing..., Wait</font>")
        QApplication.processEvents()

        measurements_dir = os.path.join(self.plugin.rootfolder, self.plugin.filename_base, 'measurements')

        os.makedirs(measurements_dir, exist_ok=True)

        # 3D properties calculation
        props_3d = regionprops(self.plugin.labeled_volume)
        measurements_3d = []

        for prop in props_3d:
            measurements_3d.append({
                'Label': prop.label,
                'Volume (voxels)': prop.area,
                'Centroid (z, y, x)': prop.centroid,
                'Bounding Box (min_z, min_y, min_x, max_z, max_y, max_x)': prop.bbox,
                'Solidity': prop.solidity,
                'Extent': prop.extent,
                'Euler Number': prop.euler_number
            })

        df_3d = pd.DataFrame(measurements_3d)
        df_3d.to_csv(os.path.join(measurements_dir, "measurements_3d.csv"), index=False)

        # 2D properties calculation on max projection along z-axis
        max_proj = np.max(self.plugin.labeled_volume, axis=2)
        props_2d = regionprops(max_proj)
        measurements_2d = []

        for prop in props_2d:
            measurements_2d.append({
                'Label': prop.label,
                'Area (pixels)': prop.area,
                'Centroid (y, x)': prop.centroid,
                'Bounding Box (min_y, min_x, max_y, max_x)': prop.bbox,
                'Orientation (radians)': prop.orientation,
                'Major Axis Length': prop.major_axis_length,
                'Minor Axis Length': prop.minor_axis_length,
                'Eccentricity': prop.eccentricity,
                'Convex Area': prop.convex_area,
                'Equivalent Diameter': prop.equivalent_diameter
            })

        df_2d = pd.DataFrame(measurements_2d)
        df_2d.to_csv(os.path.join(measurements_dir, "measurements_2d.csv"), index=False)

        # Save the labeled volume
        np.save(os.path.join(measurements_dir, self.plugin.filename_base + '_StereociliaBundle_labeled_volume.npy'), self.plugin.labeled_volume)
        if self.plugin.clustering == 1:
            np.save(os.path.join(measurements_dir, self.plugin.filename_base + '_StereociliaBundle_IHC_OHC.npy'),self.plugin.IHC_OHC)
        self.extract_and_save_3D_crops()
        self.plugin.loading_label.setText("")
        QApplication.processEvents()

    def extract_and_save_3D_crops(self):
        labeled_volume = self.plugin.labeled_volume  # shape: (Y, X, Z)
        save_dir = os.path.join(self.plugin.rootfolder, self.plugin.filename_base, 'measurements', 'crops')
        os.makedirs(save_dir, exist_ok=True)

        # Load raw stack from full_stack_rotated_images folder
        image_files = sorted([
            f for f in os.listdir(self.plugin.full_stack_rotated_images)
            if f.lower().endswith(('.tif', '.tiff'))
        ])
        raw_stack = np.stack([
            imread(os.path.join(self.plugin.full_stack_rotated_images, f))
            for f in image_files
        ])  # shape: (Z, Y, X, 3)

        z_dict = {}
        margin = 5

        unique_labels = np.unique(labeled_volume)
        unique_labels = unique_labels[unique_labels != 0]

        for label_id in unique_labels:
            binary = labeled_volume == label_id  # shape: (Y, X, Z)

            # Find Z bounds
            z_proj = np.any(binary, axis=(0, 1))  # (Z,)
            z_indices = np.where(z_proj)[0]
            if len(z_indices) == 0:
                continue
            z_label_start = z_indices[0]
            z_label_end = z_indices[-1] + 1  # +1 for slicing

            # Find Y and X bounds
            y_proj = np.any(binary, axis=(1, 2))
            x_proj = np.any(binary, axis=(0, 2))
            y_indices = np.where(y_proj)[0]
            x_indices = np.where(x_proj)[0]

            y_start, y_end = y_indices[0], y_indices[-1] + 1
            x_start, x_end = x_indices[0], x_indices[-1] + 1

            # Add margins (optional)
            y_start_margin = max(0, y_start - margin)
            y_end_margin = min(labeled_volume.shape[0], y_end + margin)
            x_start_margin = max(0, x_start - margin)
            x_end_margin = min(labeled_volume.shape[1], x_end + margin)
            z_start_margin = max(0, z_label_start - margin)
            z_end_margin = min(labeled_volume.shape[2], z_label_end + margin)

            # Compute z offset within crop
            z_label_start_in_crop = z_label_start - z_start_margin
            z_label_end_in_crop = z_label_end - z_start_margin

            # Crop labeled mask: (Y, X, Z)
            label_crop = labeled_volume[y_start_margin:y_end_margin, x_start_margin:x_end_margin,
                         z_start_margin:z_end_margin]

            # Crop raw stack: (Z, Y, X, 3)
            raw_crop = raw_stack[z_start_margin:z_end_margin, y_start_margin:y_end_margin, x_start_margin:x_end_margin,
                       :]

            # Save crops
            # Default file name
            label_name = f'label_crop_{label_id}.npy'

            if self.plugin.clustering == 1:
                # Define label → cell type mapping
                cell_type_map = {
                    9: 'IHC',
                    16: 'OHC1',
                    37: 'OHC2',
                    28: 'OHC3'
                }

                # Get a binary mask for the current label
                label_mask = labeled_volume == label_id

                # Extract the corresponding values from the clustered_cells array
                cell_type_values = self.plugin.clustered_cells[label_mask]

                # Use the most frequent (mode) value as the cell type ID
                if cell_type_values.size > 0:
                    cell_type_id = int(np.bincount(cell_type_values).argmax())
                    cell_type_name = cell_type_map.get(cell_type_id)

                    if cell_type_name:
                        label_name = f'label_crop_{label_id}_{cell_type_name}.npy'

            # Save the label crop with the appropriate file name
            np.save(os.path.join(save_dir, label_name), label_crop)
            np.save(os.path.join(save_dir, f'raw_crop_{label_id}.npy'), raw_crop)

            # Save metadata
            z_dict[int(label_id)] = {
                'z_start_global': int(z_label_start),
                'z_end_global': int(z_label_end),
                'z_crop_start': int(z_start_margin),
                'z_crop_end': int(z_end_margin),
                'z_label_start_in_crop': int(z_label_start_in_crop),
                'z_label_end_in_crop': int(z_label_end_in_crop),
                'y_start': int(y_start_margin),
                'y_end': int(y_end_margin),
                'x_start': int(x_start_margin),
                'x_end': int(x_end_margin)
            }

        # Save the dictionary
        with open(os.path.join(save_dir, 'z_index_dict.json'), 'w') as f:
            json.dump(z_dict, f, indent=2)

        print(f"Saved {len(z_dict)} crops to {save_dir}")