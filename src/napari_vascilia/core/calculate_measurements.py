import os
import pandas as pd
from skimage.measure import regionprops
import numpy as np
from qtpy.QtWidgets import QApplication

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
        np.save(os.path.join(measurements_dir, 'StereociliaBundle_labeled_volume.npy'), self.plugin.labeled_volume)
        self.plugin.loading_label.setText("")
        QApplication.processEvents()
