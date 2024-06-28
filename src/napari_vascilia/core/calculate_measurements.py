import os
import pandas as pd
from skimage.measure import regionprops

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
        measurements_dir = os.path.join(self.plugin.rootfolder, self.plugin.filename_base, 'measurements')

        if not os.path.exists(measurements_dir):
            os.makedirs(measurements_dir)

        props = regionprops(self.plugin.labeled_volume)
        measurements_list = []

        for prop in props:
            label = prop.label
            volume = prop.area  # in voxels
            centroid = prop.centroid

            # Create a dictionary for each label's properties and append it to the list
            measurements_list.append({
                'Label': label,
                'Volume (voxels)': volume,
                'Centroid (y, x, z)': centroid
            })

        # Convert the list of dictionaries to a DataFrame and export to CSV
        df = pd.DataFrame(measurements_list)
        df.to_csv(os.path.join(measurements_dir, "measurements.csv"), index=False)