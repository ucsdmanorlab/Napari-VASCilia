import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from skimage.measure import regionprops
from skimage.io import imsave, imread
from scipy.ndimage import binary_dilation
from scipy import ndimage
import csv
import os
import numpy as np
from skimage.morphology import ball
from qtpy.QtWidgets import QApplication

class ComputeSignalAction:

    """
    This class handles the action of signal computation for labeled volumes.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        self.plugin = plugin

    def compute_protein_intensity(self):
        def dilate_label_within_bounding_box(label3D, structuring_element=None):
            if structuring_element is None:
                structuring_element = ball(5)  # Or any other desired size
            # Initialize the output image
            dilated_label3D = np.zeros_like(label3D)
            # Get unique labels, ignoring background (0)
            unique_labels = np.unique(label3D)[1:]
            # Iterate over each label to perform localized dilation
            for label in unique_labels:
                # Extract the bounding box of the current region
                region_slices = ndimage.find_objects(label3D == label)[0]
                local_region = label3D[region_slices]
                # Create a binary mask for the current region
                binary_mask = local_region == label
                # Dilate the binary mask within the local region
                dilated_local_mask = binary_dilation(binary_mask, structure=structuring_element)
                # Place the dilated region back into the full-size image
                dilated_label3D[region_slices][dilated_local_mask] = label
            return dilated_label3D

        def plot_responces(label3D, image_3d, celltype, barcolor, min_intensity, max_intensity, max_mean_intensity):
            label3D = dilate_label_within_bounding_box(label3D, structuring_element=None)
            props = regionprops(label3D, intensity_image=image_3d)
            # identify the background mask to do background subtraction
            background3D = label3D == 0
            background3D_intensity = np.mean(image_3d[background3D])
            # Initialize lists to store mean and total intensities
            mean_intensities = []
            total_intensities = []
            labels = []
            # Collect mean and total intensity for each region
            for region in props:
                labels.append(region.label)
                # mean_intensity
                mean_intensity = region.mean_intensity - background3D_intensity
                mean_intensity = max(mean_intensity, 0)
                mean_intensities.append(mean_intensity)
                # total_intensity
                total_intensity = region.intensity_image.sum() - (background3D_intensity * region.area)
                total_intensity = max(total_intensity, 0)
                total_intensities.append(total_intensity)

            # CSV file
            # Now, write the collected data to a CSV file
            intensity_dir = self.plugin.rootfolder + '/' + self.plugin.filename_base + '/Protein_responce/'
            with open(intensity_dir + '/' + celltype + '/' + 'region_intensities.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(['Region ID', 'Mean Intensity', 'Total Intensity'])

                # Write the data
                for label, mean_intensity, total_intensity in zip(labels, mean_intensities, total_intensities):
                    writer.writerow([label, mean_intensity, total_intensity])

            if celltype == 'Allcells':
                total_intensities = np.array(total_intensities)
                min_intensity = total_intensities.min()
                max_intensity = total_intensities.max()
                total_intensities = (total_intensities - min_intensity) / (max_intensity - min_intensity)
                mean_intensities = np.array(mean_intensities)
                max_mean_intensity = mean_intensities.max()
            else:
                total_intensities = np.array(total_intensities)
                total_intensities = (total_intensities - min_intensity) / (max_intensity - min_intensity)

            plt.figure(figsize=(12, 6))
            plt.bar(labels, mean_intensities, color=barcolor)
            plt.title('Mean Intensity for Each Hair Cell')
            plt.xlabel('Stereocilia Bundle')
            plt.ylabel('Mean Intensity')
            plt.ylim(0, max_mean_intensity)
            plt.xticks(labels)
            plt.tight_layout()
            plt.savefig(intensity_dir + '/' + celltype + '/' + 'mean_intensity_per_cell.png', dpi=300)

            plt.figure(figsize=(12, 6))
            plt.bar(labels, total_intensities, color=barcolor)
            plt.title('Total Intensity for Each Hair Cell')
            plt.xlabel('Stereocilia Bundle')
            plt.ylabel('Total Intensity')
            plt.xticks(labels)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(intensity_dir + '/' + celltype + '/' + 'total_intensity_per_cell.png', dpi=300)
            return min_intensity, max_intensity, max_mean_intensity

        self.plugin.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()
        intensity_dir = self.plugin.rootfolder + '/' + self.plugin.filename_base + '/Protein_responce/'

        if not os.path.exists(intensity_dir + '/' + 'Allcells'):
            os.makedirs(intensity_dir + '/' + 'Allcells')

        if not os.path.exists(intensity_dir + '/' + 'IHCs'):
            os.makedirs(intensity_dir + '/' + 'IHCs')

        if not os.path.exists(intensity_dir + '/' + 'OHCs'):
            os.makedirs(intensity_dir + '/' + 'OHCs')

        if not os.path.exists(intensity_dir + '/' + 'OHC1'):
            os.makedirs(intensity_dir + '/' + 'OHC1')

        if not os.path.exists(intensity_dir + '/' + 'OHC2'):
            os.makedirs(intensity_dir + '/' + 'OHC2')

        if not os.path.exists(intensity_dir + '/' + 'OHC3'):
            os.makedirs(intensity_dir + '/' + 'OHC3')

        image_files = [f for f in os.listdir(self.plugin.full_stack_rotated_images) if
                       f.endswith('.tif')]  # Adjust the extension based on your image files
        image_3d = np.zeros(self.plugin.labeled_volume.shape, dtype=np.uint8)
        for idx, image in enumerate(image_files):
            im = imread(os.path.join(self.plugin.full_stack_rotated_images, image))
            redch = im[:, :, 0]   #redch = im[:, :, 0] for red channles (eps8) and redch = im[:, :, 1] for pheloiden
            image_3d[:, :, idx] = redch

        darker_magenta = (0.8, 0, 0.8)
        [min_intensity, max_intensity, max_mean_intensity] = plot_responces(self.plugin.labeled_volume, image_3d, 'Allcells', 'magenta', 0, 0, 0) #darker_magenta
        if self.plugin.clustering == 1:
            [_, _, _] = plot_responces(self.plugin.IHC, image_3d, 'IHCs', 'yellow', min_intensity, max_intensity, max_mean_intensity)
            [_, _, _] = plot_responces(self.plugin.OHC, image_3d, 'OHCs', 'red', min_intensity, max_intensity, max_mean_intensity)
            [_, _, _] = plot_responces(self.plugin.OHC1, image_3d, 'OHC1', 'skyblue', min_intensity, max_intensity, max_mean_intensity)
            [_, _, _] = plot_responces(self.plugin.OHC2, image_3d, 'OHC2', 'lightgreen', min_intensity, max_intensity, max_mean_intensity)
            [_, _, _] = plot_responces(self.plugin.OHC3, image_3d, 'OHC3', 'thistle', min_intensity, max_intensity, max_mean_intensity)
        self.plugin.loading_label.setText("")
        QApplication.processEvents()
        print('done')
