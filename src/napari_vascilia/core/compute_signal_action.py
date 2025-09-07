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

        def save_per_layer_intensity(label3D, image_3d, celltype, background3D_intensity=0, subtract_background=True):
            """
            Save the total and mean intensity per Z-layer for each labeled region.
            Applies background subtraction if enabled.
            """
            intensity_dir = self.plugin.rootfolder + '/' + self.plugin.filename_base + '/Intensity_response/'
            output_path = os.path.join(intensity_dir, celltype, 'All_bundles_per_layer_intensity.csv')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            labels = np.unique(label3D)
            labels = labels[labels != 0]  # exclude background label

            with open(output_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Region ID', 'Z Index', 'Total Intensity', 'Mean Intensity'])

                for label in labels:
                    mask = label3D == label
                    for z in range(label3D.shape[2]):
                        mask_slice = mask[:, :, z]
                        img_slice = image_3d[:, :, z]

                        if np.any(mask_slice):
                            total_intensity = img_slice[mask_slice].sum()
                            mean_intensity = img_slice[mask_slice].mean()

                            if subtract_background:
                                total_intensity -= (background3D_intensity * mask_slice.sum())
                                mean_intensity -= background3D_intensity

                            total_intensity = max(total_intensity, 0)
                            mean_intensity = max(mean_intensity, 0)
                        else:
                            total_intensity = 0
                            mean_intensity = 0

                        writer.writerow([label, z, total_intensity, mean_intensity])

            print(f"Per-layer intensity saved to {output_path}")

        def dilate_label_within_bounding_box(label3D, structuring_element=None):
            if structuring_element is None:
                structuring_element = ball(7)  # Or any other desired size
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

        def plot_responces(label3D, image_3d, celltype, barcolor, min_intensity, max_intensity, max_mean_intensity, subtract_background=True):

            # if celltype == 'Allcells':
            #     self.plugin.viewer.add_labels(label3D, name=f"{celltype}_dilated")
            props = regionprops(label3D, intensity_image=image_3d)
            # identify the background mask to do background subtraction
            background3D_intensity = 0
            if subtract_background:
                print('Background Subtraction has been applied')
                background3D = label3D == 0
                background3D_intensity = np.mean(image_3d[background3D])
            # Initialize lists to store mean and total intensities
            mean_intensities = []
            total_intensities = []
            labels = []
            # Collect mean and total intensity for each region
            for region in props:
                labels.append(region.label)

                if subtract_background:
                    mean_intensity = region.mean_intensity - background3D_intensity
                    total_intensity = region.intensity_image.sum() - (background3D_intensity * region.area)
                else:
                    mean_intensity = region.mean_intensity
                    total_intensity = region.intensity_image.sum()

                mean_intensity = max(mean_intensity, 0)
                total_intensity = max(total_intensity, 0)

                mean_intensities.append(mean_intensity)
                total_intensities.append(total_intensity)

            # CSV file
            # Now, write the collected data to a CSV file
            intensity_dir = self.plugin.rootfolder + '/' + self.plugin.filename_base + '/Intensity_response/'
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
            return min_intensity, max_intensity, max_mean_intensity, background3D_intensity

        self.plugin.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()
        intensity_dir = self.plugin.rootfolder + '/' + self.plugin.filename_base + '/Intensity_response/'

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
            signalch = im[:, :, self.plugin.signal_intensity_channel]   #redch = im[:, :, 0] for red channles (eps8) and redch = im[:, :, 1] for pheloiden
            image_3d[:, :, idx] = signalch
            if idx == 0:
                print('The intensity processed image is:')
                print(self.plugin.signal_intensity_channel)

        darker_magenta = (0.8, 0, 0.8)
        print(self.plugin.subtract_background)
        label3D = self.plugin.labeled_volume
        IHC = self.plugin.IHC
        OHC = self.plugin.OHC
        OHC1 = self.plugin.OHC1
        OHC2 = self.plugin.OHC2
        OHC3 = self.plugin.OHC3
        if self.plugin.dilate_labels:
            label3D = dilate_label_within_bounding_box(label3D, structuring_element=None)
            layer_name = 'Dilated Labeled Volume'
            if layer_name in self.plugin.viewer.layers:
                self.plugin.viewer.layers[layer_name].data = label3D
            else:
                self.plugin.viewer.add_labels(label3D, name=layer_name)

            IHC = dilate_label_within_bounding_box(IHC, structuring_element=None)
            OHC = dilate_label_within_bounding_box(OHC, structuring_element=None)
            OHC1 = dilate_label_within_bounding_box(OHC1, structuring_element=None)
            OHC2 = dilate_label_within_bounding_box(OHC2, structuring_element=None)
            OHC3 = dilate_label_within_bounding_box(OHC3, structuring_element=None)
        [min_intensity, max_intensity, max_mean_intensity, background3D_intensity] = plot_responces(label3D, image_3d, 'Allcells', 'magenta', 0, 0, 0, self.plugin.subtract_background) #darker_magenta
        save_per_layer_intensity(label3D, image_3d, 'Allcells', background3D_intensity =background3D_intensity, subtract_background=self.plugin.subtract_background)
        if self.plugin.clustering == 1:
            [_, _, _, _] = plot_responces(IHC, image_3d, 'IHCs', 'yellow', min_intensity, max_intensity, max_mean_intensity, self.plugin.subtract_background)
            [_, _, _, _] = plot_responces(OHC, image_3d, 'OHCs', 'red', min_intensity, max_intensity, max_mean_intensity, self.plugin.subtract_background)
            [_, _, _, _] = plot_responces(OHC1, image_3d, 'OHC1', 'skyblue', min_intensity, max_intensity, max_mean_intensity, self.plugin.subtract_background)
            [_, _, _, _] = plot_responces(OHC2, image_3d, 'OHC2', 'lightgreen', min_intensity, max_intensity, max_mean_intensity, self.plugin.subtract_background)
            [_, _, _, _] = plot_responces(OHC3, image_3d, 'OHC3', 'thistle', min_intensity, max_intensity, max_mean_intensity, self.plugin.subtract_background)
        self.plugin.loading_label.setText("")
        QApplication.processEvents()
        plt.close('all')
        print('done')
