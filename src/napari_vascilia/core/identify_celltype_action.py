import matplotlib
matplotlib.use('Qt5Agg')
from skimage.measure import label, regionprops
from skimage.io import imsave, imread
import cv2
from magicgui import magicgui
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
# Region classification
import torch
import os
import numpy as np
from cv2 import resize
from skimage.transform import resize as skiresize
#-------------- Qui
from qtpy.QtWidgets import QMessageBox
from qtpy.QtWidgets import QApplication, QPushButton
from qtpy.QtWidgets import  QLineEdit, QHBoxLayout
from qtpy.QtWidgets import QPushButton, QWidget, QComboBox
from .VASCilia_utils import save_attributes  # Import the utility functions

class CellClusteringAction:
    """
    This class handles the action of clustering cells using various methods.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the CellClusteringAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def create_clustering_widget(self):
        @magicgui(call_button='Perform Cell Clustering', method={'choices': ['GMM', 'KMeans','Deep Learning']})
        def clustering_widget(method: str):
            self.perform_clustering(method)
        return clustering_widget

    def perform_clustering(self, method: str):
        def find_max_overlap_label(cc_2d_mask, finalmask):
            # Ensure cc_2d_mask's label is represented as a boolean mask for the calculation
            cc_2d_boolean_mask = cc_2d_mask > 0

            # Find unique labels in finalmask (excluding background)
            unique_labels = np.unique(finalmask[finalmask > 0])

            # Initialize variable to track the largest overlap
            max_overlap = 0
            max_overlap_label = 0
            for label in unique_labels:
                # Calculate overlap between cc_2d_mask and current label in finalmask
                overlap = np.sum((finalmask == label) & cc_2d_boolean_mask)

                # Update max_overlap and max_overlap_label if this label has a larger overlap
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_overlap_label = label.copy()
            return max_overlap_label

        flag = os.path.exists(self.plugin.rootfolder + '\\' + self.plugin.filename_base + '\\Distances\\' + 'Physical_distances.csv')
        if self.plugin.analysis_stage < 5 or not flag:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Bundle Height Calculation')
            msg_box.setText('Press first Calculate Bundle Height and Save Bundle Height')
            msg_box.exec_()
            return

        self.plugin.loading_label.setText("<font color='red'>Cell Type Processing..., Wait</font>")
        QApplication.processEvents()
        self.plugin.viewer.layers['Lines'].visible = False
        self.plugin.viewer.layers['Base Points'].visible = False
        self.plugin.viewer.layers['Peak Points'].visible = False
        self.plugin.viewer.layers['Labeled Image'].visible = False
        layer_name = 'IHCs vs OHCs'
        if layer_name in self.plugin.viewer.layers:
            self.plugin.viewer.layers['IHCs vs OHCs'].visible = False

        if method == 'GMM':
            Basepoints_y = [point[0] for point in self.plugin.end_points]  # Extracting the y-values
            lebel_list = []
            for ((point_lbl, reallable), val) in zip(self.plugin.IDtoPointsMAP, Basepoints_y):
                lebel_list.append({
                    'Label': reallable,
                    'Centroid (y, x)': val})
            Basepoints_y_array = np.array(Basepoints_y).reshape(-1, 1)

            gmm = GaussianMixture(n_components=4, random_state=0)
            gmm.fit(Basepoints_y_array)
            labels = gmm.predict(Basepoints_y_array)
            cluster_centers = gmm.means_
            print('gmm')

        elif method == 'Deep Learning':
            Basepoints_y = [point[0] for point in self.plugin.end_points]  # Extracting the y-values
            lebel_list = []
            for ((point_lbl, reallable), val) in zip(self.plugin.IDtoPointsMAP, Basepoints_y):
                lebel_list.append({
                    'Label': reallable,
                    'Centroid (y, x)': val})
            Basepoints_y_array = np.array(Basepoints_y).reshape(-1, 1)

            gmm = GaussianMixture(n_components=4, random_state=0)
            gmm.fit(Basepoints_y_array)
            labels = gmm.predict(Basepoints_y_array)
            cluster_centers = gmm.means_
            #
            imglist = os.listdir(self.plugin.full_stack_rotated_images)
            maximg = imread(os.path.join(self.plugin.full_stack_rotated_images, imglist[0]))
            maximg = maximg[:, :, 1]
            imgshape = maximg.shape
            maximg = resize(maximg, (320, 320), cv2.INTER_LANCZOS4)

            for imgname in imglist:
                im = imread(os.path.join(self.plugin.full_stack_rotated_images, imgname))
                im = im[:, :, 1]
                im = resize(im, (320, 320), cv2.INTER_LANCZOS4)
                maximg = np.maximum(im, maximg)

            #rawimage = cv2.cvtColor(maximg, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(maximg, cv2.COLOR_GRAY2RGB)
            #image = np.pad(rawimage, ((8, 8), (8, 8), (0, 0)), mode='constant')
            mean = np.array([0.485, 0.485, 0.485])
            std = np.array([0.229, 0.229, 0.229])
            image = image / 255.  # scale pixel values to [0, 1]
            image = image - mean  # zero-center
            image = image / std  # normalize
            image = image[np.newaxis, :]
            image = np.transpose(image, (0, 3, 1, 2))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x_tensor = torch.from_numpy(image.astype('float32')).to(device)
            #
            best_model = torch.load(os.path.join(self.plugin.model_celltype_identification,'best_model_ihc_v3.pth'))
            pr_mask_ihc = best_model.predict(x_tensor)
            pr_mask_ihc = (pr_mask_ihc.squeeze().cpu().numpy().round())
            #
            best_model = torch.load(os.path.join(self.plugin.model_celltype_identification,'best_model_ohc1_v3.pth'))
            pr_mask_ohc1 = best_model.predict(x_tensor)
            pr_mask_ohc1 = (pr_mask_ohc1.squeeze().cpu().numpy().round())
            #
            best_model = torch.load(os.path.join(self.plugin.model_celltype_identification,'best_model_ohc2_v3.pth'))
            pr_mask_ohc2 = best_model.predict(x_tensor)
            pr_mask_ohc2 = (pr_mask_ohc2.squeeze().cpu().numpy().round())
            #
            best_model = torch.load(os.path.join(self.plugin.model_celltype_identification,'best_model_ohc3_v3.pth'))
            pr_mask_ohc3 = best_model.predict(x_tensor)
            pr_mask_ohc3 = (pr_mask_ohc3.squeeze().cpu().numpy().round())

            final_mask = pr_mask_ihc.copy()
            final_mask[pr_mask_ohc1 == 1] = 2
            final_mask[pr_mask_ohc3 == 1] = 4
            final_mask[pr_mask_ohc2 == 1] = 3

            final_mask = skiresize(final_mask, imgshape, order=0, preserve_range=True, anti_aliasing=False).astype(final_mask.dtype)
            print('deep learning')


        elif method == 'KMeans':
            props = regionprops(self.plugin.labeled_volume)
            centroids = []
            lebel_list = []

            for prop in props:
                label = prop.label
                centroid = prop.centroid[0]

                # Create a dictionary for each label's properties and append it to the list
                lebel_list.append({
                    'Label': label,
                    'Centroid (y, x)': centroid
                })

                centroids.append(centroid)
            centroids = np.array(centroids).reshape(-1, 1)
            kmeans = KMeans(n_clusters=4, random_state=0)
            kmeans.fit(centroids)
            labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_
            print('kmeans')

        # Prepare a list of labeled centroids to return
        labeled_centroids = [
            {"Label": lbl_dict["Label"], "Centroid": lbl_dict["Centroid (y, x)"], "Cluster": int(cluster_label)}
            for lbl_dict, cluster_label in zip(lebel_list, labels)
        ]
        # Step 1: Create a mapping from original labels to cluster IDs
        label_to_cluster = {item['Label']: item['Cluster'] for item in labeled_centroids}
        # Step 2: Create self.plugin.labeled_volume_clustered by replacing labels with cluster IDs
        self.plugin.labeled_volume_clustered = np.copy(self.plugin.labeled_volume)
        # Replace each label in the volume with its corresponding cluster ID
        for original_label, cluster_id in label_to_cluster.items():
            self.plugin.labeled_volume_clustered[self.plugin.labeled_volume == original_label] = cluster_id + 9
        ihc_cluster = np.argmax(cluster_centers[:, 0])
        for item in labeled_centroids:
            item['Cell Type'] = 'IHC' if item['Cluster'] == ihc_cluster else 'OHC'
        self.plugin.IHC_OHC = np.copy(self.plugin.labeled_volume_clustered)
        for label_info in labeled_centroids:
            label_val = 9 if label_info['Cell Type'] == 'IHC' else 10  # Example: 9 for IHC, 10 for OHC
            original_label = label_info['Label']
            self.plugin.IHC_OHC[self.plugin.labeled_volume == original_label] = label_val

        self.plugin.IHC = np.zeros_like(self.plugin.IHC_OHC)
        self.plugin.OHC = np.zeros_like(self.plugin.IHC_OHC)
        self.plugin.IHC[self.plugin.IHC_OHC == 9] = 1
        self.plugin.OHC[self.plugin.IHC_OHC == 10] = 1
        self.plugin.IHC = self.plugin.IHC * self.plugin.labeled_volume
        self.plugin.OHC = self.plugin.OHC * self.plugin.labeled_volume

        # identify OHC1
        cluster_centers[ihc_cluster] = 0
        ohc1_cluster = np.argmax(cluster_centers[:, 0])
        for item in labeled_centroids:
            item['Cell Type'] = 'OHC1' if item['Cluster'] == ohc1_cluster else 'any'
        self.plugin.OHC1 = np.copy(self.plugin.labeled_volume_clustered)

        for label_info in labeled_centroids:
            label_val = 10 if label_info['Cell Type'] == 'OHC1' else 11  # Example: 9 for IHC, 10 for OHC
            original_label = label_info['Label']
            self.plugin.OHC1[self.plugin.labeled_volume == original_label] = label_val

        temp = self.plugin.OHC1
        self.plugin.OHC1[temp == 10] = 1
        self.plugin.OHC1[temp == 11] = 0
        self.plugin.OHC1 = self.plugin.OHC1 * self.plugin.labeled_volume

        # identify OHC2
        cluster_centers[ohc1_cluster] = 0
        ohc2_cluster = np.argmax(cluster_centers[:, 0])

        for item in labeled_centroids:
            item['Cell Type'] = 'OHC2' if item['Cluster'] == ohc2_cluster else 'any'

        self.plugin.OHC2 = np.copy(self.plugin.labeled_volume_clustered)

        for label_info in labeled_centroids:
            label_val = 11 if label_info['Cell Type'] == 'OHC2' else 12  # Example: 9 for IHC, 10 for OHC
            original_label = label_info['Label']
            self.plugin.OHC2[self.plugin.labeled_volume == original_label] = label_val

        temp = self.plugin.OHC2
        self.plugin.OHC2[temp == 11] = 1
        self.plugin.OHC2[temp == 12] = 0
        self.plugin.OHC2 = self.plugin.OHC2 * self.plugin.labeled_volume

        # identify OHC3
        cluster_centers[ohc2_cluster] = 0
        ohc3_cluster = np.argmax(cluster_centers[:, 0])

        for item in labeled_centroids:
            item['Cell Type'] = 'OHC3' if item['Cluster'] == ohc3_cluster else 'any'

        self.plugin.OHC3 = np.copy(self.plugin.labeled_volume_clustered)

        for label_info in labeled_centroids:
            label_val = 12 if label_info['Cell Type'] == 'OHC3' else 13  # Example: 9 for IHC, 10 for OHC

            original_label = label_info['Label']

            self.plugin.OHC3[self.plugin.labeled_volume == original_label] = label_val

        temp = self.plugin.OHC3
        self.plugin.OHC3[temp == 12] = 1
        self.plugin.OHC3[temp == 13] = 0
        self.plugin.OHC3 = self.plugin.OHC3 * self.plugin.labeled_volume
        self.plugin.clustered_cells = np.copy(self.plugin.labeled_volume)  # to visualize it with colors
        self.plugin.clustered_cells[self.plugin.OHC3 > 0] = 28
        self.plugin.clustered_cells[self.plugin.OHC2 > 0] = 37
        self.plugin.clustered_cells[self.plugin.OHC1 > 0] = 16
        self.plugin.clustered_cells[self.plugin.IHC > 0] = 9

        if method == 'Deep Learning':

            temp_l3d = self.plugin.clustered_cells.copy()
            props = regionprops(self.plugin.labeled_volume)
            for prop in props:
                cc_3d_mask = np.zeros(self.plugin.labeled_volume.shape)
                cc = prop.label
                cc_3d_mask[self.plugin.labeled_volume == cc] = cc
                cc_2d_mask = np.max(cc_3d_mask, axis=2)
                cc_2d_mask_value = find_max_overlap_label(cc_2d_mask, final_mask)
                if cc_2d_mask_value != 0:
                    temp_l3d[self.plugin.labeled_volume == cc] = cc_2d_mask_value

            self.plugin.IHC = self.plugin.labeled_volume.copy()
            self.plugin.OHC1 = self.plugin.labeled_volume.copy()
            self.plugin.OHC2 = self.plugin.labeled_volume.copy()
            self.plugin.OHC3 = self.plugin.labeled_volume.copy()
            self.plugin.OHC = self.plugin.labeled_volume.copy()
            self.plugin.IHC[temp_l3d != 1] = 0
            self.plugin.OHC1[temp_l3d != 2] = 0
            self.plugin.OHC2[temp_l3d != 3] = 0
            self.plugin.OHC3[temp_l3d != 4] = 0
            self.plugin.OHC[temp_l3d == 1] = 0
            self.plugin.clustered_cells[temp_l3d == 4] = 28
            self.plugin.clustered_cells[temp_l3d == 3] = 37
            self.plugin.clustered_cells[temp_l3d == 2] = 16
            self.plugin.clustered_cells[temp_l3d == 1] = 9
            self.plugin.IHC_OHC = self.plugin.clustered_cells.copy()
            self.plugin.IHC_OHC[self.plugin.clustered_cells == 28] = 10
            self.plugin.IHC_OHC[self.plugin.clustered_cells == 37] = 10
            self.plugin.IHC_OHC[self.plugin.clustered_cells == 16] = 10

        Distance_path = self.plugin.rootfolder + '\\' + self.plugin.filename_base + '\\Distances\\' + 'Physical_distances.csv'
        data = pd.read_csv(Distance_path)
        data['CLass'] = 'Unknown'
        data.loc[data['ID'].isin(np.unique(self.plugin.IHC)), 'CLass'] = 'IHC'
        data.loc[data['ID'].isin(np.unique(self.plugin.OHC1)), 'CLass'] = 'OHC1'
        data.loc[data['ID'].isin(np.unique(self.plugin.OHC2)), 'CLass'] = 'OHC2'
        data.loc[data['ID'].isin(np.unique(self.plugin.OHC3)), 'CLass'] = 'OHC3'
        data.to_csv(Distance_path, index=False)
        self.plugin.clustering = 1
        self.plugin.delete_allowed = False
        save_attributes(self.plugin, self.plugin.pkl_Path)

        layer_name = 'Clustered Cells'
        if layer_name in self.plugin.viewer.layers:
            self.plugin.viewer.layers['Clustered Cells'].data = self.plugin.clustered_cells
            self.plugin.viewer.layers['Clustered Cells'].visible = True

        else:
            self.plugin.viewer.add_labels(self.plugin.clustered_cells, name='Clustered Cells')

        layer_name = 'IHCs vs OHCs'
        if layer_name in self.plugin.viewer.layers:
            self.plugin.viewer.layers['IHCs vs OHCs'].data = self.plugin.IHC_OHC
            self.plugin.viewer.layers['IHCs vs OHCs'].visible = False
        else:
            self.plugin.viewer.add_labels(self.plugin.IHC_OHC, name='IHCs vs OHCs')
            self.plugin.viewer.layers['IHCs vs OHCs'].visible = False

        self.plugin.loading_label.setText("")
        QApplication.processEvents()

    def find_IHC_OHC(self):
        self.plugin.viewer.layers['Clustered Cells'].visible = False
        layer_name = 'IHCs vs OHCs'
        if layer_name in self.plugin.viewer.layers:
            self.plugin.viewer.layers['IHCs vs OHCs'].data = self.plugin.IHC_OHC
            self.plugin.viewer.layers['IHCs vs OHCs'].visible = True
        else:
            self.plugin.viewer.add_labels(self.plugin.IHC_OHC, name='IHCs vs OHCs')

    def perform_reassign_clustering(self, method: str, mylabel: str):
        if self.plugin.clustering != 1:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Clustering is not available')
            msg_box.setText('Press First (Perform Cell Clustering) button')
            msg_box.exec_()
            return

        mylabel = int(mylabel)
        if mylabel not in np.unique(self.plugin.labeled_volume):
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Label Info')
            msg_box.setText('Label does not exist, please write a proper label')
            msg_box.exec_()
            return

        clusters_dic = {
            'IHC': (self.plugin.IHC, 9),
            'OHC1': (self.plugin.OHC1, 16),
            'OHC2': (self.plugin.OHC2, 37),
            'OHC3': (self.plugin.OHC3, 28)
        }

        mytargetarray, newvalue = clusters_dic[method]
        self.plugin.IHC[self.plugin.labeled_volume == mylabel] = 0
        self.plugin.OHC1[self.plugin.labeled_volume == mylabel] = 0
        self.plugin.OHC2[self.plugin.labeled_volume == mylabel] = 0
        self.plugin.OHC3[self.plugin.labeled_volume == mylabel] = 0
        self.plugin.OHC[self.plugin.labeled_volume == mylabel] = 0
        self.plugin.IHC_OHC[self.plugin.labeled_volume == mylabel] = 0

        mytargetarray[self.plugin.labeled_volume == mylabel] = newvalue
        self.plugin.clustered_cells[self.plugin.labeled_volume == mylabel] = newvalue
        if method in ['OHC1', 'OHC2', 'OHC3']:
            self.plugin.OHC[self.plugin.labeled_volume == mylabel] = 10
            self.plugin.IHC_OHC[self.plugin.labeled_volume == mylabel] = 10
        else:
            self.plugin.IHC_OHC[self.plugin.labeled_volume == mylabel] = 9
        if method == 'IHC':
            self.plugin.IHC[self.plugin.labeled_volume == mylabel] = mylabel
        elif method == 'OHC1':
            self.plugin.OHC1[self.plugin.labeled_volume == mylabel] = mylabel
        elif method == 'OHC2':
            self.plugin.OHC2[self.plugin.labeled_volume == mylabel] = mylabel
        elif method == 'OHC3':
            self.plugin.OHC3[self.plugin.labeled_volume == mylabel] = mylabel

        self.plugin.viewer.layers['Clustered Cells'].data = self.plugin.clustered_cells
        self.plugin.viewer.layers['IHCs vs OHCs'].data = self.plugin.IHC_OHC

        Distance_path = self.plugin.rootfolder + '\\' + self.plugin.filename_base + '\\Distances\\' + 'Physical_distances.csv'
        data = pd.read_csv(Distance_path)
        data['CLass'] = 'Unknown'
        data.loc[data['ID'].isin(np.unique(self.plugin.IHC)), 'CLass'] = 'IHC'
        data.loc[data['ID'].isin(np.unique(self.plugin.OHC1)), 'CLass'] = 'OHC1'
        data.loc[data['ID'].isin(np.unique(self.plugin.OHC2)), 'CLass'] = 'OHC2'
        data.loc[data['ID'].isin(np.unique(self.plugin.OHC3)), 'CLass'] = 'OHC3'
        data.to_csv(Distance_path, index=False)

        save_attributes(self.plugin, self.plugin.pkl_Path)
        print(f"Re-assigning label {mylabel} to {method}")

    def create_reassignclustering_widget(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)  # Horizontal layout

        # Create the dropdown menu for selecting the clustering method
        method_dropdown = QComboBox()
        method_dropdown.addItems(['IHC', 'OHC1', 'OHC2', 'OHC3'])

        # Create the textbox for entering the label number
        label_input = QLineEdit()
        label_input.setPlaceholderText("Enter label number")

        # Create the re-assign button
        reassign_button = QPushButton("Re-assign labels")
        reassign_button.clicked.connect(
            lambda: self.perform_reassign_clustering(method_dropdown.currentText(), label_input.text()))

        # Add widgets to the layout
        layout.addWidget(method_dropdown)
        layout.addWidget(label_input)
        layout.addWidget(reassign_button)

        return widget