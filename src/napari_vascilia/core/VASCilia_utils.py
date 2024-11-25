
import os
import numpy as np
from skimage.io import imread
import pickle

"""
Those are utility function that almost all other files needs them

Author: Yasmin Kassim
"""

def display_images(viewer, display, full_stack_raw_images, full_stack_raw_images_trimmed, full_stack_rotated_images, filename_base, loading_name):
    images = []
    red_images = []
    blue_images = []
    if display is None:
        display_path = full_stack_raw_images
    elif display == 1:
        display_path = full_stack_raw_images_trimmed
    elif display == 2:
        display_path = full_stack_rotated_images
    rawim_files = sorted(
        [os.path.join(display_path, f) for f in os.listdir(display_path) if
         f.endswith('.tif')])  # Change '.png' if you're using a different format

    # Read each 2D mask and stack them
    for rawim_file in rawim_files:
        im = imread(rawim_file)
        red_images.append(im[:, :, 0])
        images.append(im[:, :, 1])
        blue_images.append(im[:, :, 2])

    red_3d = np.stack(red_images, axis=-1)
    im_3d = np.stack(images, axis=-1)
    blue_3d = np.stack(blue_images, axis=-1)


    # Add three dummy slices to avoid adding the stacks that has three slices as RGB image
    dummy_slice = np.zeros_like(im_3d[..., 0])
    red_3d = np.concatenate([red_3d, np.stack([dummy_slice] * 3, axis=-1)], axis=-1)
    im_3d = np.concatenate([im_3d, np.stack([dummy_slice] * 3, axis=-1)], axis=-1)
    blue_3d = np.concatenate([blue_3d, np.stack([dummy_slice] * 3, axis=-1)], axis=-1)


    if 'Original Volume' in viewer.layers:
        viewer.layers['Original Volume'].data = im_3d
        viewer.layers['Protein Volume'].data = red_3d
        viewer.layers['Protein Volume2'].data = blue_3d
    else:
        viewer.add_image(im_3d, name='Original Volume', colormap='green', blending='additive')
        viewer.add_image(red_3d, name='Protein Volume', colormap='red', blending='additive')
        viewer.add_image(blue_3d, name='Protein Volume2', colormap='blue', blending='additive')

    # Remove dummy slices
    viewer.layers['Original Volume'].data = im_3d[..., :-3]
    viewer.layers['Protein Volume'].data = red_3d[..., :-3]
    viewer.layers['Protein Volume2'].data = blue_3d[..., :-3]

    viewer.dims.order = (2, 0, 1)
    loading_name.setText(filename_base)


def save_attributes(plugin, filename):
    attributes_to_save = {
        'rootfolder': plugin.rootfolder,
        'wsl_executable': plugin.wsl_executable,
        'model': plugin.model,
        'analysis_stage': plugin.analysis_stage,
        'pkl_Path': plugin.pkl_Path,
        'BUTTON_WIDTH': plugin.BUTTON_WIDTH,
        'BUTTON_HEIGHT': plugin.BUTTON_HEIGHT,
        'filename_base': plugin.filename_base,
        'full_stack_raw_images': plugin.full_stack_raw_images,
        'full_stack_length': plugin.full_stack_length,
        'full_stack_raw_images_trimmed': plugin.full_stack_raw_images_trimmed,
        'full_stack_rotated_images': plugin.full_stack_rotated_images,
        'physical_resolution': plugin.physical_resolution,
        'format': plugin.format,
        'npy_dir': plugin.npy_dir,
        'obj_dir': plugin.obj_dir,
        'start_trim': plugin.start_trim,
        'end_trim': plugin.end_trim,
        'display': plugin.display,
        'labeled_volume': plugin.labeled_volume,
        'filtered_ids': plugin.filtered_ids,
        'num_components': plugin.num_components,
        'delete_allowed': plugin.delete_allowed,
        'id_list_annotation': plugin.id_list_annotation,
        'ID_positions_annotation': plugin.ID_positions_annotation,
        'physical_distances': plugin.physical_distances,
        'start_points_most_updated': plugin.start_points_most_updated,
        'end_points_most_updated': plugin.end_points_most_updated,
        'start_points': plugin.start_points,
        'end_points': plugin.end_points,
        'IDs': plugin.IDs,
        'IDtoPointsMAP': plugin.IDtoPointsMAP,
        'Clustering_state': plugin.clustering,
        'clustered_cells': plugin.clustered_cells,
        'IHC': plugin.IHC,
        'IHC_OHC': plugin.IHC_OHC,
        'OHC': plugin.OHC,
        'OHC1': plugin.OHC1,
        'OHC2': plugin.OHC2,
        'OHC3': plugin.OHC3,
        'gt': plugin.gt,
        'lines_with_z_swapped': plugin.lines_with_z_swapped,
        'text_positions': plugin.text_positions,
        'text_annotations': plugin.text_annotations,
        'id_list': plugin.id_list,
        'orientation': plugin.orientation,
        'scale_factor': plugin.scale_factor,
        'start_points_layer_properties': plugin.start_end_points_properties,
        'rot_angle': plugin.rot_angle
    }
    with open(filename, 'wb') as file:
        pickle.dump(attributes_to_save, file)

def load_attributes(plugin, filename):
    with open(filename, 'rb') as file:
        loaded_attributes = pickle.load(file)

    plugin.rootfolder = loaded_attributes.get('rootfolder', None)
    plugin.wsl_executable = loaded_attributes.get('wsl_executable', None)
    plugin.model = loaded_attributes.get('model', None)
    plugin.analysis_stage = loaded_attributes.get('analysis_stage', None)
    plugin.pkl_Path = loaded_attributes.get('pkl_Path', None)
    plugin.BUTTON_WIDTH = loaded_attributes.get('BUTTON_WIDTH', None)
    plugin.BUTTON_HEIGHT = loaded_attributes.get('BUTTON_HEIGHT', None)
    plugin.filename_base = loaded_attributes.get('filename_base', None)
    plugin.full_stack_raw_images = loaded_attributes.get('full_stack_raw_images', None)
    plugin.full_stack_length = loaded_attributes.get('full_stack_length', None)
    plugin.full_stack_raw_images_trimmed = loaded_attributes.get('full_stack_raw_images_trimmed', None)
    plugin.full_stack_rotated_images = loaded_attributes.get('full_stack_rotated_images', None)
    plugin.physical_resolution = loaded_attributes.get('physical_resolution', None)
    plugin.format = loaded_attributes.get('format', None)
    plugin.npy_dir = loaded_attributes.get('npy_dir', None)
    plugin.obj_dir = loaded_attributes.get('obj_dir', None)
    plugin.start_trim = loaded_attributes.get('start_trim', None)
    plugin.end_trim = loaded_attributes.get('end_trim', None)
    plugin.display = loaded_attributes.get('display', None)
    plugin.labeled_volume = loaded_attributes.get('labeled_volume', None)
    plugin.filtered_ids = loaded_attributes.get('filtered_ids', None)
    plugin.num_components = loaded_attributes.get('num_components', None)
    plugin.delete_allowed = loaded_attributes.get('delete_allowed', True)
    plugin.id_list_annotation = loaded_attributes.get('id_list_annotation', None)
    plugin.ID_positions_annotation = loaded_attributes.get('ID_positions_annotation', None)
    plugin.physical_distances = loaded_attributes.get('physical_distances', None)
    plugin.start_points_most_updated = loaded_attributes.get('start_points_most_updated', None)
    plugin.end_points_most_updated = loaded_attributes.get('end_points_most_updated', None)
    plugin.start_points = loaded_attributes.get('start_points', None)
    plugin.end_points = loaded_attributes.get('end_points', None)
    plugin.IDs = loaded_attributes.get('IDs', None)
    plugin.IDtoPointsMAP = loaded_attributes.get('IDtoPointsMAP', None)
    plugin.clustering = loaded_attributes.get('Clustering_state', None)
    plugin.clustered_cells = loaded_attributes.get('clustered_cells', None)
    plugin.IHC = loaded_attributes.get('IHC', None)
    plugin.IHC_OHC = loaded_attributes.get('IHC_OHC', None)
    plugin.OHC = loaded_attributes.get('OHC', None)
    plugin.OHC1 = loaded_attributes.get('OHC1', None)
    plugin.OHC2 = loaded_attributes.get('OHC2', None)
    plugin.OHC3 = loaded_attributes.get('OHC3', None)
    plugin.gt = loaded_attributes.get('gt', None)
    plugin.id_list = loaded_attributes.get('id_list', None)
    plugin.orientation = loaded_attributes.get('orientation', None)
    plugin.lines_with_z_swapped = loaded_attributes.get('lines_with_z_swapped', None)
    plugin.text_positions = loaded_attributes.get('text_positions', None)
    plugin.text_annotations = loaded_attributes.get('text_annotations', None)
    plugin.scale_factor = loaded_attributes.get('scale_factor', 1)
    plugin.start_end_points_properties = loaded_attributes.get('start_points_layer_properties', {})
    plugin.rot_angle = loaded_attributes.get('rot_angle', 0)