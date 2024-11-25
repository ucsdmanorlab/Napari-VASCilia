import numpy as np
import json
import os
from skimage import io
import matplotlib.pyplot as plt


class Prepare_length_json_Action:
    """
    This class handles the action of preparing and saving distances for labeled volumes.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        """
        Initializes the SaveDistanceAction with a reference to the main plugin.

        Args:
            plugin: The main plugin instance that this action will interact with.
        """
        self.plugin = plugin

    def execute(self):
        """
        Executes the action to save distances for labeled volumes.
        """

        def convert_numpy(obj):
            """Recursively convert numpy types in a JSON structure to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            else:
                return obj

        def prepare_detectron2_data(padding=15):
            # Directory setup for saving cropped images and JSON files
            crops_dir = os.path.join(self.plugin.rootfolder, self.plugin.filename_base, 'crops')

            if not os.path.exists(crops_dir):
                os.makedirs(crops_dir)

            self.plugin.start_points_layer.properties = {
                key: np.array(value) for key, value in self.plugin.start_end_points_properties.items()
            }

            self.plugin.end_points_layer.properties = {
                key: np.array(value) for key, value in self.plugin.start_end_points_properties.items()
            }

            # Load the max projection of the raw image
            raw_image = np.max(self.plugin.viewer.layers['Original Volume'].data, axis=2)
            # label_projection = np.max(self.plugin.labeled_volume, axis=2)
            # binary_label_projection = (label_projection > 0).astype(raw_image.dtype)
            # processed_image = raw_image * binary_label_projection

            # Initialize JSON data structure for Detectron2
            coco_json = {
                "images": [],
                "annotations": [],
                "categories": [
                    {
                        "id": 1,
                        "name": "stereocilia_bundle",
                        "supercategory": "object",
                        "keypoints": ["top", "bottom"],
                        "skeleton": [[0, 1]]
                    }
                ]
            }

            annotation_id = 1  # Counter for annotation IDs
            for label_id in self.plugin.IDs:
                # Find coordinates where the label exists in labeled_volume
                coords = np.where(self.plugin.labeled_volume == label_id)
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])

                # Apply padding for context
                y_min = max(0, y_min - padding)
                y_max = min(raw_image.shape[0], y_max + padding)
                x_min = max(0, x_min - padding)
                x_max = min(raw_image.shape[1], x_max + padding)

                # Crop the region in the raw image
                #cropped_image = raw_image[y_min:y_max, x_min:x_max]
                # Generate the specific label projection
                specific_label_projection = np.max((self.plugin.labeled_volume == label_id).astype(int), axis=2)
                # Ensure dtype compatibility
                specific_label_projection = specific_label_projection.astype(raw_image.dtype)
                # Apply mask to the raw image to get only the label of interest
                processed_image = raw_image * specific_label_projection
                # Crop the processed image to the bounding box
                #cropped_image = processed_image[y_min:y_max, x_min:x_max]
                cropped_image = raw_image[y_min:y_max, x_min:x_max]

                crop_file_name = f"{self.plugin.filename_base}_stereocilia_crop_{label_id}.png"
                crop_path = os.path.join(crops_dir, crop_file_name)
                io.imsave(crop_path, cropped_image)

                # Relative bounding box coordinates with respect to the cropped image
                rel_bbox = [0, 0, x_max - x_min, y_max - y_min]

                # Calculate relative keypoints (top and bottom) with respect to the cropped image
                # Find the index of the label_id in the label_id array
                start_point = np.where(self.plugin.start_points_layer.properties["label_id"] == label_id)[0][0]
                end_point = np.where(self.plugin.end_points_layer.properties["label_id"] == label_id)[0][0]

                if self.plugin.start_points_most_updated is None:
                    rel_top = [
                        self.plugin.start_points[start_point][1] - x_min,
                        self.plugin.start_points[start_point][0] - y_min, 2
                    ]
                else:
                    rel_top = [
                        self.plugin.start_points_most_updated[start_point][1] - x_min,
                        self.plugin.start_points_most_updated[start_point][0] - y_min, 2
                    ]

                if self.plugin.end_points_most_updated is None:
                    rel_bottom = [
                        self.plugin.end_points[end_point][1] - x_min,
                        self.plugin.end_points[end_point][0] - y_min, 2
                    ]
                else:
                    rel_bottom = [
                        self.plugin.end_points_most_updated[end_point][1] - x_min,
                        self.plugin.end_points_most_updated[end_point][0] - y_min, 2
                    ]

                #Save visualization of cropped image with keypoints drawn on it
                # vis_file_name = f"{self.plugin.filename_base}_stereocilia_crop_{label_id}_vis.png"
                # vis_path = os.path.join(crops_dir, vis_file_name)
                # plt.imshow(cropped_image, cmap='gray')
                # plt.scatter([rel_top[0], rel_bottom[0]], [rel_top[1], rel_bottom[1]], c=['red', 'blue'], s=40)
                # plt.text(rel_top[0], rel_top[1], 'Top', color='red', fontsize=8, ha='right')
                # plt.text(rel_bottom[0], rel_bottom[1], 'Bottom', color='blue', fontsize=8, ha='right')
                # plt.axis('off')
                # plt.savefig(vis_path, bbox_inches='tight', pad_inches=0)
                # plt.close()

                # Add cropped image and annotations to the JSON structure
                coco_json["images"].append({
                    "id": label_id,
                    "file_name": crop_file_name,
                    "width": cropped_image.shape[1],
                    "height": cropped_image.shape[0]
                })

                coco_json["annotations"].append({
                    "id": annotation_id,
                    "image_id": label_id,
                    "category_id": 1,
                    "bbox": rel_bbox,
                    "keypoints": rel_top + rel_bottom,
                    "num_keypoints": 2
                })
                annotation_id += 1

            # Save the JSON file for Detectron2 training
            coco_json = convert_numpy(coco_json)
            # Save the JSON file # Yasmin, uncomment this if you want to have the json file for training
            # with open(os.path.join(crops_dir, "annotations.json"), 'w') as f:
            #     json.dump(coco_json, f, indent=4)

        # Run the data preparation function
        prepare_detectron2_data()
