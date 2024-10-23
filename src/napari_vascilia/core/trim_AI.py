import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os

"""
    This class handles the action of predicting the start and end index of Z-Focus-Tracker.
    It is designed to work with the trim_cochlea_action.py

    Author: Yasmin Kassim
"""

class ImprovedZSOINetModel(nn.Module):
    def __init__(self, nb_classes=3):
        super(ImprovedZSOINetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, nb_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Trim_AI_prediction:
    def __init__(self, plugin):
        self.plugin = plugin

    def execute(self):
        # Function to predict the class of a single image
        def predict_image(image_path, model):
            image = Image.open(os.path.join(self.plugin.full_stack_raw_images, image_path)).convert('L')  # Convert image to grayscale
            image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

            model.eval()  # Ensure the model is in evaluation mode
            with torch.no_grad():
                output = model(image)  # Get the model output
                probabilities = torch.softmax(output, dim=1)  # Apply softmax to get probabilities
                pred_class = probabilities.argmax(dim=1)  # Get the predicted class

                return pred_class.item()

        # Function to predict classes for a given list of image paths
        def predict_images_from_paths(image_paths, model):
            image_paths = os.listdir(image_paths)
            predictions = {}
            for image_path in image_paths:
                predicted_class = predict_image(image_path, model)
                image_name = os.path.basename(image_path)  # Get the filename from the path
                predictions[image_name] = predicted_class
            return predictions

        def find_start_end_indices(sequence):
            """Find the first and last index of the longest continuous block of 1's in the sequence."""
            start_index = None
            end_index = None
            max_length = 0

            current_start = None
            current_length = 0

            for i, value in enumerate(sequence):
                if value == 1:
                    if current_start is None:
                        current_start = i
                    current_length += 1
                else:
                    if current_length >= max_length:
                        max_length = current_length
                        start_index = current_start
                        end_index = i - 1
                    current_start = None
                    current_length = 0

            # Check the last segment
            if current_length > max_length:
                start_index = current_start
                end_index = len(sequence) - 1
                # Handle the case where no 1's are found in the sequence
            if start_index is None and end_index is None:
                return None, None

            return start_index, end_index
        # Define the transformations (these should match the ones used during training)
        transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
        # Load the saved model
        model = ImprovedZSOINetModel(nb_classes=3)
        model.load_state_dict(torch.load(os.path.join(self.plugin.model_ZFT_prediction, 'best_model_ZSOI_ImprovedZSOINetModel_alldata.pth')))
        model.eval()  # Set the model to evaluation mode
        predictions = predict_images_from_paths(self.plugin.full_stack_raw_images, model)
        sequences = []
        for value in predictions.values():
            sequences.append(value)

        start_index, end_index = find_start_end_indices(sequences)

        return start_index, end_index



