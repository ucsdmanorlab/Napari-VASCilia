import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

"""
    This class handles the action of predicting the rotation angle.
    It is designed to work withi the rotate_cochlea_action.py

    Author: Yasmin Kassim
"""

class Rotate_AI_prediction:
    def __init__(self, plugin):
        self.plugin = plugin

    def execute(self):
        class DenseNetRotNet(nn.Module):
            def __init__(self, nb_classes=72):
                super(DenseNetRotNet, self).__init__()
                self.densenet = models.densenet121(pretrained=True)
                self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                                         bias=False)  # Modify the input layer for 1 channel (grayscale)
                self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, nb_classes)

            def forward(self, x):
                x = self.densenet(x)
                return x


        # Define the prediction function

        def remove_padding(image):
            # Convert the image to a NumPy array
            image_np = np.array(image)
            # Find rows and columns that are not completely zero
            non_zero_rows = np.any(image_np != 0, axis=1)
            non_zero_cols = np.any(image_np != 0, axis=0)
            # Use these indices to slice the original array
            cropped_image_np = image_np[np.ix_(non_zero_rows, non_zero_cols)]
            # Convert the cropped NumPy array back to a PIL image
            cropped_image = Image.fromarray(cropped_image_np)
            return cropped_image

        def predict_angle(image_path, modelpath):
            model.load_state_dict(torch.load(modelpath, map_location=device))
            model.to(device)
            model.eval()
            prediction = []

            for item in os.listdir(image_path):
                # Load the image
                image = Image.open(os.path.join(image_path, item)).convert('L')
                image = remove_padding(image)
                # Preprocess the image
                image = transform(image)
                image = image.unsqueeze(0)  # Add a batch dimension
                # Move the image to the appropriate device
                image = image.to(device)
                # Make the prediction
                with torch.no_grad():
                    output = model(image)
                    predicted_class = torch.argmax(output, dim=1).item()
                    print(predicted_class)
                    prediction.append(output)

            return prediction

        # Define the transformations as used during training
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load the trained model
        model_path = os.path.join(self.plugin.model_rotation_prediction, 'best_model_all_densenet121.pth')
        model = DenseNetRotNet(nb_classes=72)  # Ensure nb_classes matches the number of classes in your dataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Define the prediction function
        predicted_classes = predict_angle(self.plugin.full_stack_raw_images_trimmed, model_path)
        final_decision = sum(predicted_classes) / len(predicted_classes)
        correctclass = torch.argmax(final_decision, dim=1).item()
        angle = correctclass * 5
        angle_to_rotate = 360 - angle
        print(f'The slide angle is: {angle}')
        print(f'The slide needs to be rotated to: {angle_to_rotate}')

        return angle_to_rotate