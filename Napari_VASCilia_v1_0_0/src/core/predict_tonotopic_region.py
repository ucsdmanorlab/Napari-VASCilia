import matplotlib
matplotlib.use('Qt5Agg')
# Region classification
import torch
from torchvision import transforms
from torchvision.models import resnet50
import os
from PIL import Image
from torch import nn
from collections import Counter
#-------------- Qui
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QApplication

class PredictRegionAction:

    """
    This class handles the action of region prediction for labeled volumes.
    It is designed to work within a Napari plugin environment.

    Author: Yasmin Kassim
    """

    def __init__(self, plugin):
        self.plugin = plugin

    def predict_region(self):
        def predict_image(image_path, model, transform):
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            return predicted.item()

        def select_images_around_middle(mouse_folder, num_images='all'):
            image_files = [os.path.join(mouse_folder, f) for f in os.listdir(mouse_folder) if f.endswith('.tif')]
            total_images = len(image_files)

            if num_images == 'all' or num_images >= total_images:
                return image_files

            num_images = max(1, min(num_images, total_images))
            if num_images % 2 == 0:
                num_images += 1

            half_window = num_images // 2
            middle_index = total_images // 2

            start_index = max(0, middle_index - half_window)
            end_index = min(total_images, middle_index + half_window + 1)

            if start_index == 0:
                end_index = min(num_images, total_images)
            elif end_index == total_images:
                start_index = max(0, total_images - num_images)

            median_images = image_files[start_index:end_index]

            return median_images

        def evaluate_accuracy_per_mouse(root_dir, model, transform, num_images_for_decision):
            median_images = select_images_around_middle(root_dir, num_images=num_images_for_decision)

            votes = []
            for image_path in median_images:
                prediction = predict_image(image_path, model, transform)
                votes.append(prediction)

            most_common, num_most_common = Counter(votes).most_common(1)[0]
            if most_common == 0:
                predicted_class = 'APEX'
            elif most_common == 1:
                predicted_class = 'BASE'
            elif most_common == 2:
                predicted_class = 'MIDDLE'
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Region Prediction')
            msg_box.setText(f"Region predicted as {predicted_class} ")
            msg_box.exec_()
            print(most_common)

        self.plugin.loading_label.setText("<font color='red'>Processing..., Wait</font>")
        QApplication.processEvents()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.plugin.model_region_prediction)
        model = resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        evaluate_accuracy_per_mouse(self.plugin.full_stack_rotated_images, model, val_transform, num_images_for_decision=13)
        self.plugin.loading_label.setText("")
        QApplication.processEvents()