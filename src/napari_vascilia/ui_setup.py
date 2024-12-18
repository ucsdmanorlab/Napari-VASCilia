import os
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt, QSize
import importlib.resources
# from core.open_cochlea_action import OpenCochleaAction
# from core.upload_cochlea_action import UploadCochleaAction
# from core.trim_cochlea_action import TrimCochleaAction
# from core.rotate_cochlea_action import RotateCochleaAction
# from core.segment_cochlea_action import SegmentCochleaAction
# from core.visualize_track_action import VisualizeTrackAction
# from core.delete_action import DeleteAction
# from core.calculate_measurements import CalculateMeasurementsAction
# from core.calculate_distance import CalculateDistanceAction
# from core.save_distance import SaveDistanceAction
# from core.identify_celltype_action import CellClusteringAction
# from core.compute_signal_action import ComputeSignalAction
# from core.predict_tonotopic_region import PredictRegionAction
# from core.compute_orientation_action import ComputeOrientationAction
# from core.commute_training_action import commutetraining
# from core.reset_exit_action import reset_exit
# from core.batch_action import BatchCochleaAction
# from core.Process_multiple_stacks import BatchCochleaAction_multi_stacks
from .core.open_cochlea_action import OpenCochleaAction
from .core.upload_cochlea_action import UploadCochleaAction
from .core.trim_cochlea_action import TrimCochleaAction
from .core.rotate_cochlea_action import RotateCochleaAction
from .core.segment_cochlea_action import SegmentCochleaAction
from .core.visualize_track_action import VisualizeTrackAction
from .core.delete_action import DeleteAction
from .core.calculate_measurements import CalculateMeasurementsAction
from .core.calculate_distance import CalculateDistanceAction
from .core.save_distance import SaveDistanceAction
from .core.identify_celltype_action import CellClusteringAction
from .core.compute_signal_action import ComputeSignalAction
from .core.predict_tonotopic_region import PredictRegionAction
from .core.compute_orientation_action import ComputeOrientationAction
from .core.commute_training_action import commutetraining
from .core.reset_exit_action import reset_exit
from .core.batch_action import BatchCochleaAction
from .core.Process_multiple_stacks import BatchCochleaAction_multi_stacks

def create_ui(plugin):
    print("create_ui called")  # Debugging print statement
    container = QWidget()
    layout = QVBoxLayout(container)

    layout.setContentsMargins(0, 0, 0, 0)    #del (setup)
    script_dir = os.path.dirname(os.path.abspath(__file__))  #del (setup)
    logo_path = os.path.join(script_dir, 'assets', 'VASCilia_logo1.png')   #del (setup)
    logo_pixmap = QPixmap(str(logo_path))
    # with importlib.resources.path('napari_vascilia.assets', 'VASCilia_logo1.png') as logo_path:  # erturn this (setup)
    #     logo_pixmap = QPixmap(str(logo_path))           # erturn this (setup)
    logo_size = QSize(125, 75)
    scaled_logo_pixmap = logo_pixmap.scaled(logo_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    logo_label = QLabel()
    logo_label.setPixmap(scaled_logo_pixmap)
    logo_label.setAlignment(Qt.AlignCenter)
    logo_label.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(logo_label)

    buttons_info = [
        ("Open Cochlea Datasets (CZI,LIF,TIF) and Preprocess", lambda: OpenCochleaAction(plugin, batch = 0, batch_file_path = 'None').execute()),
        ("Upload Processed Stack", lambda: UploadCochleaAction(plugin).execute()),
        ("Trim Full Stack", lambda: TrimCochleaAction(plugin).execute()),
        ("Rotate", lambda: RotateCochleaAction(plugin).execute()),
        ("Segment with 3DBundleSeg", lambda: SegmentCochleaAction(plugin).execute()),
        ("Reconstruct and Visualize", lambda: VisualizeTrackAction(plugin).execute())

    ]

    for text, func in buttons_info:
        button = QPushButton(text)
        button.clicked.connect(func)
        button.setMinimumSize(plugin.BUTTON_WIDTH, plugin.BUTTON_HEIGHT)
        layout.addWidget(button)

    delete_action = DeleteAction(plugin)
    delete_widget = delete_action.create_filter_component_widget()
    layout.addWidget(delete_widget.native)

    plugin.distance_action = CalculateDistanceAction(plugin)
    buttons_info = [
        ("Calculate Measurements", lambda: CalculateMeasurementsAction(plugin).execute()),
        ("Calculate Bundle Height", lambda: plugin.distance_action.execute()),
        ("Save Bundle Height", lambda: SaveDistanceAction(plugin).execute())
    ]

    for text, func in buttons_info:
        button = QPushButton(text)
        button.clicked.connect(func)
        button.setMinimumSize(plugin.BUTTON_WIDTH, plugin.BUTTON_HEIGHT)
        layout.addWidget(button)

    clustering_action = CellClusteringAction(plugin)
    plugin.clustering_widget_choice = clustering_action.create_clustering_widget()
    layout.addWidget(plugin.clustering_widget_choice.native)

    buttons_info = [
        ("Find IHCs and OHCs", lambda: CellClusteringAction(plugin).find_IHC_OHC())
    ]

    for text, func in buttons_info:
        button = QPushButton(text)
        button.clicked.connect(func)
        button.setMinimumSize(plugin.BUTTON_WIDTH, plugin.BUTTON_HEIGHT)
        layout.addWidget(button)

    reassign_clustering_action = CellClusteringAction(plugin)
    reassign_clustering_widget = reassign_clustering_action.create_reassignclustering_widget()
    layout.addWidget(reassign_clustering_widget)

    buttons_info = [
        ("Compute Fluorescence Intensity", lambda: ComputeSignalAction(plugin).compute_protein_intensity()),
        ("Predict region", lambda: PredictRegionAction(plugin).predict_region())
    ]


    for text, func in buttons_info:
        button = QPushButton(text)
        button.clicked.connect(func)
        button.setMinimumSize(plugin.BUTTON_WIDTH, plugin.BUTTON_HEIGHT)
        layout.addWidget(button)

    plugin.orientation_action = ComputeOrientationAction(plugin)
    plugin.orientation_widget_choice = plugin.orientation_action.create_orientation_widget()
    layout.addWidget(plugin.orientation_widget_choice.native)

    separator_label = QLabel("------- Batch Processing Section ------")
    separator_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(separator_label)

    buttons_info = [
    ("Current Stack Batch Processing", lambda: BatchCochleaAction(plugin).execute()),
    ("Multi Stack Batch Processing", lambda: BatchCochleaAction_multi_stacks(plugin).execute())
    ]

    for text, func in buttons_info:
        button = QPushButton(text)
        button.clicked.connect(func)
        button.setMinimumSize(plugin.BUTTON_WIDTH, plugin.BUTTON_HEIGHT)
        layout.addWidget(button)


    separator_label = QLabel("------- Training Section ------")
    separator_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(separator_label)

    buttons_info = [
        ("Create/Save Ground Truth", lambda: commutetraining(plugin).creategt()),
        ("Generate Ground Truth Masks", lambda: commutetraining(plugin).savemasks()),
        ("Display Stored Ground Truth", lambda: commutetraining(plugin).display_stored_gt()),
        ("Copy Segmentation Masks to Ground Truth", lambda: commutetraining(plugin).copymasks()),
        ("Move Ground Truth to Training Folder", lambda: commutetraining(plugin).move_gt()),
        ("Check Training Data", lambda: commutetraining(plugin).check_training_data()),
        ("Train New Model for 3DBundleSeg", lambda: commutetraining(plugin).train_cilia()),
        ("Reset VASCilia", lambda: reset_exit(plugin).reset_button()),
        ("Exit VASCilia", lambda: reset_exit(plugin).exit_button())
    ]

    for text, func in buttons_info:
        button = QPushButton(text)
        button.clicked.connect(func)
        button.setMinimumSize(plugin.BUTTON_WIDTH, plugin.BUTTON_HEIGHT)
        layout.addWidget(button)

    plugin.loading_label = QLabel("")
    layout.addWidget(plugin.loading_label)

    plugin.loading_name = QLabel("")
    layout.addWidget(plugin.loading_name)

    print("create_ui completed")  # Debugging print statement

    return container
