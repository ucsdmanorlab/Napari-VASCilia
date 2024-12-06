Installation
============

Follow these steps to install **VASCilia**.

Requirements
------------

.. code-block:: text

   torch==1.12.1+cu113
   torchvision==0.13.1+cu113
   torchaudio==0.12.1+cu113
   segmentation-models-pytorch
   opencv-python
   matplotlib
   imagecodecs
   tifffile
   napari[all]
   scikit-learn
   readlif
   czitools==0.4.1
   czifile
   npe2
   numpy==1.26.4
   colormap==1.1.0

Steps
-----
Set Up WSL
==========

Follow these steps to set up Windows Subsystem for Linux (WSL) with the Ubuntu 20.04 distribution.

1. **Install WSL**:
   1. Open the Command Prompt and install the Ubuntu 20.04 distribution by copying and pasting the following command:
   
      .. code-block:: bash

         wsl --install -d Ubuntu-20.04

      After the setup successfully completes, reboot your computer.

   2. Open Ubuntu by typing "Ubuntu" in the Windows search bar.

2. **Verify Installation**:
   To check if CUDA and the GPU are correctly installed and available, type the following command in the Ubuntu terminal:

   .. code-block:: bash

      nvidia-smi

   This should display information about your GPU and CUDA installation.

STEP 2: Download the Deep Learning Trained Models
=================================================

1. Download the **VASCilia_trained_models** from the following link:
   `VASCilia Trained Models on Dropbox <https://www.dropbox.com/scl/fo/jsvldda8yvma3omfijxxn/ALeDfYUbiOuj69Flbc728rs?rlkey=mtilfz33qiizpul7uyisud5st&st=41kjlbw0&dl=0>`_

   After downloading, you should have a folder structure like this:

   .. code-block::

      📁 models [Trained models]
      ├── 📁 cell_type_identification_model
      │      Contains weights for cell type identification (IHC vs OHC).
      ├── 📁 new_seg_model
      │      Where fine-tuned models will be stored.
      ├── 📁 region_prediction
      │      Contains weights for region prediction.
      ├── 📁 seg_model
      │      Contains weights for the 3D instance segmentation model.
      ├── 📁 Train_predict_stereocilia_exe
      │      Executable for segmentation and retraining the model using WSL.
      ├── 📁 ZFT_trim_model
      │      Contains deep learning model weights for the z-focus tracker algorithm.
      └── 📁 rotation_correction_model
             Contains deep learning model weights for correcting stack orientation.

STEP 3: Download a Test Dataset
===============================

1. Download one of our sample datasets to test **VASCilia**:
   `Sample Datasets on Dropbox <https://www.dropbox.com/scl/fo/pg3i39xaf3vtjydh663n9/h?rlkey=agtnxau73vrv3ism0h55eauek&dl=0>`_

2. After downloading, create a folder named `raw_data` and place the dataset inside it. Your folder structure should look like this:

   .. code-block::

      📁 raw_data [Raw data (stacks) is placed here]
      └── 📄 Litter 12 Mouse 4 MIDDLE - delBUNdelCAP_Airyscan Processing.czi

3. Create another folder named `processed_data`. This is where the plugin will store the analysis results.

   .. code-block::

      📁 processed_data [Processed data will be stored here]


Instructions for Cloning and Installing the Repository
=======================================================

You can set up **VASCilia** by following **Option A** or **Option B**:

Option A: Cloning the Repository
--------------------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/ucsdmanorlab/Napari-VASCilia.git
      cd Napari-VASCilia

2. Create and activate the conda environment:

   .. code-block:: bash

      conda create -y -n napari-VASCilia -c conda-forge python=3.10
      conda activate napari-VASCilia

3. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt
      pip install -e .

4. Launch Napari:

   .. code-block:: bash

      napari

Option B: Installing via PyPI
-----------------------------

1. Create and activate the conda environment:

   .. code-block:: bash

      conda create -y -n napari-VASCilia -c conda-forge python=3.10
      conda activate napari-VASCilia

2. Download the `requirements.txt` file from this repository and ensure it is in your working directory.

3. Install dependencies and VASCilia:

   .. code-block:: bash

      pip install -r requirements.txt
      pip install Napari-VASCilia

4. Launch Napari:

   .. code-block:: bash

      napari



Update the paths in `config.json` as needed. The `config.json` file will be generated upon running the plugin for the first time. The folder structure will look like this:

.. code-block::

   📁 C:/Users/Username/ [Your home folder]
   ├── 📁 .napari-vascilia [Folder path]
   └── 📄 config.json

2. **Update the `config.json` File**:
Edit the `config.json` file to reflect your system’s paths. Replace `/.../` portions with the correct paths for your system. Example:

.. code-block:: json

   {
       "rootfolder": "C:/Users/.../processed_data/",
       "wsl_executable": "C:/Users/.../models/Train_predict_stereocilia_exe/Train_Predict_stereocilia_exe_v2",
       "model": "C:/Users/.../models/seg_model/stereocilia_v7/",
       "model_output_path": "C:/Users/.../models/new_seg_model/stereocilia_v8/",
       "model_region_prediction": "C:/Users/.../models/region_prediction/resnet50_best_checkpoint_resnet50_balancedclass.pth",
       "model_celltype_identification": "C:/Users/.../models/cell_type_identification_model/",
       "ZFT_trim_model": "C:/Users/.../models/ZFT_trim_model/",
       "rotation_correction_model": "C:/Users/.../models/rotation_correction_model/",
       "green_channel": 0,
       "red_channel": 1,
       "blue_channel": -1,
       "signal_intensity_channel": 0,
       "flag_to_resize": false,
       "flag_to_pad": false,
       "resize_dimension": 1200,
       "pad_dimension": 1500,
       "button_width": 100,
       "button_height": 35
   }

3. **Congratulations! 🎉**
You are now ready to use **VASCilia**. Enjoy working with the plugin!


Multi-Batch Processing Feature: Required File
=============================================

The Multi-Batch Processing feature in this package requires an additional file: **track_me_SORT_v3_exe.exe**. This file is not included in the repository or the pip installation due to size constraints.

Download the File
-----------------
You can download the file from the following link:

`Download track_me_SORT_v3_exe.exe <https://www.dropbox.com/scl/fo/sud3ziayvo7efcsbzgrd7/ACeJ6uMjNLcyk7ev0upJREE?rlkey=e6nzvpz8aoebzq4w3o5f339le&st=a9m73egz&dl=0>`_

Instructions
------------

# If You Clone the Repository
1. Download the file from the link above.
2. Place the file in the following directory within the cloned repository:

.. code-block:: python

    src/napari_vascilia/core/


# If You Installed the Package via pip
1. Download the file from the link above.
2. Locate the installation directory for the package. To find the installation path, run the following Python code:

.. code-block:: python

   import napari_vascilia
   print(napari_vascilia.__file__)

3. Place the downloaded file in the following directory:


**Note**: All other features of the package will function as expected without this file. This file is exclusively required for batch processing of multiple files.

