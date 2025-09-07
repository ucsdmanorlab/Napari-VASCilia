Installation
============

Check the system requirements, then follow the steps to install **VASCilia**.

System Requirements
-----

- **OS:** Windows 10/11 (64-bit) with **WSL2** and **Ubuntu 20.04 LTS** installed.
- **GPU:** NVIDIA CUDA-capable, **GeForce RTX 4070 Ti–class or better** (≥ 12 GB VRAM).
- **RAM:** ≥ 32 GB recommended.
- **Drivers:** Recent NVIDIA driver and CUDA runtime.

Quick guide (what each step does)
---------------------------------

**Step 1 – Set up WSL (Ubuntu 20.04):** Install WSL2, reboot, and verify the GPU inside Ubuntu with ``nvidia-smi``.

**Step 2 – Download trained models:** Get the **VASCilia_trained_models** from Dropbox. You can place the ``models`` folder anywhere; you’ll point to it later in ``config.json``.

**Step 3 – Download a test dataset:** Create ``raw_data`` (put the sample stack here) and ``processed_data`` (results go here). You’ll set ``rootfolder`` to ``processed_data`` in ``config.json``.

**Step 4 – Install VASCilia:** Create a conda env and install either from GitHub (editable) or via PyPI. Launch Napari.

**Step 5 – Generate & edit ``config.json``:** Open **VASCilia UI** once to auto-create the file, then update **all paths** to match your machine.

**Step 6 – Restart Napari:** Close and relaunch so the plugin reads the updated config and start processing your cochlear stacks .



Steps
==========
STEP1: Set Up WSL
-----

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

Important note: WSL installation is straightforward with the command above. However, if you encounter issues, they’re likely due to Windows features that need to be enabled. Don’t panic—ask IT for assistance.
   

STEP2: Download the Deep Learning Trained Models
-----

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

Important note: You can place the models folder anywhere on your computer. After installing VASCilia, update all related paths in your config file to point to your chosen location. 

.. code-block:: json

   {
     "wsl_executable": "C:/Users/Yasmin/....../models/Train_predict_stereocilia_exe/Train_Predict_stereocilia_exe_v2",
     "model": "C:/Users/....../models/seg_model/stereocilia_v7/",
     "model_output_path": "C:/Users/....../models/new_seg_model/stereocilia_v8/",
     "model_region_prediction": "C:/Users/...../models/region_prediction/resnet50_best_checkpoint_resnet50_balancedclass.pth",
     "model_celltype_identification": "C:/Users/......./models/cell_type_identification_model/",
     "ZFT_trim_model": "C:/Users/......./models/ZFT_trim_model/",
     "rotation_correction_model": "C:/Users/....../models/rotation_correction_model/"
   }


STEP3: Download a Test Dataset
-----

1. Download one of our sample datasets to test **VASCilia**:
   `Sample Datasets on Dropbox <https://www.dropbox.com/scl/fo/pg3i39xaf3vtjydh663n9/h?rlkey=agtnxau73vrv3ism0h55eauek&dl=0>`_

2. After downloading, create a folder named `raw_data` and place the dataset inside it. Your folder structure should look like this:

   .. code-block::

      📁 raw_data [Raw data (stacks) is placed here]
      └── 📄 Litter 12 Mouse 4 MIDDLE - delBUNdelCAP_Airyscan Processing.czi

3. Create another folder named `processed_data`. This is where the plugin will store the analysis results.

   .. code-block::

      📁 processed_data [Processed data will be stored here]

Important Note: After installing vascilia, remember to update the 'rootfolder' parameter in the config file with the 'processed_data' path

STEP4: Installing VASCilia
-----
Instructions for Cloning and Installing the Repository
-----

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

      pip install --extra-index-url https://download.pytorch.org/whl/cu113 torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113
      pip install Napari-VASCilia

4. Launch Napari:

   .. code-block:: bash

      napari


STEP5: Edit ``config.json``
-----
From the **Plugins** menu, select **VASCilia UI**. Do not start any processing yet. 
On first launch, the plugin creates a configuration file named ``config.json`` in your user config directory 
(Windows: ``C:\Users\<username>\.napari-vascilia\config.json``)

Edit ``config.json`` to update the paths as needed. The folder structure will look like this:


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


STEP6: Run VASCilia
-----
If you’ve already launched Napari once to generate `.napari-vascilia/config.json` and finished updating that file, restart the plugin by fully closing Napari and relaunching it from the terminal:

.. code-block:: bash

      napari

You can now start processing your cochlear stacks.

**Congratulations! 🎉**
You are now ready to use **VASCilia**. Enjoy working with the plugin!


Optional Feature: Multi-Batch Processing Feature: 
=============================================

The Multi-Batch Processing feature in this package requires an additional csv file: **file_names_for_batch_processing.csv**. Place this file in the same directory as the rootfolder you specify in the config file.
To try an example, a copy of this file is included with our sample datasets for VASCilia:
   `Sample Datasets on Dropbox <https://www.dropbox.com/scl/fo/pg3i39xaf3vtjydh663n9/h?rlkey=agtnxau73vrv3ism0h55eauek&dl=0>`_
Be sure to update the paths in the CSV so they match your computer.



