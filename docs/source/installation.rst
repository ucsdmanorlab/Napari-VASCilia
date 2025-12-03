Installation
============
Video Recording
-----
Please click the link below to watch the installation process:

(https://youtu.be/9snkuhpKN3c)

Check the system requirements, then follow the steps to install **VASCilia**.

System Requirements
-----

- **OS:** Windows 10/11 (64-bit) with **WSL2** and **Ubuntu 20.04 LTS** installed.
- **GPU:** NVIDIA CUDA-capable (≥ 8 GB VRAM).
   The software is very flexible and has been successfully tested on the following GPUs:
   NVIDIA RTX 2070 (8 GB), RTX 2080 Ti (11 GB), RTX 3080 (10 GB), RTX 4070 Ti (12 GB), RTX 4080 (16 GB), and RTX 4090 (24 GB).
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
Before installing **WSL**, please verify that the required Windows features and virtualization settings are properly enabled.

1. **Open the Windows Features panel**

   Press **Windows + R**, type:

   .. code-block:: bash

      optionalfeatures

   and press **Enter**.

   In the Windows Features window that appears, make sure the following boxes are checked:

   - ✅ **Windows Subsystem for Linux**
   - ✅ **Virtual Machine Platform**

   Click **OK** and restart your computer if prompted.

2. **Check that hardware virtualization is enabled**

   - Open **Task Manager → Performance → CPU** and confirm that it shows **Virtualization: Enabled**.
   - If it is disabled, ask your IT department to enable it in the **BIOS/UEFI settings**.

---

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

Important notes: 

1. WSL installation is straightforward with the command above. However, if you encounter issues, they’re likely due to Windows features that need to be enabled. Don’t panic—ask IT for assistance.
2. After completing the installation and rebooting your system, you do **not** need to open the Ubuntu terminal again.  
   All subsequent steps, including package installation and running VASCilia, will be performed through the **Anaconda Prompt** (not the Ubuntu terminal).
   

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

   **Important Notes:**

   1. Place the **models** folder on your **C:** drive.  
   
   2. After installing **VASCilia**, update all related paths in your configuration file to point to the chosen location.

   3. Ensure that your file paths **do not contain spaces or special characters** (underscores are fine).

   4. Keep your path length **reasonably short**, as Python has a character limit for file paths on Windows.


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

**Important Notes:**

   After installing **VASCilia**, remember to update the **'rootfolder'** parameter in the configuration file with the path to your **'processed_data'** directory.  
   This directory stores all processed stacks and can be located anywhere on your computer.

   **Example:**

   In my configuration file:

   .. code-block:: json

      "rootfolder": "C:/Users/Yasmin/Data/vascilia_project/processed_data/"

STEP4: Installing VASCilia
-----
Instructions for Cloning and Installing the Repository
-----

You can set up **VASCilia** by following **Option A** or **Option B**:

**Important Note:**  
If you do **not** already have **Anaconda** installed on your system, please install it first from the official website:  
👉 [https://www.anaconda.com/download](https://www.anaconda.com/download)

Make sure to install the **Windows version** of Anaconda, as all installation steps for VASCilia will be performed through the **Anaconda Prompt**.

Option A: Installing via PyPI
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

Option B: Cloning the Repository
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




STEP5: Edit ``config.json``
-----

From the **Plugins** menu, select **VASCilia UI**. Do **not** start any processing yet.  
On first launch, the plugin automatically creates a configuration file named ``config.json`` in your user configuration directory.

**Important Note**

   To locate the correct ``.napari-vascilia\config.json`` path on your system, open **Command Prompt** and type:

   .. code-block:: bash

      echo %USERPROFILE%

   Follow the path displayed in the output — your ``.napari-vascilia`` folder and ``config.json`` file will be created inside that location.

   Depending on your system setup, the configuration file may be stored in one of the following locations:

   - ``C:\Users\<username>\.napari-vascilia\config.json``
   - ``C:\Windows\System32\config\systemprofile\.napari-vascilia\config.json``

Edit ``config.json`` to update the paths as needed. The folder structure will look like this:

.. code-block:: text

   📁 C:/Users/Username/
   ├── 📁 .napari-vascilia/
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


.. _config-json-parameters:

Configuration reference (``config.json``) — field-by-field guide
================================================================

This section explains **every parameter** in your ``config.json`` and how to change it safely.


Note: VASCilia reads the configuration **at startup**. After editing
   ``config.json``, fully close Napari and relaunch it so changes take effect.

Paths (models, executables, and outputs)
----------------------------------------

- **``rootfolder``** (*string, path*)  
  Working directory for the project. **All intermediate results, plots, and CSV files** will be stored here after processing.

- **``wsl_executable``** (*string, path*)  
  Path to the Windows-side executable used for segmentation/retraining via WSL.

- **``model``** (*string, path*)  
  Folder containing **pretrained 3D instance-segmentation weights**.

- **``model_output_path``** (*string, path*)  
  Folder where **newly fine-tuned models** will be saved.

- **``model_region_prediction``** (*string, path to file*)  
  Checkpoint for the **region-prediction** model.

- **``model_celltype_identification``** (*string, path*)  
  Folder with weights for **cell-type identification** (IHC vs OHC).

- **``ZFT_trim_model``** (*string, path*)  
  Folder with weights for the **Z-focus tracker** model.

- **``rotation_correction_model``** (*string, path*)  
  Folder with weights for **stack orientation correction**.

.. tip::
   You already set up these model/executable paths; **you usually don’t need to change them again**.
   You should place the ``models`` folder in your 'C' Drive

Channel mapping
---------------

- **``green_channel``**, **``red_channel``**, **``blue_channel``** (*integers; zero-based*)  
  Indices for your image channels. Example: if your stacks have only two channels (green and red),
  use ``green_channel: 0``, ``red_channel: 1``, and set ``blue_channel: -1`` to **disable** blue.  
  If you *do* have a third channel, set ``blue_channel: 2``.

Intensity measurement options
-----------------------------

- **``signal_intensity_channel``** (*integer*)  
  Specifies which image channel is used by the **Compute Fluorescence Intensity** function.  
  You can modify this value to generate plots and CSV files for different fluorescence channels,  
  **but you must restart Napari after each change** for it to take effect.

  **Important:** To determine the correct channel number, open the ``raw_images`` folder inside your dataset (within ``processed_data``).  
  Assign **``signal_intensity_channel``** to:  
     - ``0`` → if you want to quantify the signal that appears **red** in the image  
     - ``1`` → if you want to quantify the signal that appears **green** in the image  
     - ``2`` → if you want to quantify the signal that appears **blue** in the image (if present)

- **``subtract_background``** (*boolean; default ``false``*)  
  When ``true``, perform background subtraction before intensity computation.  
  (This is specifically applied by **Compute Fluorescence Intensity**.)

- **``dilate_labels``** (*boolean; default ``false``*)  
  When ``true``, dilate the segmentation labels slightly to capture more boundary signal.  
  (Also applied by **Compute Fluorescence Intensity**.)

Resizing and padding
--------------------

- **``flag_to_upscale``** (*boolean; default ``false``*)  
  If ``true``, make the image larger when it’s small (typically **< 900 pixels** on the long side).

- **``flag_to_downscale``** (*boolean; default ``false``*)  
  If ``true``, make the image smaller when it’s very large (typically **> 2500 pixels** on the long side).

- **``flag_to_pad``** (*boolean; default ``false``*)  
  If ``true``, pad the image canvas when bundles have good scale/shape but the overall frame is small.

- **``resize_dimension``** (*integer; pixels; e.g., ``1500``*)  
  Target size for upscaling or downscaling (applies when either flag above is enabled).

- **``pad_dimension``** (*integer; pixels; e.g., ``2000``*)  
  Target canvas size when padding is enabled.

Acquisition metadata
--------------------

- **``force_manual_resolution``** (*0 or 1; default ``0``*)  
  The plugin reads resolution from the image metadata automatically. If metadata is missing, it prompts you.  
  Set to ``1`` to **force** manual entry of resolution values even if metadata exists.

UI preferences
--------------

- **``button_width``**, **``button_height``** (*integers; pixels; e.g., ``60`` and ``18``*)  
  Width and height of plugin buttons. Increase/decrease if buttons look too small or large on your machine.

Important
---------

- Changes to any of the above **do not take effect until you restart the plugin**
  (fully close Napari and launch it again).

