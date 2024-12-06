Upload Previously Processed Cochlea Data
=========================================

Overview
--------

This feature allows users to upload previously processed cochlear datasets and resume analysis from where it was left off. It seamlessly loads all associated data, including segmented images, ground truth, labels, and analysis stages, into the Napari plugin environment.

Usage Instructions
------------------

1. **Ensure Current Stack is Unloaded:**

   - If a dataset is already loaded, you must reset VASCilia before proceeding to load a new dataset.

2. **Upload File:**

   - Click the **Upload** button and select the `Analysis_state.pkl` file associated with the dataset you wish to load.
   - Ensure the `Analysis_state.pkl` file is in the correct folder structure and matches the original filename for successful loading.

3. **Load Data:**

   - Upon successful upload, the following data is loaded automatically:
     - Raw and processed image stacks.
     - Segmented labels and clustered data (if available).
     - Annotation layers such as ground truth, distance calculations, and orientation points (if previously computed).

4. **Resume Analysis:**

   - Depending on the last analysis stage:
     - Labels and lines connecting peaks and bases will be displayed.
     - Clustered cells and cell types (IHCs vs OHCs) will be visualized.
     - Ground truth masks and orientation data will be available for review and further processing.

Features
--------

- **Layer Management:**

  - The tool manages visibility for all layers, ensuring only relevant data is displayed.

- **Ground Truth and Orientation Layers:**

  - Automatically loads stored ground truth and orientation data, including annotations for angles and centroids.

- **Interactive Updates:**

  - Any adjustments made to the loaded data are immediately reflected in the viewer.

Error Handling
--------------

- **Existing Analysis:**

  - If the current stack is already loaded, the tool will prompt you to reset VASCilia before proceeding.

- **Filename Mismatch:**

  - If the uploaded dataset has been renamed, the system will warn you to revert to the original filename or restart the analysis.

- **Incomplete Dataset:**

  - Missing files or incomplete directory structures may result in errors during loading.

Advanced Features
-----------------

- **Clustered Data Loading:**

  - If clustering was performed previously, the plugin will load clustered cells and IHCs vs OHCs labels.

- **Orientation Annotations:**

  - Displays orientation points, lines, and annotations if orientation calculations were part of the previous analysis.

- **Scale Factor Verification:**

  - Ensures that the correct scale factor is loaded for accurate measurements and analysis.

Notes
-----

- Make sure the file paths and folder structures remain consistent with the original analysis folder to avoid issues.
- To perform further analysis, ensure the loaded dataset is compatible with the current analysis configuration.


Extending the Functionality
---------------------------
To add or modify functionality, edit the following files:
- **upload_cochlea_action.py**


---

.. image:: _static/upload.png
   :alt: upload Action Example