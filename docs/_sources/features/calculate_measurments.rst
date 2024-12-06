Calculating Measurements for Labeled Volumes
============================================

The **Calculate Measurements** feature in VASCilia enables the extraction of quantitative data from labeled 3D volumes and their 2D projections. This feature supports detailed analysis of cochlear structures by calculating key morphological properties.

Key Features
------------

### **3D Measurements**
- Calculates 3D properties of labeled regions using `regionprops`.
- Key measurements include:

  - **Volume (voxels):** Total voxel count per label.
  - **Centroid:** Coordinates of the region's center in (z, y, x).
  - **Bounding Box:** Minimum and maximum coordinates (z, y, x).
  - **Solidity:** Ratio of the volume to its convex hull volume.
  - **Extent:** Fraction of the bounding box occupied by the region.
  - **Euler Number:** Topological complexity of the region.

### **2D Measurements**
- Extracts 2D properties from the maximum projection along the z-axis.
- Key measurements include:

  - **Area (pixels):** Total pixel count per label.
  - **Centroid:** Center of the region in (y, x).
  - **Bounding Box:** Minimum and maximum coordinates (y, x).
  - **Orientation:** Angle of the major axis relative to the horizontal axis.
  - **Major and Minor Axis Lengths:** Dimensions of the region’s ellipse approximation.
  - **Eccentricity:** Shape deviation from a circle.
  - **Convex Area:** Area of the convex hull.
  - **Equivalent Diameter:** Diameter of a circle with the same area as the region.

### **Data Storage**
- Saves the extracted measurements in CSV format:

  - **`measurements_3d.csv`** for 3D properties.
  - **`measurements_2d.csv`** for 2D properties.

- Saves the labeled volume as `StereociliaBundle_labeled_volume.npy` for future analysis.

Usage Instructions
------------------

1. **Initiate the Measurement Process**:

   - Click the **Calculate Measurements** button in the plugin interface.

2. **Processing**:

   - The plugin calculates 3D and 2D measurements for all labeled regions.

3. **Access Results**:

   - Processed measurements are saved in the `measurements` folder within the working directory.

4. **Review Saved Data**:

   - Open the saved CSV files or the labeled volume `.npy` file for detailed analysis.

Practical Considerations
------------------------

### **Output Organization**
- Results are stored in a dedicated `measurements` folder under the dataset's root directory.

### **Data Formats**
- CSV files for easy visualization and integration with statistical or visualization tools.
- Numpy `.npy` file for storing labeled volume data.

### **Analysis Compatibility**
- Measurements support downstream tasks, including feature extraction, clustering, and visualization.

Example Workflow
----------------

1. Click the **Calculate Measurements** button.
2. Wait for processing to complete (indicated by progress updates).
3. Navigate to the `measurements` folder to access the CSV files and labeled volume data.
4. Use the measurements for downstream analysis or reporting.


Extending the Functionality
---------------------------
To add or modify functionality, edit the following files:
    - **calculate_measurments.py**


---

.. image:: _static/measurments.png
   :alt: measurments Action Preprocessing Example