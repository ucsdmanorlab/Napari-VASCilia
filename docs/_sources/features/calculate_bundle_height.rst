Calculating and Saving length for Bundle Heights
====================================================

The **Distance Calculation and Saving** feature in VASCilia allows for precise measurements of distances between labeled points in 3D cochlear datasets. This functionality supports comprehensive morphological analysis and downstream applications.

Key Features
------------

### **Bundle Height Calculation**
- Calculates length between peak and base points for each labeled region in 3D space.
- Utilizes physical resolution settings to compute real-world distances.

**Steps in Calculation**:
1. Identifies the **Peak Point**:

   - The highest point of a labeled structure along the z-axis.

2. Identifies the **Base Point**:

   - The centroid or bottom-most point in the structure.
3. Erosion Filtering:
   - Applies binary erosion to refine the structure.
4. Computes Distances:
   - Uses the Euclidean formula adjusted by the dataset's physical resolution.

### **Length Saving**
- Uses the physical resolution values (already saved when the opens the image).
- Calculates distances in micrometers and saves them in a CSV file.
- Outputs stored in a `Distances` folder within the dataset directory.

**Saved Data**:
- CSV file: Contains labeled IDs and their corresponding distances.
- Updated Points: Saves updated start and end points for further analysis.

Usage Instructions
------------------

### **Length Calculation**
1. **Initiate Calculation**:

   - Click the **Calculate Bundle Height** button in the plugin.

2. **Monitor Progress**:

   - A progress bar indicates the calculation status.
3. **Output**:

   - Start and end points are visualized in Napari with labels.

### **Length Saving**
1. **Save Results**:

   - Click the **Save Bundle Height** button to save calculated distances.

2. **Access Output**:

   - Navigate to the `Distances` folder for results.


### **Visualization**
- Peak and base points are displayed with colored annotations:

  - **Red** for peak points.
  - **Green** for base points.
  - **Cyan lines** connecting the points.

Example Workflow
----------------

1. Click **Calculate Bundle Height** to compute distances between peak and base points.
2. Review the calculated points:
   - **Peak Points (Red)** and **Base Points (Green)** are adjustable.
   - Listeners are set up to allow users to interactively move these points in the Napari viewer.
3. Note: Adjustments to the points will not update the distances in the CSV file automatically.
4. Click **Save Bundle Height** to save the new measurements and update the CSV file.
5. Use the saved data for downstream analysis or reporting.


Extending the Functionality
---------------------------
To add or modify functionality, edit the following files:

    - **calculate_distance.py** and
    - **save_distance.py**


---

.. image:: _static/distance1.png
   :alt: Length Action1 Preprocessing Example

---

.. image:: _static/distance2.png
   :alt: Length Action2 Preprocessing Example

---

.. image:: _static/distance3.png
   :alt: Length Action3 Preprocessing Example

