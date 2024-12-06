Orientation Computation
=======================

The **Orientation Computation** feature in **VASCilia** calculates the orientation angles of labeled volumes in cochlear datasets. This tool is essential for analyzing the structural alignment and orientation of cellular components.

Key Features
------------

1. **Orientation Points Calculation**:

   - Identifies the leftmost and rightmost points of each labeled region.
   - Supports two methods:
     - **Height_only**: Uses the height of the lowest points to compute orientation.
     - **Height_Distance**: Combines height and distance from the peak point for a more robust calculation.

2. **Angle Calculation**:

   - Computes the orientation angle between the identified points for each region.
   - Outputs angles in degrees.

3. **Interactive Visualization**:

   - Displays orientation lines, points, and angle annotations in the Napari viewer.
   - Updates lines and angles dynamically in response to user modifications.

4. **CSV Export**:

   - Saves the calculated angles and their associated labels to a CSV file in the `orientation` directory.

Usage Instructions
------------------

1. **Choose Method**:

   - Use the **Compute Orientation** widget to select a method: **Height_only** or **Height_Distance**.

2. **Compute Orientation**:

   - Click the **Compute Orientation** button to calculate the orientation points, lines, and angles.

3. **Review Orientation**:

   - Orientation lines, points, and angles are displayed in the viewer:

     - **Orientation Lines**: Yellow lines connecting the identified points.
     - **Orientation Points**: Magenta points marking key locations.
     - **Angle Annotations**: Lime green text displaying the angles.

4. **Save Results**:

   - Results are automatically saved to the `orientation` directory in the project folder:

     - `angle_annotations.csv` contains the computed angles and their corresponding labels.

Example Workflow
----------------

1. Open the **Orientation Computation** widget in the plugin interface.
2. Select the preferred computation method (Height_only or Height_Distance).
3. Click **Compute Orientation** to calculate and visualize the orientation.
4. Adjust visibility of layers in the viewer as needed.
5. Access saved angle data in the `orientation` directory for further analysis.

Practical Considerations
------------------------

- The **Height_only** method is simpler and suitable for BASE and MIDDLE datasets.
- The **Height_Distance** method provides more accurate results for APEX datasets.
- Adjustments to orientation points and lines are dynamically reflected in the visualization.

Outputs
-------

1. **Visualization Layers**:

   - Orientation Lines: Yellow lines connecting points of interest.
   - Orientation Points: Magenta points marking key regions.
   - Angle Annotations: Text annotations of calculated angles.

2. **Saved Data**:

   - `angle_annotations.csv`: Contains the label IDs and corresponding angles in degrees.

Extending the Functionality
---------------------------
To add or modify functionality, edit the following file:
    - **compute_orientation_action.py**


---

.. image:: _static/orientation.png
   :alt: Orientation Action Preprocessing Example