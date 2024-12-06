Signal Computation
==================

The **Compute Fluorescense Intensity** feature in **VASCilia** allows users to compute and visualize signal intensities for labeled volumes. This tool facilitates the analysis of protein responses and other cellular signals within cochlear datasets, supporting downstream quantitative analyses.

Key Functionalities
-------------------

1. **Localized Dilation**:
   - Performs localized binary dilation for labeled regions to enhance the signal representation.

2. **Intensity Calculation**:
   - Computes **mean intensity** and **total intensity** for each region, with background subtraction for more accurate measurements.

3. **Visualization**:
   - Generates bar plots for both mean and total intensity for each stereocilia bundle.
   - Saves plots as `.png` files for reporting and documentation.

4. **CSV Export**:
   - Exports computed intensities (mean and total) for each region into a CSV file for further analysis.

5. **Support for Multiple Cell Types**:
   - Computes signal responses for:

     - **All cells**.
     - **Inner Hair Cells (IHCs)**.
     - **Outer Hair Cells (OHCs)**, including subtypes OHC1, OHC2, and OHC3.

Workflow
--------

1. **Preprocessing**:
   - Loads and processes the 3D labeled volume.
   - Applies localized binary dilation within the bounding box of each labeled region.

2. **Signal Intensity Computation**:
   - Reads intensity images from the specified signal channel.
   - Subtracts background intensity and computes the mean and total intensity for each labeled region.

3. **Visualization**:
   - Plots bar charts for mean and total intensity for all regions and saves them to respective directories.

4. **File Management**:
   - Outputs intensity data and plots into a structured directory under `Protein_responce/`.

Directory Structure
-------------------

Generated results are saved in the following directory structure:

.. code-block:: bash

   Protein_responce/
   ├── Allcells/
   │   ├── region_intensities.csv
   │   ├── mean_intensity_per_cell.png
   │   └── total_intensity_per_cell.png
   ├── IHCs/
   ├── OHCs/
   ├── OHC1/
   ├── OHC2/
   └── OHC3/

Usage Instructions
------------------

1. **Compute Signals**:
   - Click the **Compute Fluorescense Intensity** button in the plugin interface.
   - Review the generated plots and CSV files for intensity measurements.

2. **Analyze Outputs**:
   - Use the exported CSV files and plots for further analysis or reporting.

Practical Considerations
------------------------

- **Background Subtraction**:
  - Background intensity is computed using unlabeled regions (label = 0) and subtracted from each region's intensity.

- **Custom Signal Channels**:
  - Signal intensity can be calculated from any specified channel using the "signal_intensity_channel" in the config.json file inside .napari-vascilia in your system path.

    "signal_intensity_channel": 0  for calculating the first channel intensity
    "signal_intensity_channel": 1  for calculating the second channel intensity
    "signal_intensity_channel": 2  for calculating the third channel intensity

- **Visualization Colors**:
  - Bar plots use distinct colors for each cell type for clarity:

    - **All cells**: Magenta.
    - **IHCs**: Yellow.
    - **OHCs**: Red.
    - **OHC1**: Skyblue.
    - **OHC2**: Light green.
    - **OHC3**: Thistle.


Output Examples
---------------

1. **Mean Intensity**:
   - A bar chart showing the mean intensity for each region, with a clear distinction of cell types.

2. **Total Intensity**:
   - A bar chart showing the total signal intensity for each region, normalized for comparison.

3. **Bar Plots**:
   - Two bar plots for each cell type


Extending the Functionality
---------------------------
To add or modify functionality, edit the following file:
    - **compute_signal_action.py**

---

.. image:: _static/intensity.png
   :alt: intensity Action Preprocessing Example
