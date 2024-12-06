Overview
========

Welcome to the **VASCilia** documentation!

VASCilia: Vision Analysis StereoCilia
--------------------------------------
**A Napari Plugin for Deep Learning-Based 3D Analysis of Cochlear Hair Cell Stereocilia Bundles**.


Pipeline Overview
-----------------

The following diagram illustrates the VASCilia pipeline for analyzing cochlear hair cell stereocilia bundles:

.. image:: _static/VASCilia_pipeline2.png
   :alt: VASCilia pipeline diagram




Features
--------
**VASCilia** is a Napari plugin designed to facilitate 3D segmentation and quantification of stereocilia bundles in cochlear hair cells. Equipped with a variety of advanced features, VASCilia serves as a powerful tool for auditory research. Key functionalities include:

- **Open the Stack**: Load and preprocess all frames in an image stack.
- **Upload Processed Stack**: Import previously processed stacks using a pickle file.
- **Z-Focus Tracker**: Automatically or manually select the cellular region of interest.
- **PCPAlignNet**: Automatically or manually align and rotate stereocilia bundle rows to the planar cell polarity (PCP).
- **3D BundleSeg**: Perform 3D instance segmentation for the stereocilia bundles
- **Track and Visualize**: Interactively visualize 3D stacks and segmentation results.
- **Bundle Deletion**: Remove unnecessary regions from the analysis.
- **Tonotopic Region Prediction**: Predict whether the analyzed region belongs to the **BASE**, **MIDDLE**, or **APEX** of the cochlea.
- **Cell Type Identification**: Identify rows corresponding to **IHC**, **OHC1**, **OHC2**, and **OHC3**.
- **Measurement Analysis**: Perform automatic 2D and 3D measurements with ease.
- **Fluorescence Intensity Analysis**: Measure fluorescence intensity for any channel, generating a CSV file and plots for each cell type.
- **3D Bundle Height Calculation**: Calculate the length from the top of the bundle to its base.
- **Bundle Orientation**: Obtain precise bundle orientation measurements.
- **Batch Processing**: Enable high-throughput batch processing for multiple files.
- **Training Section**: Allow other labs to fine-tune the model for their specific needs.

Applications
------------
VASCilia is particularly suited for:

- **Biomedical Cochlear Image Analysis**: Analyze and interpret cochlear hair cell stereocilia bundles.
- **High-Throughput Processing**: Streamline research studies with batch processing.
- **Quantitative Analysis**: Conduct precise measurements of stereocilia bundles for auditory research.

Getting Started
---------------
To get started with VASCilia:
1. Refer to the :doc:`installation` section for setup instructions.
2. Check out the :doc:`quick_start` guide for a simple example of using VASCilia.

For detailed instructions on specific features, refer to the :doc:`features/open_stack` and other feature pages.
