Deleting Labels in Cochlea Segmentations
========================================

The **Delete Label** feature in VASCilia allows users to selectively remove labels from cochlear segmentations. This functionality provides flexibility for refining segmentation results and ensures accurate analysis.

Key Features
------------

### **Interactive Label Deletion**

    - Users can input the label number they wish to delete.
    - Provides real-time updates to the segmentation after deletion.

### **Dynamic Updates**

    - Automatically adjusts:
    - **Region properties:** Updates annotations and label positions.
    - **Connected data:** Ensures consistency across all related layers, including `Labeled Image`, `ID Annotations`, and `Clustered Cells`.

### **Compatibility with Analysis Stages**

    - Integrates seamlessly with different stages of analysis:
    - Adjusts clustering and segmentation data when deletion occurs.
    - Updates peak and base points, line connections, and associated mappings.

Usage Instructions
------------------

1. **Access the Delete Label Feature**:
   - Use the `Delete Label` button in the plugin.

2. **Input the Label Number**:
   - Enter the label number to be removed in the interactive input field.

3. **Confirm Deletion**:
   - The plugin will process the deletion and update all relevant data layers.


Extending the Functionality
---------------------------
To add or modify functionality, edit the following files:

    - **delete_action.py**


---

.. image:: _static/delete.png
   :alt: Delete Action Preprocessing Example