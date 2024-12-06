Reset or Exit VASCilia
======================

This module provides functionalities to reset or exit the VASCilia environment. It ensures that all necessary states and configurations are cleared or properly saved before exiting.

### Methods

#### `exit_button()`

- **Purpose**: Closes the Napari viewer and clears all associated layers.
- **Usage**:
  - This method is called when the user wants to exit the application.

#### `reset_button()`

- **Purpose**: Resets all plugin configurations, states, and loaded data.
- **Details**:

  - Clears all viewer layers.
  - Resets plugin attributes such as:
    - Analysis stage
    - File paths
    - Loaded volumes
    - Clustering data
    - Measurement and annotation data
  - Resets the state of the UI components like loading labels and text.
  - Prepares the environment for a fresh analysis session.


Extending the Functionality
---------------------------
To add or modify functionality, edit the following files:

    - **reset_exit.py**


---

.. image:: _static/reset_exit.png
   :alt: reset_exit Action Example