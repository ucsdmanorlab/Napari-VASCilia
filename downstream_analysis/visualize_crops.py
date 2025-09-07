"""
VASCilia — Crop Visualization & Slice Export
Author: Yasmin Kassim

Purpose
-------
Given one crop ID (label_id) inside a VASCilia `measurements/crops` folder,
this script:
  1) Opens the crop in napari (native or isotropically resampled Z),
  2) Exports per-slice PNGs (raw, label mask, and color overlay)
     into a per-crop subfolder, e.g. .../visualize_crops/crop6/.

Inputs (you set these at the bottom in the Example usage)
---------------------------------------------------------
- input_dir   : path to the `.../measurements/crops` directory that contains:
                  raw_crop_{ID}.npy    # shape (Z, Y, X, 3)  RGB
                  label_crop_{ID}.npy  # shape (Y, X, Z)     integer labels
                  z_index_dict.json    # metadata (used for visualize_crop)
- label_id    : integer crop ID to load (e.g., 6)
- output_dir  : a parent folder where per-crop folders will be created,
                e.g., .../visualize_crops/. The script will save into
                .../visualize_crops/crop{label_id}/

Outputs
-------
Inside  <output_dir>/crop{label_id}/  you get, for each Z slice:
  z###_raw.png      # contrast-normalized grayscale raw slice
  z###_label.png    # binary label mask with boundaries zeroed
  z###_overlay.png  # overlay of labels on the raw (magenta/yellow/green)

Additionally, napari viewer windows will open for visual inspection:
  - visualize_crop(...)              # native sampling
  - visualize_crop_isotropic(...)    # Z upsampled to ~0.0425 µm for XY/Z isotropy

Color mapping used in overlays
------------------------------
Index→color: 0=transparent, 1=magenta, 2=yellow, 3=green.
Per slice, label IDs are ranked by area: largest→magenta, 2nd→yellow, 3rd→green.

Function summaries
------------------
visualize_crop_isotropic(crop_dir, label_id)
    Loads raw (Z,Y,X,3) and label (Y,X,Z); transposes label to (Z,Y,X),
    converts raw to grayscale, and upsamples Z by (0.11 / 0.0425321) so
    voxels become isotropic. Adds image+labels to napari and shows a scale bar.

visualize_crop(crop_dir, label_id)
    Loads the same volumes at native sampling, aligns label to (Z,Y,X),
    reads the relevant Z range from z_index_dict.json (metadata), and opens
    both layers in napari (no resampling).

save_label_crop_visualizations(label_id, crop_dir, output_dir, opacity=0.5)
    Creates a per-crop folder (crop{label_id}) under output_dir and, for each
    Z slice, writes:
      - raw PNG (contrast-normalized),
      - binary label PNG,
      - overlay PNG (magenta/yellow/green as above).
    The overlay colors are assigned by ranking label IDs by pixel count per slice.

Dependencies
------------
Python 3.9+, numpy, scipy, scikit-image, matplotlib, napari, qtpy, imageio.

Usage
-----
1) Edit the three variables at the bottom:
       label_id   = 6
       input_dir  = r"...\measurements\crops"
       output_dir = r"...\visualize_crops"
   The script will save into:  output_dir / f"crop{label_id}"

2) Run the file. It will:
   - write PNGs into the per-crop folder,
   - open napari in native view (set isotropic=1 to open the isotropic view).


"""

from skimage.color import rgb2gray
import os
import numpy as np
import imageio.v2 as imageio
import napari
from skimage.util import img_as_ubyte
import json
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
import matplotlib.colors as mcolors
from scipy.ndimage import zoom
import matplotlib
matplotlib.use("Agg")  # To allow saving without GUI

def visualize_crop_isotropic(crop_dir: str, label_id: int, px_xy, px_z):
    """
    Load (Z,Y,X,3) raw crop and (Y,X,Z) label crop, make them isotropic by
    upsampling along Z, then display in napari.
    """

    # ---- file paths
    label_path = os.path.join(crop_dir, f"label_crop_{label_id}.npy")
    raw_path   = os.path.join(crop_dir, f"raw_crop_{label_id}.npy")
    meta_path  = os.path.join(crop_dir, "z_index_dict.json")

    # ---- load
    label_crop = np.load(label_path)               # (Y, X, Z)
    raw_crop   = np.load(raw_path)                 # (Z, Y, X, 3)
    with open(meta_path, "r") as f:
        _ = json.load(f)  # kept in case you need z indices later

    # ---- align label to (Z, Y, X)
    label_zyx = np.transpose(label_crop, (2, 0, 1))    # (Z, Y, X)
    gray_zyx = np.stack([rgb2gray(s) for s in raw_crop], axis=0)  # (Z, Y, X)

    # ---- resample Z to isotropic: z_zoom = z_step / xy_pixel
    z_zoom = px_z / px_xy  # ~2.586

    # image: linear interpolation; labels: nearest
    gray_iso  = zoom(gray_zyx,  (z_zoom, 1.0, 1.0), order=1)
    labels_iso = zoom(label_zyx, (z_zoom, 1.0, 1.0), order=0)

    # ---- view (isotropic now; scale optional)
    viewer = napari.Viewer()
    viewer.add_image(gray_iso,  name=f"Raw Crop {label_id}", rendering="mip", colormap="gray")
    viewer.add_labels(labels_iso, name=f"Label Crop {label_id}")
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"

    napari.run()

def visualize_crop(crop_dir, label_id):
    # File paths
    label_path = os.path.join(crop_dir, f"label_crop_{label_id}.npy")
    raw_path = os.path.join(crop_dir, f"raw_crop_{label_id}.npy")
    meta_path = os.path.join(crop_dir, "z_index_dict.json")

    # Load data
    label_crop = np.load(label_path)  # shape: (Y, X, Z)
    raw_crop = np.load(raw_path)     # shape: (Z, Y, X, 3)
    with open(meta_path, 'r') as f:
        z_dict = json.load(f)

    # Align label to raw by transposing
    label_crop_aligned = np.transpose(label_crop, (2, 0, 1))  # (Z, Y, X)

    # Ensure label mask and raw crop match in Z
    meta = z_dict[str(label_id)]
    z_label_start = meta['z_label_start_in_crop']
    z_label_end = meta['z_label_end_in_crop']

    # Optional: Only show the label region if needed
    # label_crop_aligned[z_label_end:] = 0
    # label_crop_aligned[:z_label_start] = 0

    # Launch Napari viewer
    viewer = napari.Viewer()
    gray_crop = np.stack([rgb2gray(frame) for frame in raw_crop])  # shape: (Z, Y, X)
    viewer.add_image(gray_crop, name=f"Raw Crop {label_id}", rendering='mip', colormap='gray')
    viewer.add_labels(label_crop_aligned, name=f"Label Crop {label_id}")

    napari.run()




def save_label_crop_visualizations(
    label_id: int,
    crop_dir: str,
    output_dir: str,
    opacity: float = 0.5
):
    """
    Visualizes and saves 2D and 3D views for a specific labeled crop.
    Raw crop is shown in grayscale and mask overlay in magenta.

    Args:
        label_id (int): The label ID to visualize.
        crop_dir (str): Directory with label_crop_X.npy, raw_crop_X.npy, z_index_dict.json.
        output_dir (str): Where to save the images.
        opacity (float): Opacity of the segmentation overlay [0, 1].
    """

    # Create a custom magenta colormap with transparency
    magenta_cmap = mcolors.ListedColormap([
        (0, 0, 0, 0),  # Background - fully transparent
        (1, 0, 1, opacity)  # Foreground (label > 0) - magenta with desired opacity
    ])

    yellow_cmap = mcolors.ListedColormap([
        (0, 0, 0, 0),  # Background - fully transparent
        (1, 1, 0, opacity)  # Foreground - yellow with desired opacity
    ])

    # Example RGBA for coral red (approx): R=1.0, G=0.4, B=0.4
    soft_red_cmap = mcolors.ListedColormap([
        (0, 0, 0, 0),  # Background - fully transparent
        (1.0, 0.4, 0.4, opacity)  # Foreground - soft red with desired opacity
    ])

    label_path = os.path.join(crop_dir, f"label_crop_{label_id}.npy")
    raw_path = os.path.join(crop_dir, f"raw_crop_{label_id}.npy")
    z_dict_path = os.path.join(crop_dir, "z_index_dict.json")

    if not all(os.path.exists(p) for p in [label_path, raw_path, z_dict_path]):
        raise FileNotFoundError("One or more input files are missing.")

    label_crop = np.load(label_path)              # (Y, X, Z)
    raw_crop_rgb = np.load(raw_path)              # (Z, Y, X, 3)
    with open(z_dict_path, 'r') as f:
        z_info = json.load(f)[str(label_id)]

    # Convert to grayscale
    raw_crop_gray = np.stack([rgb2gray(raw_crop_rgb[z]) for z in range(raw_crop_rgb.shape[0])])  # (Z, Y, X)

    # Save 2D slice visualizations
    for z in range(raw_crop_gray.shape[0]):
        raw_slice = raw_crop_gray[z]  # (Y, X)
        label_slice = label_crop[:, :, z]  # (Y, X)

        # Save raw
        raw_out = os.path.join(output_dir, f"z{z:03d}_raw.png")
        raw_normalized = rescale_intensity(raw_slice, in_range='image', out_range=(0, 255)).astype(np.uint8)
        imageio.imwrite(raw_out, img_as_ubyte(raw_normalized))

        # Save binary label mask with boundaries
        mask_vis = np.uint8((label_slice > 0) * 255)
        boundaries = find_boundaries(label_slice, mode='outer')
        mask_vis[boundaries] = 0
        label_out = os.path.join(output_dir, f"z{z:03d}_label.png")
        imageio.imwrite(label_out, mask_vis)

        # Overlay preparation
        overlay_idx = np.zeros_like(label_slice, dtype=np.uint8)


        labels, counts = np.unique(label_slice[label_slice > 0], return_counts=True)
        if labels.size > 0:
            order = np.argsort(-counts)  # descending by size
            ranked = labels[order]
            color_slots = [1, 2, 3]  # 1=magenta, 2=green, 3=yellow
            for slot, lab in zip(color_slots, ranked[:3]):
                overlay_idx[label_slice == lab] = slot

        # Colormap (0=transparent, 1=magenta, 2=green, 3=yellow)
        magenta_green_yellow = mcolors.ListedColormap([
            (0, 0, 0, 0),
            (1, 0, 1, opacity),
            (1, 1, 0, opacity),  # green
            (0, 1, 0, opacity)
        ])

        overlay_out = os.path.join(output_dir, f"z{z:03d}_overlay.png")

        fig, ax = plt.subplots()
        ax.imshow(raw_slice, cmap='gray')
        ax.imshow(overlay_idx, cmap=magenta_green_yellow, vmin=0, vmax=3)
        ax.axis('off')

        fig.savefig(overlay_out, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


# Example usage:
isotropic = 0
label_id = 6
input_dir = r"C:\Users\Yasmin\Yasmin\Data\Eps8_project\deleteme\Litter19Mouse3APEX-Eps8++Cdh23correctedcontro\measurements\crops"
output_dir = r"C:\Users\Yasmin\Yasmin\Data\Eps8_project\deleteme\Litter19Mouse3APEX-Eps8++Cdh23correctedcontro\measurements\visualize_crops"
px_xy = 0.0425321  # microns
px_z = 0.11  # microns

per_crop_dir = os.path.join(output_dir, f"crop{label_id}")
os.makedirs(per_crop_dir, exist_ok=True)

save_label_crop_visualizations(
    label_id,
    crop_dir= input_dir,
    output_dir = per_crop_dir,
    opacity=0.5
)

if isotropic == 0:
    visualize_crop(input_dir, label_id)
else:
    visualize_crop_isotropic(input_dir, label_id, px_xy, px_z)