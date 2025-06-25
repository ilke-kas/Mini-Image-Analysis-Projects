# 🧠 Image Registration Using Affine Transformation (MATLAB)

This repository implements affine image registration in MATLAB. The goal is to align a floating image (e.g., an anatomical atlas) to a reference image (e.g., a brain scan) using manually selected control points. It supports outlier rejection, blending, and color overlay to evaluate registration quality.

---

## 📁 Folder Structure

image_registration/
├── atlas.tiff # Floating image
├── brain.tiff # Reference image
├── affine.m # Affine registration function
├── Resulting_Images/ # Output folder for result images
├── selectedMovingPointsQ*.mat # Saved moving control points
├── selectedFixedPointsQ*.mat # Saved fixed control points
└── README.md # Project documentation


---

## ✅ Features

- 🖱 Interactive control point selection with `cpselect`
- 📐 Affine transformation with 4 or more control points
- 🔁 Handles overdetermined systems using pseudo-inverse
- 🚫 Optional outlier rejection based on Euclidean distance
- 🔄 Backward warping and bilinear interpolation
- 🖼️ Automatically saves:
  - Transformed image
  - Blended image with reference
  - Color overlay
  - Distance plots (if outlier rejection is enabled)

---

## 🛠 Requirements

- MATLAB (tested with R2022a or later)
- Image Processing Toolbox

---

## 🚀 How to Use

```matlab
clear; clc; close all;

atlas_img = imread("atlas.tiff");     % Floating image
brain_img = imread("brain.tiff");     % Reference image

% Basic affine transformation
affine(brain_img, atlas_img, false, 10, "Q1");

% Overdetermined system with pseudo-inverse
affine(brain_img, atlas_img, false, 10, "Q2");

% With outlier rejection
affine(brain_img, atlas_img, true, 8, "Q6");
```
A GUI will open for each call, where you must select at least 4 matching control points between the two images.

If reject is set to true, the function:

- Calculates distances between corresponding control points

- Computes mean and standard deviation

- Removes the point farthest from the mean if the standard deviation exceeds the threshold

- Displays a distance plot with labels and mean line

All outputs are saved to the Resulting_Images/ folder.

You will find:

✅ Transformed (registered) atlas image

✅ Blended image using imfuse

✅ Color overlay for visual comparison (e.g., green + red)

✅ Distance plot (if outlier rejection was used)

Example filenames:

Atlas Image After Affine with 4 control points Q1.jpg

Colored Overlay.jpg

## 🧪 How It Works

Affine transformation is modeled as:

x' = A0 + A1x + A2y + A3xy
y' = B0 + B1x + B2y + B3xy


- If `n = 4` points: system is solved using **direct matrix inversion**
- If `n > 4`: system is **overdetermined** and solved using **pseudo-inverse**
- Image warping is performed via **backward mapping** (output → input)
- Pixel values are computed using **bilinear interpolation**

