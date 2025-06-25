# Mini-Image-Analysis-Projects
# 🧠 Image Registration Toolbox in MATLAB

This repository contains a comprehensive collection of MATLAB tools for performing image registration using both **manual affine transformation** and **automated similarity metric-based optimization** techniques. The goal is to align grayscale images using various methods such as control point selection, translation-based registration, and deformable registration using similarity metrics.

---


---

## ✅ Requirements

- MATLAB (R2021a or newer recommended)
- Image Processing Toolbox

---

## 🧠 Project Overview

### 🔹 Part 1: Affine Registration with Control Points

This section enables interactive registration of images using selected control points.

#### Features

- Interactive point selection via `cpselect`
- Affine transformation with 4+ control points
- Outlier rejection via Euclidean distance
- Blended overlays and color comparisons
- Distance plots for control point deviation

#### Usage Example

```matlab
atlas = imread("atlas.tiff");
brain = imread("brain.tiff");

% Simple affine
affine(brain, atlas, false, 10, "Q1");

% Overdetermined system
affine(brain, atlas, false, 10, "Q2");

% With outlier rejection
affine(brain, atlas, true, 8, "Q6");
```

#### Output Includes
✅ Transformed image

✅ Color overlay with imfuse

✅ Distance plot (if enabled)

### 🔹 Part 2: Registration with Similarity Metrics

This section registers image pairs using translation and deformable transformation, guided by:

- **Normalized Cross-Correlation (NCC)**
- **Sum of Squared Errors (SSE)**

---

#### 🧰 Key Functions

- `myNCC.m`: Custom implementation of normalized cross-correlation
- `RegistrationScript.m`: Main driver script for all registration experiments

---

#### ⚙️ Optimization Strategy

- Uses `fminsearch` for parameter optimization
- Grid search over `TolX`, `TolFun`, and initial conditions
- Deformable registration using `imregdemons`

---

### 🧪 Experiments Summary

#### 🔧 Q2: Manual Translation

- Manually shifts `Contrast2_new.tif` to align with `Contrast1_new.tif`
- Outputs before/after images and similarity scores

#### 🤖 Q3: Automated Optimization

- Optimizes translation via `fminsearch`
- Metrics: NCC and SSE
- Visual outputs include difference images and registration progress videos

#### 📊 Q4: Sensitivity Analysis

- Analyzes effect of:
  - `TolX`, `TolFun`
  - Step size
- Compares convergence speed and accuracy

#### 🔁 Q5: Deformable Registration

- Uses `imregdemons` on `live_new.tif` and `mask_new.tif`
- Displacement field visualization (quiver plots)
- Fused overlays to show registration effectiveness

---

### 🖼️ Output Artifacts

- **Images**: JPGs of intermediate and final registration states
- **Videos**: AVI files showing optimization iterations
- **Tables**: Parameter settings and resulting similarity scores

---

### 🚀 How to Run

1. Launch MATLAB and set the working directory to `image_registration/`
2. To run affine-based manual registration:
   - Execute the `affine` function with appropriate parameters
3. To run similarity-based optimization and deformable registration:
   - Run `RegistrationScript.m`
4. Output will be saved into respective folders
5. Review `.avi` videos and fused images for results

---

### 🔍 Visual Tools

- `imfuse`: For overlay comparison
- Subtraction images: For error visualization
- Quiver plots: To show deformation vectors (`imregdemons`)
- Distance plots: To evaluate control point spread

---

### 📌 Notes

- Ensure `.tif` files are not corrupted or missing
- You may need to select control points manually using a GUI popup
- Outputs are automatically saved with descriptive filenames for traceability

---

