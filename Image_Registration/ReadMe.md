# 📌 Image Registration Using Similarity Metrics in MATLAB

This project implements and compares image registration techniques using **Normalized Cross-Correlation (NCC)** and **Sum of Squared Errors (SSE)** for evaluating similarity between images. Both **rigid (translation-based)** and **non-rigid (deformable)** registrations are explored.

---

## 📁 Folder Structure

image_registration/
├── Contrast1_new.tif
├── Contrast2_new.tif
├── live_new.tif
├── mask_new.tif
├── myNCC.m
├── RegistrationScript.m
├── Q2_Resulting_Images/
├── Q3_Resulting_Videos/
├── Q4_Resulting_Images/
├── Q4_Resulting_Videos/
├── Q5_Resulting_Images/
├── Q5_Resulting_Videos/
└── README.md

---

## ✅ Requirements

- MATLAB (R2021a or newer recommended)
- Image Processing Toolbox

---

## 🧠 Project Summary

The objective is to register two grayscale images by minimizing misalignment using affine translation and non-linear deformation, evaluated with:

- `myNCC`: Normalized cross-correlation (ideal value: 1)
- `sumsqr`: Sum of squared errors (ideal value: 0)

### Optimization

- `fminsearch` is used to optimize translations
- Grid search over tolerances and step scalings
- Intermediate steps visualized via subtraction and overlay images
- Videos are saved to show optimization progress over iterations

### Registration Techniques

- **Translation-based** via affine transformation
- **Deformation-based** via `imregdemons`

---

## 🧪 Key Experiments

### Q2: Manual Registration

- Applies known affine translation to align `Contrast2_new.tif` with `Contrast1_new.tif`
- Shows before/after results
- Compares similarity values

### Q3: Automatic Optimization

- Uses `fminsearch` to optimize translation based on:
  - Normalized Cross-Correlation (`myNCC`)
  - Sum of Squared Errors (`SSE`)
- Initialization from `(0,0)` and `(1,1)`
- Generates subtraction images and optimization videos

### Q4: Sensitivity Analysis

- Varies:
  - Optimization tolerances (`TolX`, `TolFun`)
  - Step scalings for `fminsearch`
- Compares iterations, convergence, and registration error

### Q5: Best Parameters & Deformable Registration

- Applies best settings from Q4 to register `live_new.tif` and `mask_new.tif`
- Uses `imregdemons` for deformable registration
- Visualizes results using:
  - Fused overlays
  - Quiver plots of displacement fields

---

## 🖼️ Outputs

- **Images**: Subtracted and registered image outputs (`.jpg`)
- **Videos**: Optimization progress (`.avi`)
- **Tables**: Final results including optimal parameters, iterations, and error

---

## 🚀 How to Run

1. Open MATLAB and navigate to the `image_registration/` folder
2. Run `RegistrationScript.m`
3. Ensure `.tif` images are present
4. Results will be saved automatically in subfolders
5. Open `.avi` files to view optimization steps

---

## 🔍 Visual Inspection

- Image overlays created using `imfuse`
- Subtraction images provide qualitative assessment of registration
- Displacement fields plotted for `imregdemons` results
