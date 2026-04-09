# Image Segmentation & 3D Stereo Vision in MATLAB

> SC4061 Computer Vision, Nanyang Technological University
> Choo Yi Ken | U2240710B | College of Computing and Data Science (CCDS)

A MATLAB lab implementing and comparing three document image thresholding algorithms (Otsu's, Niblack's, and Sauvola's), followed by a from-scratch Sum of Squared Differences (SSD) disparity map algorithm for 3D stereo vision.

---

## Overview

| Section | Topic | Input Files |
|---------|-------|-------------|
| 2.1.a | Otsu's Global Thresholding | `document01–04.bmp` + GT |
| 2.1.b | Niblack's Local Thresholding | `document01–04.bmp` + GT |
| 2.1.c | Sauvola's Normalised Thresholding | `document01–04.bmp` + GT |
| 3.1.a | Disparity Map Algorithm (SSD) | — (function definition) |
| 3.1.b–c | Stereo Disparity: Corridor | `corridorl.jpg`, `corridorr.jpg`, `corridor_disp.jpg` |
| 3.1.d | Stereo Disparity: Triclopsi | `triclopsi2l.jpg`, `triclopsi2r.jpg`, `triclopsid.jpg` |

---

## Techniques Implemented

### 2.1 Image Segmentation

The goal is to segment dark foreground text from bright backgrounds in four degraded document images, evaluated against ground truth binary images. Accuracy is measured by pixel-wise XOR comparison:

```
accuracy = (1 - error_pixels / total_pixels) * 100
```

Polarity inversion is automatically handled — the segmentation with fewer misclassified pixels (original vs inverted) is selected.

---

#### 2.1.a — Otsu's Global Thresholding

Finds a single optimal threshold `T` that maximises the inter-class variance between foreground and background:

```
T = argmax_T σ²_between(T)
```

Uses MATLAB's `graythresh` (Otsu) and `imbinarize`. Visualisation includes a pixel intensity histogram with the threshold marked in red, and a 2×2 subplot comparing the original, segmented, ground truth, and difference images.

---

#### 2.1.b — Niblack's Local Thresholding

Computes a spatially-varying threshold at every pixel using the local neighbourhood statistics:

```
threshold(x,y) = mean(x,y) + k * std(x,y)
```

Local mean and standard deviation are computed efficiently using `imboxfilt`. A full grid search is performed over:
- **k values**: −1.5, −1, −0.5, 0, 0.5, 1, 1.5
- **Window sizes**: 11, 51, 101, 301, 501

Results are visualised as a 2D heatmap (k vs window size, coloured by error count), and the best parameter combination's threshold map is displayed as a 3D surface.

---

#### 2.1.c — Sauvola's Normalised Thresholding

An improvement on Niblack that normalises the standard deviation term by a dynamic range constant `R`, reducing over-segmentation in low-contrast regions:

```
threshold(x,y) = mean(x,y) * (1 + k * (std(x,y) / R - 1))
```

A three-parameter grid search is performed over:
- **k values**: −1.5, −1, −0.5, 0, 0.5, 1, 1.5
- **R values**: 64, 96, 128, 160, 192
- **Window sizes**: 11, 51, 101, 301, 501

The full parameter space is visualised as a 3D scatter plot coloured by error count, with the best-performing combination marked as a red star.

---

### 3.1 3D Stereo Vision

#### 3.1.a — Disparity Map Algorithm (SSD)

A custom block-matching disparity algorithm is implemented from scratch. For each pixel in the left image, an 11×11 template is matched against candidate positions in the right image within a maximum disparity window of 15 pixels (searching left only, as expected for rectified stereo pairs). The best match is found by minimising the Sum of Squared Differences (SSD):

```
SSD = Σ(I_right - G_left)² = ΣI² + ΣG² - 2·Σ(I·G)
```

Squared image buffers (`img²`) are precomputed for efficiency.

#### 3.1.b–c — Corridor Stereo Pair

The left/right corridor images (`corridorl.jpg`, `corridorr.jpg`) are loaded, converted to grayscale, and passed into the disparity algorithm with an 11×11 template. The resulting disparity map is displayed using a gray colormap with range [−15, 15] and compared visually against the ground truth (`corridor_disp.jpg`).

#### 3.1.d — Triclopsi Stereo Pair

The same pipeline is rerun on a second stereo pair (`triclopsi2l.jpg`, `triclopsi2r.jpg`) and compared against its ground truth (`triclopsid.jpg`), demonstrating the algorithm's generalisability across different scenes.

---

## Requirements

- MATLAB with the **Image Processing Toolbox**

---

## Setup & Usage

1. Place all input images in the same directory as `sc4061lab2.m`:

```
document01.bmp   document01-GT.tiff
document02.bmp   document02-GT.tiff
document03.bmp   document03-GT.tiff
document04.bmp   document04-GT.tiff
corridorl.jpg    corridorr.jpg    corridor_disp.jpg
triclopsi2l.jpg  triclopsi2r.jpg  triclopsid.jpg
```

2. Open `sc4061lab2.m` in MATLAB and run it section by section using **Run Section**, or run the entire script at once.

> **Note:** The Niblack and Sauvola sections perform exhaustive parameter grid searches and may take several minutes to complete depending on image size and hardware.

> **Note:** The disparity map computation (Section 3.1) is computationally intensive due to the pixel-level nested loop. Runtime scales with image resolution and `maxDisparity`.

---

## Author

**Choo Yi Ken** | U2240710B  
College of Computing and Data Science (CCDS)  
Nanyang Technological University
