# numkit/image

[![License: 0BSD](https://img.shields.io/badge/License-0BSD-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)

The **Image Processing Toolbox** for **[NumKit](https://github.com/numkit/numkit)**.

High-performance image processing, filtering, morphology, segmentation, and color space algorithms with MATLAB R2025b parity.

---

## Features

- **Filtering & Restoration**: 2-D/3-D Gaussian, box, median, bilateral, guided, non-local means, Gabor, local Laplacian, Wiener, and custom neighborhood filters (`imfilter`, `fspecial`, `imgaussfilt`, `medfilt2`, `imbilatfilt`, `imguidedfilter`).
- **Morphology**: Structuring elements (`strel`), erosion, dilation, opening, closing, morphological reconstruction (`imreconstruct`), hole filling (`imfill`), skeletonization and thinning (`bwmorph`, `bwmorph3`).
- **Region & Object Analysis**: Connected components (`bwconncomp`, `bwlabel`), region properties (`regionprops`), boundary tracing (`bwboundaries`), Euclidean and geodesic distance transforms (`bwdist`, `bwdistgeodesic`), Canny/Sobel edge detection (`edge`), Hough transform (`hough`, `houghpeaks`, `houghlines`), corner detection (`corner`).
- **Color & Contrast**: RGB/HSV/YCbCr/Lab/XYZ conversions, chromatic adaptation (`chromadapt`), illumination estimation, histogram equalization (`histeq`, `adapthisteq`), Otsu thresholding (`graythresh`, `multithresh`, `imbinarize`).
- **Geometric & Transforms**: 2-D/3-D image resize, rotation, cropping, translation, pyramids, 2-D DCT (`dct2`, `idct2`), normalized 2-D cross-correlation (`normxcorr2`), deconvolution (`deconvwnr`, `deconvreg`), Shepp-Logan phantom.
- **Image Quality & Texture**: SSIM, MS-SSIM, PSNR, MSE, gray-level co-occurrence matrix (`graycomatrix`, `graycoprops`).
- **Image I/O**: Zero-dependency PNG, JPEG, BMP, TGA, PNM, and TIFF/BigTIFF reading and writing (`imread`, `imwrite`, `imfinfo`).

---

## Building as a Dynamic Module

```bash
cmake -B build -S .
cmake --build build --config Release
```

Produces `numkit_image.dll` (Windows) or `libnumkit_image.so` (Linux/macOS).

---

## Usage in NumKit

```matlab
% Loaded automatically on first call to any image function:
h = fspecial('gaussian', 3, 0.5);
I = phantom(64);
J = imfilter(I, h, 'replicate');
BW = edge(J, 'canny');
```

---

## License

This software is released under the **[BSD Zero Clause License (0BSD)](LICENSE)**.
