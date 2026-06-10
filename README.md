# CD8+ Lymphocyte Segmentation in Urothelial Carcinoma

Automated segmentation and detection of CD8+ T-cells in fluorescence microscopy images from urothelial carcinoma tissue microarrays (TMA), developed as part of the *Digital Image Processing* course at UPNA (Biomedical Engineering, Autumn 2024).

---

## Overview

This project implements two image processing pipelines:

1. **DAPI nuclear segmentation** — identifies and labels all cell nuclei in DAPI-stained fluorescence images.
2. **CD8+ T-cell detection** — detects CD8+ lymphocytes using the CD8 channel and cross-references them with the DAPI-derived nuclear labelmap.

Performance is evaluated using the **Dice coefficient** (segmentation quality) and **detection error** (absolute difference between detected and ground-truth cell counts).

---

## Results

| Metric | Value |
|---|---|
| Mean Dice coefficient | 0.769 |
| Mean detection error (normalised) | 0.151 |
| Images with zero detection error | 15 / 21 |

---

## Repository Structure

```
.
├── main.py                     # Entry point: runs optimal configuration + sensitivity analysis
├── lymphocytes.py              # Core pipeline: Analyze_set() and segmentation/detection logic
├── public_functions_project.py # Provided evaluation utilities (Dice, detection error, visualisation)
├── requirements.txt            # Python dependencies
├── INFORME_PDI.pdf             # Full technical report (methodology, results, conclusions)
└── README.md
```

> **Note:** The image dataset (`lympho_database_v20241117/`) is **not included** in this repository due to size and data privacy constraints. See [Dataset](#dataset) below.

---

## Methods

### DAPI Segmentation
Bilateral filter → Top-hat → Adaptive thresholding → Morphological closing + hole filling → Distance transform → Fixed thresholding → Erosion → Connected components → Border cleaning

### CD8+ Detection
LoG filter → Top-hat → Adaptive thresholding → Dilation + hole filling → Erosion → Morphological opening reconstruction → AND with DAPI labelmap → Connected components

Key tunable parameters are exposed via `params_seg` and `params_det` dictionaries (see `main.py`).

---

## Dataset

The pipeline expects the following directory layout:

```
lympho_database_v20241117/
└── database_v20241117/
    ├── dapi/           # DAPI fluorescence images
    ├── dapi_mask/      # Ground truth nuclear labelmaps
    ├── cd8/            # CD8 channel images
    ├── cd8_mask/       # Ground truth CD8 labelmaps
    └── out/            # Output directory (created automatically)
```

Images are 256×256 px PNG/TIFF files. The dataset used is not publicly distributed.

---

## Usage

```bash
python main.py
```

`main.py` runs the optimal configuration followed by a sensitivity analysis that varies individual parameters to illustrate their effect on Dice and detection error. Results are printed to stdout.

To run a custom configuration:

```python
import lymphocytes as lymph

config = {
    'imgDAPI_path':  'path/to/dapi',
    'gtNuclei_path': 'path/to/dapi_mask',
    'imgCD8_path':   'path/to/cd8',
    'gtCD8_path':    'path/to/cd8_mask',
    'out_path':      'path/to/out',
    'params_seg': {
        'dFiltro': 7, 'sColor': 60, 'sSpace': 200,
        'se_tophat': 34, 'se_close': 4, 'num_dilate': 1,
        'se_dilation': 3, 'th_dt': 15, 'se_erosion': 3, 'se_cleaning': 6
    },
    'params_det': {
        'se_tophat': 4, 'c_th': -2, 'se_dilate': 4,
        'num_dilate': 1, 'se_erosion': 9, 'num_erosion': 1,
        'se_open': 3, 'num_open': 3
    }
}

results = lymph.Analyze_set(config)
```

---

## Dependencies

| Package | Version |
|---|---|
| opencv-contrib-python | 4.8.0.76 |
| numpy | 1.25.2 |
| scipy | 1.11.2 |
| matplotlib | 3.7.3 |

See `requirements.txt` for the full list.

---

## Authors

- **Paula Crespo**
- **Cristina García**

*Students of Biomedical Engineering — Universidad Pública de Navarra (UPNA)*
