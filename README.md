## Dataset Description

This study uses the **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** dataset, which is publicly available on Kaggle:

**Dataset link:**
CBIS-DDSM Breast Cancer Image Dataset (Kaggle)

The CBIS-DDSM dataset is a curated and standardized version of the Digital Database for Screening Mammography (DDSM) and is widely used in breast cancer diagnosis research.

### Dataset Characteristics

* Modality: Mammography
* Image type: Grayscale breast images
* Classes:

  * Benign
  * Malignant
* Annotations:

  * Lesion masks (for selected cases)
  * Image-level diagnostic labels

This dataset supports both **segmentation-based analysis** and **classification-based diagnosis**, making it suitable for the proposed hybrid framework.

---

## Dataset Organization (Used in This Work)

After downloading and extracting the dataset from Kaggle, the images were reorganized into the following structure for experimental consistency:

```
dataset_root/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── masks/                 # Ground-truth lesion masks (when available)
│   ├── image_001.png
│   └── ...
└── labels.csv
```

### `labels.csv` Format

```
filename,label
image_001.png,0
image_002.png,1
```

Where:

* `0` → Benign
* `1` → Malignant

---

## Data Usage Notes

* Original DICOM images were converted to PNG format.
* All images were resized to **224 × 224** pixels.
* CLAHE-based contrast enhancement and intensity normalization were applied prior to segmentation.
* Only image-level labels were used for classification, while available lesion masks were used **solely for segmentation performance evaluation**.

---

## Licensing and Access

The CBIS-DDSM dataset is publicly available for **research and educational purposes** under Kaggle’s data usage policy.
Users must agree to Kaggle’s terms before downloading the dataset.

---

## Reproducibility Statement

The provided code is compatible with the CBIS-DDSM dataset and reproduces the experimental protocol reported in the manuscript, subject to minor numerical variations due to hardware and software environments.
