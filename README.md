# Rice Paddy Moisture Prediction via Image Color Analysis

[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-iSAI--NLP%202025-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11320486)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](code/requirements.txt)

Official implementation and dataset for our paper:

> **[Developing Prediction Model for Estimating Moisture of Rice Paddy using Correlation Between Color Values of Rice Paddy and Moisture Content](https://ieeexplore.ieee.org/document/11320486)**  
> *Kirati Pothidang, Ritchanon Bunseng, Waiyawat Suansawan*  
> Department of Mathematics, Princess Chulabhorn Science High School Phetchaburi, Thailand  
> Published in *IEEE iSAI-NLP 2025* · [IEEE Xplore](https://ieeexplore.ieee.org/document/11320486)

---

## Overview

Accurate moisture measurement is essential for pricing harvested rice paddy and preserving milling quality. Conventional oven-drying is slow and destructive, while portable moisture meters are costly and drift without frequent calibration.

This project provides a **low-cost, non-destructive computer vision & regression pipeline** that predicts wet-basis moisture content (%) directly from RGB images of unmilled (husked) Thai rice paddy captured in a simple 20 × 20 cm lightbox.

### Pipeline Architecture

```mermaid
flowchart LR
    A["RGB Paddy Image<br/>(4896 × 3264)"] --> B["Border Crop<br/>(586, 0, 4252, 3164)"]
    B --> C["HSV Band-Stop Denoise<br/>Reject Hue ∈ [100, 140]"]
    C --> D["HSV Color-Splitting<br/>Full / Green / Brown"]
    D --> E["Color Feature Extraction<br/>81 features × 3 = 243"]
    E --> F["Pearson Screening<br/>|r| ≥ 0.80, p < 0.01 → 38"]
    F --> G["Z-Score OLS Regression<br/>10-Decile Stratified 85/15"]
    G --> H["Predicted Moisture (%)<br/>+ ANOVA & Diagnostics"]
```

---

## Dataset

All raw paddy images were captured under controlled lighting in a 20 × 20 cm lightbox at **4896 × 3264** resolution using Thai rice paddy samples supplied by Poonsin Thai Ricemill (Suphanburi, Thailand).

| File / Folder | Samples | Moisture Range | Description |
| :--- | :---: | :---: | :--- |
| `data/mapping.csv` | **512** | 9.6% – 30.2% (mean 21.17%) | Ground-truth moisture labels (`0001.jpg`–`0512.jpg`) |
| `data/pictures/` | **512** | — | Raw RGB lightbox paddy images (`0001.jpg`–`0512.jpg`) |
| `data/features/` | **296** | 9.6% – 30.2% (mean 21.08%) | Extracted color feature tables (`All_color.csv`, `Green_color.csv`, `Brown_color.csv`, `Correlated_FULL.xlsx`) |

---

## Results

Evaluated using a 10-decile stratified 85/15 split (`N_train = 251`, `N_test = 45`, `seed = 42`):

| Metric | Value |
| :--- | :---: |
| Total Studied Variables | 243 (81 × 3 segments) |
| Retained Features (`\|r\| ≥ 0.80`, `p < 0.01`) | 38 (36 Green + 2 Full) |
| Top Single Predictor | Green `Entropy_S` (`r = +0.8964`, `p = 6.51e-106`) |
| Held-out **RMSE** | **2.270%** |
| Held-out **MAE** | **1.703%** |
| Held-out **MAPE** | **9.75%** |
| Held-out **R²** | **0.840** |
| Training **R² / Adjusted R²** | 0.901 / 0.884 |
| Regression ANOVA **F(38, 212)** | 51.013 (`p = 7.11e-87`) |
| Regression ANOVA **SS_Total / MS_Res** | 8431.127 / 3.9198 |

### Actual vs. Predicted Moisture

![Actual vs. Predicted Moisture](outputs/actual_vs_predicted.png)

> **Multicollinearity Note:** As discussed in Section V.B of the paper, the 38-feature OLS design matrix exhibits severe multicollinearity (in particular, `TGI = (130/3)ExG − (25/3)ExB` identically). While held-out predictions remain accurate (`R² = 0.840`), individual raw regression coefficients should not be interpreted in isolation; future iterations should apply regularization (Ridge/ElasticNet) or parsimonious feature selection.

---

## Repository Structure

```text
rice-moisture-prediction/
├── README.md
├── .gitignore
├── data/
│   ├── mapping.csv                        # Ground-truth moisture labels (0001.jpg–0512.jpg)
│   ├── pictures/                          # Raw lightbox images (0001.jpg–0512.jpg)
│   └── features/                          # Extracted color feature tables
│       ├── All_color.csv
│       ├── Green_color.csv
│       ├── Brown_color.csv
│       └── Correlated_FULL.xlsx
├── code/
│   ├── config.py                          # Hyperparameters, HSV ranges, and feature manifest
│   ├── 01_preprocess_and_extract.py       # Crop, Hue denoise, HSV split, 81×3 color features
│   ├── 02_correlation_select.py           # Pearson r & p-value feature screening
│   ├── 03_train_and_evaluate.py           # Stratified OLS training, ANOVA, and evaluation
│   ├── predict.py                         # Single-image / batch CLI moisture inference
│   ├── run_pipeline.py                    # End-to-end pipeline entrypoint
│   └── requirements.txt
└── outputs/                               # Saved metrics, ANOVA table, equation, plot, and model bundle
```

---

## Quick Start

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

### 2. Train & Evaluate the Model
```bash
python3 code/run_pipeline.py
```
Runs correlation screening and OLS training/evaluation using `data/features/` and writes all artifacts to `outputs/`:
- `pearson_all_243.csv` & `selected_features_summary.csv`
- `regression_anova_table.csv`
- `actual_vs_predicted.png`
- `model_coefficients.csv` & `regression_equation.txt`
- `test_predictions.csv`, `metrics_summary.json`, and `moisture_ols_model.joblib`

### 3. Predict Moisture from Raw Paddy Images
```bash
python3 code/predict.py data/pictures/0001.jpg data/pictures/0294.jpg
```

### 4. Extract Features & Train From Raw Pictures
```bash
python3 code/run_pipeline.py --mode extract_and_train
```
You can also run each stage independently:
```bash
python3 code/01_preprocess_and_extract.py --mapping data/mapping.csv --pictures data/pictures --out-dir data/features
python3 code/02_correlation_select.py --features-dir data/features --out-dir outputs
python3 code/03_train_and_evaluate.py --data outputs/Correlated_FULL.csv --out-dir outputs
```

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{pothidang2025rice,
  author    = {Pothidang, Kirati and Bunseng, Ritchanon and Suansawan, Waiyawat},
  title     = {Developing Prediction Model for Estimating Moisture of Rice Paddy using Correlation Between Color Values of Rice Paddy and Moisture Content},
  booktitle = {Proceedings of the International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP 2025)},
  year      = {2025},
  publisher = {IEEE},
  url       = {https://ieeexplore.ieee.org/document/11320486}
}
```

## Acknowledgments

We thank **Mr. Jirakoon Erbim** (Department of Mathematics, Princess Chulabhorn Science High School Phetchaburi) for his mentorship and **Mr. Nat Satapanapitakkit** of **Poonsin Thai Ricemill** (Suphanburi, Thailand) for providing the rice paddy samples.
