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
    G --> H["Predicted Moisture (%)<br/>+ ANOVA & Fig. 6"]
```

---

## Dataset

All raw paddy images were captured under controlled lighting in a 20 × 20 cm lightbox at **4896 × 3264** resolution using Thai rice paddy samples supplied by Poonsin Thai Ricemill (Suphanburi, Thailand). The core study uses the 296-sample dataset (`data/mapping_paper_296.csv` and `data/extracted_paper_296/`), while `data/mapping.csv` and `data/pictures/` also include the extended 512-sample collection (`0001.jpg`–`0512.jpg`).

| File / Folder | Samples | Moisture Range | Description |
| :--- | :---: | :---: | :--- |
| `data/mapping_paper_296.csv` | **296** | 9.6% – 30.2% (mean 21.08%) | Core 296-sample ground-truth labels (`0001.jpg`–`0296.jpg`) |
| `data/extracted_paper_296/` | **296** | — | Extracted `All_color.csv`, `Green_color.csv`, `Brown_color.csv`, and `Correlated_FULL.xlsx` |
| `data/mapping.csv` | **512** | 9.6% – 30.2% (mean 21.17%) | Extended ground-truth labels (`0001.jpg`–`0512.jpg`) |
| `data/pictures/` | **512** | — | Raw RGB lightbox paddy images (`0001.jpg`–`0512.jpg`) |

---

## Results

Evaluated on the 296-sample dataset using a 10-decile stratified 85/15 split (`N_train = 251`, `N_test = 45`, `seed = 42`):

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

### Actual vs. Predicted Moisture (Fig. 6)

![Actual vs. Predicted Moisture](outputs/paper_296/fig6_actual_vs_predicted.png)

> **Multicollinearity Note:** As discussed in Section V.B of the paper, the 38-feature OLS design matrix exhibits severe multicollinearity (in particular, `TGI = (130/3)ExG − (25/3)ExB` identically). While held-out predictions remain accurate (`R² = 0.840`), individual raw regression coefficients should not be interpreted in isolation; future iterations should apply regularization (Ridge/ElasticNet) or parsimonious feature selection.

---

## Repository Structure

```text
rice-moisture-prediction/
├── README.md
├── .gitignore
├── data/
│   ├── mapping_paper_296.csv              # Core 296-sample moisture labels
│   ├── mapping.csv                        # Extended 512-sample moisture labels
│   ├── pictures/                          # Raw lightbox images (0001.jpg–0512.jpg)
│   └── extracted_paper_296/               # Extracted color tables for N=296
│       ├── All_color.csv
│       ├── Green_color.csv
│       ├── Brown_color.csv
│       └── Correlated_FULL.xlsx
├── code/
│   ├── config.py                          # Hyperparameters, HSV ranges, and feature manifest
│   ├── 01_preprocess_and_extract.py       # Sec. IV.B–D: Crop, Hue denoise, HSV split, 81×3 features
│   ├── 02_correlation_select.py           # Sec. IV.E & V.A: Pearson r & p-value screening
│   ├── 03_train_and_evaluate.py           # Sec. IV.F–G, Table I, Fig. 6: Stratified OLS & ANOVA
│   ├── predict.py                         # Single-image / batch CLI moisture inference
│   ├── run_pipeline.py                    # End-to-end pipeline entrypoint
│   └── requirements.txt
└── outputs/
    └── paper_296/                         # Saved metrics, ANOVA table, equation, Fig. 6, and .joblib model
```

---

## Quick Start

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

### 2. Train & Evaluate the Model (`N = 296`)
```bash
python3 code/run_pipeline.py --mode train
```
Generates all tables, plots, coefficients, operational equation, and model bundle in `outputs/paper_296/`:
- `pearson_all_243.csv` & `selected_features_summary.csv`
- `regression_anova_table.csv` (Table I)
- `fig6_actual_vs_predicted.png` (Fig. 6)
- `model_coefficients.csv` & `regression_equation.txt`
- `test_predictions.csv`, `metrics_summary.json`, and `moisture_ols_model.joblib`

### 3. Predict Moisture from Raw Paddy Images
```bash
python3 code/predict.py data/pictures/0001.jpg data/pictures/0294.jpg
```

### 4. Extract Features & Train From Raw Pictures (`N = 296` or `N = 512`)
```bash
python3 code/run_pipeline.py --mode extract_and_train --dataset 512
```
You can also run each stage independently:
```bash
python3 code/01_preprocess_and_extract.py --mapping data/mapping.csv --pictures data/pictures --out-dir outputs/custom/extracted
python3 code/02_correlation_select.py --features-dir outputs/custom/extracted --out-dir outputs/custom
python3 code/03_train_and_evaluate.py --data outputs/custom/Correlated_FULL.csv --out-dir outputs/custom
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
