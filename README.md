<div align="center">

# 💙 Financial Stress & Heart Disease Risk Predictor

### Predicting cardiovascular disease risk from economic vulnerability — no lab tests required.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-AUROC%200.813-orange?logo=xgboost)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://financial-stress-health-prediction.streamlit.app)
[![Zenodo](https://img.shields.io/badge/Preprint-Zenodo-1682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.19960062)
[![GitHub](https://img.shields.io/badge/GitHub-HariVykuntapu-181717?logo=github&logoColor=white)](https://github.com/HariVykuntapu)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**[🚀 Launch Live Demo →](https://financial-stress-health-prediction.streamlit.app)** &nbsp;·&nbsp; **[📄 Read the Preprint →](https://doi.org/10.5281/zenodo.19960062)**

---

**Hari Vykuntapu** · MS Artificial Intelligence · Southwest Baptist University

</div>

---

## What This Project Does

Most heart disease prediction models require cholesterol panels, blood pressure readings, and lab results — data that millions of Americans never have because they can't afford to see a doctor.

**This project asks: can income, employment status, and healthcare affordability alone predict cardiovascular disease risk?**

Using CDC BRFSS survey data from 441,000+ Americans, I engineered the **Financial Stress Score (FSS)** and trained an XGBoost classifier that achieves **AUROC 0.8131 and 80% recall** — without a single clinical measurement. The three FSS components collectively account for **31.5% of model gain**, the largest combined contribution of any feature group in the model.

**[→ Try the live prediction tool](https://financial-stress-health-prediction.streamlit.app)**

---

## 📄 Preprint

Published on Zenodo: https://doi.org/10.5281/zenodo.19960062

Vykuntapu, H. (2026). Financial Stress as a Predictor of Cardiovascular Disease Risk: A Machine Learning Analysis of CDC BRFSS Data. Zenodo. https://doi.org/10.5281/zenodo.19960062

---

## The Financial Stress Score

The FSS collapses three CDC survey variables into a single 0–1 number. The weights reflect how directly each component cuts off access to cardiovascular care:

$$FSS = (\text{income\_inverse} \times 0.40) + (\text{medcost\_burden} \times 0.35) + (\text{unemployment\_flag} \times 0.25)$$

| Component | Weight | Source Variable | Encoding |
|---|---|---|---|
| Income (inverted & normalized) | **40%** | `INCOME2` (CDC scale 1–8) | `(8 − INCOME2) / 7` |
| Healthcare cost barrier | **35%** | `MEDCOST` | `1` = couldn't afford to see a doctor |
| Unemployment flag | **25%** | `EMPLOY1` | `1` = actively unemployed |

**Buckets:** Low `[0.00 – 0.33)` · Medium `[0.33 – 0.66)` · High `[0.66 – 1.00]`

---

## What the Data Showed

- **Income gradient is stark.** Walk the income ladder from >$75K down to <$10K and heart disease rates more than double — consistently across every subgroup.
- **Skipping the doctor has a measurable cost.** People who couldn't afford care had notably higher heart disease prevalence. The magnitude in the data was striking.
- **Financial stress is a real predictor.** EMPLOY1, INCOME2, and MEDCOST — the three FSS components — collectively account for **31.5% of XGBoost model gain**, the largest combined contribution of any thematic feature group, outpacing BMI and smoking history.
- **XGBoost achieves AUROC 0.8131 and 80% recall** — prioritising sensitivity, which is appropriate for a cardiovascular screening context where missing a true positive carries higher clinical cost than a false alarm.
- **Honest caveat:** AUROC in the low 0.8s means this won't replace a clinical workup. As a screening signal for people who never make it to the clinic, though, the results are worth taking seriously.

### Model Results

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| Logistic Regression | 70.6% | 0.193 | 0.733 | 0.306 | 0.789 |
| Random Forest | 85.4% | 0.261 | 0.360 | 0.303 | 0.796 |
| **XGBoost (Tuned)** | 69.6% | 0.196 | **0.800** | 0.315 | **0.813** |

> XGBoost uses `scale_pos_weight=10.47` to up-weight the minority class, prioritising recall over accuracy — the right trade-off for a screening tool.

---

## Project Structure

```
financial-stress-health-prediction/
├── data/
│   ├── raw/                          # 2015.csv (excluded from git — too large)
│   └── processed/
│       ├── brfss_cleaned.csv
│       └── brfss_features.csv        # includes FSS column
├── notebooks/
│   ├── 01_EDA.ipynb                  # exploratory analysis, recoding, EDA charts
│   ├── 02_feature_engineering.ipynb  # FSS construction
│   ├── 03_model_training.ipynb       # LR / RF / XGBoost + scale_pos_weight
│   └── 04_results_visuals.ipynb      # SHAP, ROC, confusion matrix
├── models/
│   ├── xgb_best_model.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
├── outputs/
│   ├── eda/                          # 6 EDA chart PNGs
│   └── results/                      # SHAP plots, ROC/PR curves, comparison charts
├── streamlit_app/
│   └── app.py
├── retrain_model.py                  # clean retraining script
├── requirements.txt
└── README.md
```

---

## Run It Locally

```bash
git clone https://github.com/HariVykuntapu/financial-stress-health-prediction.git
cd financial-stress-health-prediction
pip install -r requirements.txt
```

Download `2015.csv` from the [CDC BRFSS 2015 page](https://www.cdc.gov/brfss/annual_data/annual_2015.html)
and place it at `data/raw/2015.csv`. Then run the notebooks in order:

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_EDA.ipynb` | Loads raw data, recodes variables, generates EDA charts |
| 02 | `02_feature_engineering.ipynb` | Builds the FSS, exports processed dataset |
| 03 | `03_model_training.ipynb` | Trains LR / RF / XGBoost, saves model artifacts |
| 04 | `04_results_visuals.ipynb` | SHAP analysis, ROC/PR curves, all result charts |

Then launch the app:

```bash
streamlit run streamlit_app/app.py
```

---

## Dataset

| | |
|---|---|
| Source | CDC Behavioral Risk Factor Surveillance System (BRFSS) |
| Year | 2015 |
| Respondents | ~441,456 |
| Features used | `INCOME2`, `MEDCOST`, `EMPLOY1`, `_BMI5`, `SMOKE100`, `_AGE80`, `SEX`, `DIABETE3` |
| Target | `MICHD` — ever told had coronary heart disease or heart attack |
| Positive class rate | ~8.7% |

> Centers for Disease Control and Prevention. (2015). *Behavioral Risk Factor Surveillance System Survey Data and Documentation*. U.S. Department of Health and Human Services. https://www.cdc.gov/brfss/

---

## Author

<table>
<tr>
<td>

**Hari Vykuntapu**
MS Artificial Intelligence
Southwest Baptist University, United States

</td>
<td>

| | |
|---|---|
| **Email** | hpvykuntapu@gmail.com |
| **LinkedIn** | [linkedin.com/in/harivykuntapu](https://www.linkedin.com/in/harivykuntapu) |
| **GitHub** | [github.com/HariVykuntapu](https://github.com/HariVykuntapu) |
| **Live App** | [financial-stress-health-prediction.streamlit.app](https://financial-stress-health-prediction.streamlit.app) |

</td>
</tr>
</table>

---

## Cite This Work

```bibtex
@misc{vykuntapu2026financial,
  author    = {Hari Vykuntapu},
  title     = {Financial Stress as a Predictor of Cardiovascular Disease Risk:
               A Machine Learning Analysis of CDC BRFSS Data},
  year      = {2026},
  doi       = {10.5281/zenodo.19960062},
  url       = {https://doi.org/10.5281/zenodo.19960062},
  publisher = {Zenodo}
}
```

---

<div align="center">

© 2026 Hari Vykuntapu. All Rights Reserved.

</div>
