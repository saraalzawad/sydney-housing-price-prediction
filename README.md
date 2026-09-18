# 🏠 Sydney Housing Price Prediction
**SIT307 – Machine Learning Mini Project (8.1D)**
*Sara Zawad | Deakin University*

---

## Overview

This project builds a machine learning pipeline to predict residential property prices across three Sydney suburbs: **Parramatta**, **Chatswood**, and **Bondi**.

I generated a synthetic dataset of 120 property sales based on realistic 2022–2024 Sydney market data, explored the data, engineered features, trained three regression models, and deployed a Streamlit web app for real-time predictions.

**Best model: Random Forest — Test R² = 0.647, CV R² = 0.854, MAE ≈ $229k**

---

## Project Structure

```
sydney-housing-price-prediction/
│
├── data/
│   └── sydney_housing.csv               # 120 property records (40 per suburb)
│
├── plots/                               # All EDA and model plots (PNG)
│   ├── plot_price_dist.png
│   ├── plot_price_type.png
│   ├── plot_correlation.png
│   ├── plot_floor_price.png
│   ├── plot_cbd_price.png
│   ├── plot_feature_importance.png
│   ├── plot_model_comparison.png
│   ├── plot_actual_vs_predicted.png
│   └── plot_worst_errors.png
│
├── app/
│   └── app.py                           # Streamlit web app (trains model from CSV)
│
├── generate_data.py                     # Synthetic data generation script
├── SIT307_8.1D_notebook.ipynb          # Main Jupyter notebook
├── SIT307_8.1D_notebook_executed.ipynb # Notebook with all outputs
└── README.md                            # This file
```

---

## The Three Suburbs

| Suburb | Distance to CBD | Median Price (dataset) | Property Mix |
|---|---|---|---|
| Parramatta | ~23 km | $1,149,913 | 53% Apt / 27% House / 20% Townhouse |
| Chatswood | ~10 km | $2,167,488 | 33% Apt / 42% House / 25% Townhouse |
| Bondi | ~7 km | $2,082,091 | 72% Apt / 20% House / 8% Townhouse |

---

## Dataset Features

| Feature | Description |
|---|---|
| `suburb` | Parramatta / Chatswood / Bondi |
| `property_type` | Apartment / Townhouse / House |
| `bedrooms` | Number of bedrooms (1–5) |
| `bathrooms` | Number of bathrooms |
| `parking` | Parking spaces (0–3) |
| `land_size_sqm` | Land area in sqm (0 for apartments) |
| `floor_area_sqm` | Internal floor area in sqm |
| `year_built` | Year of construction |
| `property_age_years` | Age of property in 2024 |
| `sale_date` | Date of sale |
| `sale_price` | Sale price in AUD (target variable) |
| `distance_to_cbd_km` | Approximate distance to Sydney CBD |
| `school_rating` | Local school rating 1–10 |

---

## Models Trained

| Model | Test R² | Test MAE | CV R² (5-fold) |
|---|---|---|---|
| Linear Regression | 0.654 | $456,079 | 0.695 ± 0.122 |
| Decision Tree (depth=6) | 0.543 | $245,262 | 0.837 ± 0.150 |
| **Random Forest (depth=10)** | **0.647** | **$229,401** | **0.854 ± 0.099** |

**Winner: Random Forest** — lowest MAE, highest CV R², and most stable across folds.

---

## Top 3 Most Influential Features

1. **land_size_sqm** (0.424) — bigger land = much higher price, especially for houses
2. **suburb_enc** (0.266) — which suburb you're in is almost as important as land size
3. **distance_to_cbd_km** (0.140) — closer to the CBD = higher price

---

## Running the Project

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit jupyter
```

### 1. Generate data (already included in repo)
```bash
python generate_data.py
```

### 2. Open the notebook
```bash
jupyter notebook SIT307_8.1D_notebook.ipynb
```

### 4. Launch the web app
```bash
python -m streamlit run app/app.py
```
The app opens at http://localhost:8501. The app trains the model automatically from the CSV on startup — no separate model file needed.

---

## Web App

The Streamlit app lets you enter property details and get an instant price prediction with a confidence range.

**Inputs:** suburb, property type, bedrooms, bathrooms, parking, floor area, land size, year built, distance to CBD, school rating

**Output:** Predicted price + low/high confidence range based on model MAE ($229k)

---

## Key Findings

- **Land size dominates:** Feature importance of 0.424 — a bigger block adds hundreds of thousands of AUD
- **Location matters:** Suburb encoding (0.266) and distance to CBD (0.140) together account for ~40% of predictive power
- **Bondi is hard to predict:** The biggest error (49.2%) was a premium Bondi house at $5.6M — the model hasn't seen enough top-end examples
- **Random Forest beats linear models:** Non-linear relationships (e.g. Bondi house premium) can't be captured by a straight line

---

*This project was completed as part of SIT307 Machine Learning at Deakin University.*
*Data is synthetic and for educational purposes only.*
