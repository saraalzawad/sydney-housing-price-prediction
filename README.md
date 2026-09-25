# 🏠 Sydney Housing Price Prediction
**SIT307 – Machine Learning Mini Project (8.1D)**
*Sara Zawad | Deakin University*

Predicting sale prices of residential properties in three very different Sydney suburbs – **Parramatta**, **Chatswood** and **Bondi** – from a dataset of **real sold properties** collected from public sold listings on domain.com.au, and deploying the model in a Streamlit app.

> **Revision:** the earlier version used a synthetic dataset. It has been replaced by real sold-listing data, and all analysis (Parts 1–4) and the app were rerun on it.

---

## Project structure
```
sydney-housing-price-prediction/
├── data/
│   └── sydney_sold_properties.csv   # the collected dataset (one row per sold property)
├── housing_features.py              # data cleaning & feature engineering (used by notebook and app)
├── SIT307_8.1D_notebook.ipynb       # Parts 1–4 (EDA, feature engineering, models, errors)
├── app/
│   ├── app.py                       # Streamlit app
│   └── model.joblib                 # trained model saved by the notebook
├── plots/                           # figures saved by the notebook
└── README.md
```

## Data
- **Source:** public "Sold" listings on domain.com.au. Every row has the `listing_url` it came from.
- **Size:** 102 sold properties – Parramatta 36, Chatswood 32, Bondi 34 (sold Aug 2025 – Sep 2026).
- **Mix:** 71 apartments, 25 houses, 6 townhouses.
- **Fields:** suburb, address, property type, bedrooms, bathrooms, parking, land size, building size, sold price (target), sold date, sale method, agency, agent description (text), source and URL.
- **Corrections** made during cleaning are recorded in the `notes` column (e.g. impossible floor areas removed, strata land removed from apartments). Listings with the price withheld were excluded.

| Suburb | Median price | Apartments | Houses | Townhouses |
|---|---|---|---|---|
| Parramatta | $673,500 | 27 | 5 | 4 |
| Chatswood | $1,804,000 | 19 | 11 | 2 |
| Bondi | $1,510,000 | 25 | 9 | 0 |

## Method (summary)
1. **EDA:** price distribution (log-transformed), suburb × property type differences, prices over time, IQR outliers.
2. **Expected predictors before feature engineering:** suburb, property type, land size – then checked against the EDA.
3. **Feature engineering:** strata flag, log land size + missing flags, rooms, time trend, auction flag, keyword features from the agent description.
4. **Models:** Ridge regression, Random Forest, Gradient Boosting – all on log price, tuned and evaluated with **nested 5-fold cross-validation** (MAE, RMSE, MAPE, R²).
5. **Error analysis:** five largest out-of-fold errors, error by suburb/type and price band.

## Results (nested 5-fold cross-validation)
| Model | CV MAE | CV MAPE | CV R² |
|---|---|---|---|
| Ridge regression | $446,603 | 24.6% | 0.35 |
| Random Forest | $372,588 | 21.8% | 0.76 |
| **Gradient Boosting** | **$321,364** | **19.8%** | **0.83** |

The largest errors are expensive Bondi houses, where the data has only 9 examples and value depends on beach proximity and views.

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit jupyter

jupyter notebook SIT307_8.1D_notebook.ipynb   # run all cells – saves plots/ and app/model.joblib
python -m streamlit run app/app.py            # open http://localhost:8501
```


## Web app
Enter a property's suburb, type, bedrooms, bathrooms, parking, land size and floor area (optional), sale method, and an optional agent description – or upload a CSV of properties – to get a predicted price with a typical error range (± cross-validated MAPE).

