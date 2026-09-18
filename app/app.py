"""
SIT307 8.1D – Sydney Housing Price Prediction
Streamlit Web App
Trains the Random Forest model from the CSV on first run (no pkl needed).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Path to data (works from any working directory) ─────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sydney_housing.csv')

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)

    # Label encode suburb and property_type
    le_suburb = LabelEncoder()
    le_type   = LabelEncoder()
    df['suburb_enc'] = le_suburb.fit_transform(df['suburb'])
    df['type_enc']   = le_type.fit_transform(df['property_type'])

    FEATURES = [
        'suburb_enc', 'type_enc', 'bedrooms', 'bathrooms', 'parking',
        'land_size_sqm', 'floor_area_sqm', 'property_age_years',
        'distance_to_cbd_km', 'school_rating'
    ]

    X = df[FEATURES]
    y = df['sale_price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    mae = np.mean(np.abs(y_test - preds))

    return rf, le_suburb, le_type, FEATURES, mae


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Sydney Housing Price Predictor', page_icon='🏠', layout='centered')

st.title('🏠 Sydney Housing Price Predictor')
st.caption('SIT307 8.1D – Sara Zawad | Deakin University')
st.markdown('Enter property details below to get an instant price estimate.')

with st.spinner('Training model from dataset…'):
    rf, le_suburb, le_type, FEATURES, mae = train_model()

st.success('Model ready ✓')
st.divider()

col1, col2 = st.columns(2)

with col1:
    suburb = st.selectbox('Suburb', ['Bondi', 'Chatswood', 'Parramatta'])
    prop_type = st.selectbox('Property Type', ['Apartment', 'House', 'Townhouse'])
    bedrooms = st.slider('Bedrooms', 1, 5, 3)
    bathrooms = st.slider('Bathrooms', 1, 4, 2)
    parking = st.slider('Parking Spaces', 0, 3, 1)

with col2:
    floor_area = st.number_input('Floor Area (sqm)', min_value=30, max_value=500, value=100, step=5)
    land_size = st.number_input(
        'Land Size (sqm) — 0 for apartments',
        min_value=0, max_value=2000, value=0 if prop_type == 'Apartment' else 400, step=10
    )
    year_built = st.number_input('Year Built', min_value=1950, max_value=2024, value=2005)
    distance_cbd = st.number_input('Distance to CBD (km)', min_value=1.0, max_value=40.0, value=10.0, step=0.5)
    school_rating = st.slider('School Rating (1–10)', 1, 10, 7)

st.divider()

if st.button('Predict Price', type='primary', use_container_width=True):
    suburb_enc = le_suburb.transform([suburb])[0]
    type_enc   = le_type.transform([prop_type])[0]
    age        = 2024 - year_built

    row = pd.DataFrame([[
        suburb_enc, type_enc, bedrooms, bathrooms, parking,
        land_size, floor_area, age, distance_cbd, school_rating
    ]], columns=FEATURES)

    price = rf.predict(row)[0]
    low   = max(0, price - mae)
    high  = price + mae

    st.markdown('### Estimated Price')
    st.metric('Predicted Sale Price', f'${price:,.0f}')

    c1, c2 = st.columns(2)
    c1.metric('Low Estimate', f'${low:,.0f}')
    c2.metric('High Estimate', f'${high:,.0f}')

    st.caption(
        f'Confidence range based on model MAE of ${mae:,.0f}. '
        'This is a synthetic-data model for educational purposes.'
    )
