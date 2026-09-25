"""
SIT307 8.1D – Sydney Housing Price Prediction
Streamlit web app. Loads the model trained in the notebook (app/model.joblib)
and uses the same feature engineering code (housing_features.py).

Run from the project folder:   python -m streamlit run app/app.py
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from housing_features import engineer, SUBURBS, PROPERTY_TYPES, KEYWORDS   # noqa: E402

MODEL_PATH = os.path.join(HERE, 'model.joblib')

st.set_page_config(page_title='Sydney Housing Price Predictor', page_icon='🏠', layout='centered')
st.title('🏠 Sydney Housing Price Predictor')
st.caption('SIT307 8.1D – Sara Zawad | Deakin University')

if not os.path.exists(MODEL_PATH):
    st.error('No trained model found. Run the notebook SIT307_8.1D_notebook.ipynb first – '
             'its last cell saves app/model.joblib.')
    st.stop()


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


try:
    bundle = load_bundle()
except Exception as e:   # e.g. model saved with a different scikit-learn version
    st.error(f'Could not load app/model.joblib ({e}). Re-run the notebook to retrain and save the model.')
    st.stop()
model, feats = bundle['model'], bundle['features']
st.markdown(
    f"Estimate the sale price of a property in **Parramatta, Chatswood or Bondi**. "
    f"Model: **{bundle['model_name']}**, trained on **{bundle['n_properties']} real sold properties** "
    f"({bundle['sale_date_range'][0]} to {bundle['sale_date_range'][1]})."
)

tab_single, tab_upload = st.tabs(['Enter one property', 'Upload a CSV'])


def predict(df_in):
    data = engineer(df_in)
    for c in feats['cat'] + feats['num']:
        if c not in data:
            data[c] = np.nan
    return model.predict(data[feats['cat'] + feats['num']])


with tab_single:
    c1, c2 = st.columns(2)
    with c1:
        suburb = st.selectbox('Suburb', SUBURBS)
        ptype = st.selectbox('Property type', PROPERTY_TYPES)
        beds = st.number_input('Bedrooms', 0, 10, 2)
        baths = st.number_input('Bathrooms', 1, 10, 1)
        parking = st.number_input('Parking spaces', 0, 10, 1)
    with c2:
        land = None
        if ptype != 'Apartment':
            known = st.checkbox('I know the land size', value=(ptype == 'House'))
            if known:
                land = st.number_input('Land size (m²)', 20, 5000, 450, step=10)
        known_b = st.checkbox('I know the internal floor area')
        building = st.number_input('Floor area (m²)', 20, 1000, 90, step=5) if known_b else None
        method = st.selectbox('Sale method', ['Auction', 'Private treaty', 'Unknown'])
        station = None
        if 'distance_to_station_km' in feats['num']:
            station = st.number_input('Walking distance to nearest station (km)', 0.0, 10.0, 1.0, step=0.1)
    desc = st.text_area('Agent description (optional)',
                        placeholder='e.g. Renovated apartment with ocean views, walk to the station…')

    if st.button('Predict price', type='primary', use_container_width=True):
        row = pd.DataFrame([{
            'suburb': suburb, 'property_type': ptype, 'bedrooms': beds, 'bathrooms': baths,
            'parking': parking, 'land_size_sqm': land if land is not None else np.nan,
            'building_size_sqm': building if building is not None else np.nan,
            'sold_date': pd.Timestamp.today().normalize(), 'sale_method': method,
            'distance_to_station_km': station if station is not None else np.nan,
            'description': desc,
        }])
        price = predict(row)[0]
        mape = bundle['cv_mape'] / 100
        st.metric('Predicted sale price', f'${price:,.0f}')
        a, b = st.columns(2)
        a.metric('Typical low', f'${price * (1 - mape):,.0f}')
        b.metric('Typical high', f'${price * (1 + mape):,.0f}')
        found = [k.replace('kw_', '') for k in KEYWORDS
                 if engineer(row)[k].iloc[0] == 1]
        if found:
            st.caption('Description keywords used: ' + ', '.join(found))
        st.caption(f"Range = ± the model's cross-validated average error ({bundle['cv_mape']:.1f}%). "
                   'Treat predictions for unusual or luxury properties with extra caution.')

with tab_upload:
    st.markdown('Upload a CSV with the same columns as `data/sydney_sold_properties.csv` '
                '(`sold_price` is optional). Missing optional values can be left blank.')
    up = st.file_uploader('CSV file', type='csv')
    if up is not None:
        from housing_features import clean
        dfu = pd.read_csv(up, dtype={'property_id': str})
        if 'sold_price' not in dfu:
            dfu['sold_price'] = np.nan
        if 'sold_date' not in dfu:
            dfu['sold_date'] = pd.Timestamp.today().strftime('%d/%m/%Y')
        dfu = clean(dfu)
        dfu['sold_date'] = dfu['sold_date'].fillna(pd.Timestamp.today().normalize())
        dfu['predicted_price'] = predict(dfu).round(-3)
        st.dataframe(dfu[[c for c in ['property_id', 'suburb', 'address', 'property_type', 'bedrooms',
                                      'bathrooms', 'sold_price', 'predicted_price'] if c in dfu]])
        st.download_button('Download predictions', dfu.to_csv(index=False), 'predictions.csv')
