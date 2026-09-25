"""
Shared data-cleaning, validation and feature-engineering code.
Used by the notebook and the Streamlit app so that both
treat the data in exactly the same way.
"""
import re
import numpy as np
import pandas as pd

SUBURBS = ['Parramatta', 'Chatswood', 'Bondi']
PROPERTY_TYPES = ['House', 'Townhouse', 'Apartment']
SALE_METHODS = ['Auction', 'Private treaty', 'Unknown']
REF_DATE = pd.Timestamp('2023-01-01')      # time feature = months since this date

REQUIRED = ['property_id', 'suburb', 'address', 'property_type', 'bedrooms',
            'bathrooms', 'sold_price', 'sold_date', 'source_site', 'listing_url']

# Map common listing-site labels onto the three property types used in the project
TYPE_MAP = {
    'house': 'House', 'semi': 'House', 'semi-detached': 'House', 'duplex': 'House',
    'terrace': 'House', 'free standing': 'House',
    'townhouse': 'Townhouse', 'villa': 'Townhouse',
    'apartment': 'Apartment', 'unit': 'Apartment', 'flat': 'Apartment',
    'studio': 'Apartment', 'penthouse': 'Apartment',
}

# Keyword flags extracted from the agent's description (text data)
KEYWORDS = {
    'kw_renovated': r'renovat|refurbish|brand new kitchen|updated throughout|newly',
    'kw_original':  r'original condition|renovator|deceased estate|potential|opportunity to update',
    'kw_views':     r'\bview|outlook|vista|panoram',
    'kw_water':     r'ocean|beach|harbour|water',
    'kw_pool':      r'\bpool\b',
    'kw_garden':    r'garden|backyard|courtyard|lawn',
    'kw_station':   r'station|metro|train|transport',
    'kw_school':    r'school|catchment',
    'kw_develop':   r'develop|\bda\b|approved plans|subdivi|dual occ',
}


def load_raw(path):
    return pd.read_csv(path, dtype={'property_id': str})


def clean(df):
    """Standardise text labels, coerce numbers and parse dates (no rows dropped)."""
    df = df.copy()
    for c in ['suburb', 'address', 'property_type', 'sale_method', 'agency',
              'description', 'source_site', 'listing_url', 'notes']:
        if c in df:
            df[c] = df[c].astype('string').str.strip()
    df['suburb'] = df['suburb'].str.title()
    df['property_type'] = (df['property_type'].str.lower().map(TYPE_MAP)
                           .fillna(df['property_type'].str.title()))
    sm = df.get('sale_method', pd.Series(index=df.index, dtype='string')).str.lower()
    df['sale_method'] = np.select([sm.str.contains('auction', na=False),
                                   sm.str.contains('private|treaty|sold', na=False)],
                                  ['Auction', 'Private treaty'], 'Unknown')
    df['sold_price'] = pd.to_numeric(df['sold_price'].astype(str)
                                     .str.replace(r'[\$,\s]', '', regex=True), errors='coerce')
    for c in ['bedrooms', 'bathrooms', 'parking', 'land_size_sqm',
              'building_size_sqm', 'distance_to_station_km']:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['sold_date'] = pd.to_datetime(df['sold_date'], dayfirst=True, errors='coerce')
    if 'description' not in df:
        df['description'] = pd.NA
    return df


def validate(df, min_total=100, min_per_suburb=30, verbose=True):
    """Check the collected data against the task requirements. Returns list of problems."""
    problems = []
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        problems.append(f'Missing required columns: {missing_cols}')
        return problems
    if len(df) < min_total:
        problems.append(f'Only {len(df)} properties - the task needs at least {min_total}.')
    counts = df['suburb'].value_counts()
    for s in SUBURBS:
        if counts.get(s, 0) < min_per_suburb:
            problems.append(f'{s}: {counts.get(s, 0)} properties - needs at least {min_per_suburb}.')
    other = set(df['suburb'].dropna()) - set(SUBURBS)
    if other:
        problems.append(f'Unexpected suburb names (check spelling): {sorted(other)}')
    bad_type = set(df['property_type'].dropna()) - set(PROPERTY_TYPES)
    if bad_type:
        problems.append(f'Unrecognised property types: {sorted(bad_type)} - use House/Townhouse/Apartment.')
    for c in REQUIRED:
        n = df[c].isna().sum()
        if n:
            problems.append(f'{n} rows have no value for required column "{c}".')
    if df['property_id'].duplicated().any():
        problems.append(f"Duplicate property_id values: {df.loc[df['property_id'].duplicated(), 'property_id'].tolist()}")
    dup_addr = df.duplicated(subset=['suburb', 'address', 'sold_date'], keep=False) & df['address'].notna()
    if dup_addr.any():
        problems.append(f'{dup_addr.sum()} rows share the same address and sold date (possible duplicates).')
    p = df['sold_price']
    odd = df[(p < 200_000) | (p > 30_000_000)]
    if len(odd):
        problems.append(f'Check sold_price for property_id {odd["property_id"].tolist()} (outside $200k-$30M).')
    if df['sold_date'].isna().any():
        problems.append('Some sold_date values could not be read - use DD/MM/YYYY.')
    if df['bedrooms'].gt(10).any() or df['bathrooms'].gt(10).any():
        problems.append('Some bedroom/bathroom counts are above 10 - check for typos.')
    houses_no_land = df[(df['property_type'] == 'House') & df['land_size_sqm'].isna()]
    if verbose:
        print(f'Rows: {len(df)}')
        print(counts.reindex(SUBURBS).fillna(0).astype(int).to_string())
        print(f'\nHouses without a land size recorded: {len(houses_no_land)}')
        print('\nMissing values per column:')
        na = df.isna().sum()
        print(na[na > 0].to_string() if (na > 0).any() else '  none')
        print('\n' + ('All checks passed.' if not problems else 'Problems found:'))
        for pr in problems:
            print('  -', pr)
    return problems


def engineer(df):
    """Create model features. Works on the full dataset or on a single app input row."""
    df = df.copy()
    # Land: strata properties (apartments, many townhouses) have no land title of their own
    df['is_strata'] = (df['property_type'].eq('Apartment') |
                       (df['property_type'].eq('Townhouse') & df['land_size_sqm'].isna())).astype(int)
    land = df['land_size_sqm'].where(df['is_strata'].eq(0))
    df['land_missing'] = (df['is_strata'].eq(0) & land.isna()).astype(int)
    df['log_land'] = np.log1p(land.fillna(0))
    # Building size is often not listed: keep a flag so the model knows when it was imputed
    df['building_missing'] = df['building_size_sqm'].isna().astype(int)
    df['total_rooms'] = df['bedrooms'].fillna(0) + df['bathrooms'].fillna(0)
    df['bath_per_bed'] = df['bathrooms'] / df['bedrooms'].clip(lower=1)
    # Time trend
    df['months_since_2023'] = ((df['sold_date'] - REF_DATE).dt.days / 30.44).round(1)
    df['is_auction'] = df['sale_method'].eq('Auction').astype(int)
    # Text features from the agent description
    text = df['description'].fillna('').astype(str).str.lower()
    for name, pattern in KEYWORDS.items():
        df[name] = text.str.contains(pattern, regex=True).astype(int)
    df['desc_words'] = text.str.split().str.len().fillna(0)
    return df


CAT_FEATURES = ['suburb', 'property_type']
NUM_FEATURES = ['bedrooms', 'bathrooms', 'parking', 'log_land', 'building_size_sqm',
                'is_strata', 'land_missing', 'building_missing', 'total_rooms', 'bath_per_bed',
                'months_since_2023', 'is_auction', 'desc_words'] + list(KEYWORDS)
OPTIONAL_FEATURES = ['distance_to_station_km']   # used only if you collected it


def feature_lists(df):
    num = list(NUM_FEATURES)
    for c in OPTIONAL_FEATURES:
        if c in df and df[c].notna().mean() >= 0.8:
            num.append(c)
    return CAT_FEATURES, num
