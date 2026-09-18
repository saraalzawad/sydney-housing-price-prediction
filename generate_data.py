"""
Synthetic Sydney Housing Dataset
Parramatta · Chatswood · Bondi – 120 properties (40 per suburb)
Based on realistic 2023-2024 Sydney property market data.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

def make_suburb(suburb, n, prop_types, bed_dist, price_fn, base_year=2010):
    rows = []
    for i in range(n):
        ptype    = np.random.choice(prop_types[0], p=prop_types[1])
        beds     = np.random.choice(bed_dist[0], p=bed_dist[1])
        baths    = max(1, beds - np.random.choice([0,1], p=[0.4,0.6]))
        parking  = np.random.choice([0,1,2], p=[0.15,0.55,0.30]) if ptype=='Apartment' else np.random.choice([1,2,3], p=[0.3,0.5,0.2])
        if ptype == 'House':
            land   = int(np.random.normal(450, 120))
            land   = max(200, min(900, land))
            floor  = int(land * np.random.uniform(0.45, 0.65))
        elif ptype == 'Townhouse':
            land   = int(np.random.normal(200, 50))
            land   = max(100, min(350, land))
            floor  = int(np.random.normal(160, 30))
            floor  = max(100, floor)
        else:  # Apartment
            land   = 0
            floor  = int(np.random.normal(75 + beds*20, 15))
            floor  = max(40, floor)
        year_built = int(np.random.uniform(base_year, 2023))
        age        = 2024 - year_built
        month      = np.random.randint(1,13)
        year       = np.random.randint(2022,2025)
        sale_date  = f"{year}-{month:02d}-{np.random.randint(1,28):02d}"
        price      = price_fn(ptype, beds, floor, land, age, parking)
        price      = int(max(400000, price + np.random.normal(0, price*0.04)))
        rows.append({
            'suburb': suburb,
            'property_type': ptype,
            'bedrooms': beds,
            'bathrooms': baths,
            'parking': parking,
            'land_size_sqm': land,
            'floor_area_sqm': floor,
            'year_built': year_built,
            'property_age_years': age,
            'sale_date': sale_date,
            'sale_year': year,
            'sale_price': price,
        })
    return rows

# ── Parramatta ──────────────────────────────────────────────────────────────
def price_parramatta(ptype, beds, floor, land, age, parking):
    base = {'Apartment': 720_000, 'Townhouse': 950_000, 'House': 1_150_000}[ptype]
    p  = base + beds * 65_000 + floor * 800 - age * 3_000 + parking * 20_000
    if land > 0: p += land * 500
    return p

rows_p = make_suburb(
    'Parramatta', 40,
    prop_types=(['Apartment','Townhouse','House'], [0.55, 0.25, 0.20]),
    bed_dist=([1,2,3,4],        [0.15, 0.45, 0.30, 0.10]),
    price_fn=price_parramatta
)

# ── Chatswood ───────────────────────────────────────────────────────────────
def price_chatswood(ptype, beds, floor, land, age, parking):
    base = {'Apartment': 1_050_000, 'Townhouse': 1_500_000, 'House': 2_300_000}[ptype]
    p  = base + beds * 110_000 + floor * 1_200 - age * 4_000 + parking * 35_000
    if land > 0: p += land * 900
    return p

rows_c = make_suburb(
    'Chatswood', 40,
    prop_types=(['Apartment','Townhouse','House'], [0.50, 0.20, 0.30]),
    bed_dist=([1,2,3,4,5],      [0.08, 0.35, 0.32, 0.18, 0.07]),
    price_fn=price_chatswood, base_year=1970
)

# ── Bondi ────────────────────────────────────────────────────────────────────
def price_bondi(ptype, beds, floor, land, age, parking):
    base = {'Apartment': 1_400_000, 'Townhouse': 2_200_000, 'House': 3_800_000}[ptype]
    p  = base + beds * 160_000 + floor * 1_800 - age * 2_000 + parking * 50_000
    if land > 0: p += land * 1_400
    return p

rows_b = make_suburb(
    'Bondi', 40,
    prop_types=(['Apartment','Townhouse','House'], [0.60, 0.15, 0.25]),
    bed_dist=([1,2,3,4,5],      [0.10, 0.40, 0.30, 0.14, 0.06]),
    price_fn=price_bondi, base_year=1960
)

df = pd.DataFrame(rows_p + rows_c + rows_b)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add distance to CBD (approximate)
cbd_dist = {'Parramatta': 23, 'Chatswood': 10, 'Bondi': 7}
df['distance_to_cbd_km'] = df['suburb'].map(cbd_dist) + np.random.normal(0, 1.5, len(df))
df['distance_to_cbd_km'] = df['distance_to_cbd_km'].clip(2, 35).round(1)

# Add school rating (1-10)
school_base = {'Parramatta': 6.5, 'Chatswood': 8.5, 'Bondi': 7.5}
df['school_rating'] = df['suburb'].map(school_base) + np.random.normal(0, 0.8, len(df))
df['school_rating']  = df['school_rating'].clip(4, 10).round(1)

import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'data'), exist_ok=True)
df.to_csv(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'data', 'sydney_housing.csv'), index=False)
print(f"Dataset: {df.shape}")
print(df.groupby('suburb')[['sale_price']].agg(['count','mean','min','max']).round(0))
print("\nProperty types:")
print(df.groupby(['suburb','property_type']).size().unstack(fill_value=0))
