# -*- coding: utf-8 -*-
"""
add_vessel_age.py — join vessel age and class society/manager fields onto the weather-matched accident table (Fig. 3 input)

Input   accident_all_weather_2011_2021.csv   (output of match_weather_accident.py, with orig_idx)
        2002-2022maritime accident data.xlsx  (raw LLI Excel, data sheet)
Output  accident_all_weather_2011_2021_age.parquet
        new columns: age, cls, owner, tech_mgr, ism_mgr, lli_type, age_ (valid 0–60 yr), agec (vessel age band)

The join key is orig_idx, not (imo, date): orig_idx = row number of accident_all_2011_2021.parquet,
i.e. Excel data-sheet records with accident year 2011–2021 renumbered in original order (same filter as build_full_accident.py).
(imo is 6% missing; joining on (imo, date) would lose vessel age for those records.)

2026-10-12: CHECK_AGAINST_EXISTING=True vs existing file (built 2026-07-30, 31,024 rows), column by column: all new columns 100% identical.
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
import numpy as np, pandas as pd

CSV_IN  = rf'{DATA_ROOT}/AIS_processing/accident_data/accident_all_weather_2011_2021.csv'
XLS_FP  = rf'{DATA_ROOT}/AIS_processing/accident_data/2002-2022maritime accident data.xlsx'
OUT_FP  = rf'{DATA_ROOT}/AIS_processing/accident_data/accident_all_weather_2011_2021_age.parquet'
CHECK_AGAINST_EXISTING = True          # True: do not overwrite, only compare with existing file

a = pd.read_csv(CSV_IN, low_memory=False)

raw = pd.read_excel(XLS_FP, sheet_name='data')
year = pd.to_datetime(raw['Casualty Date'], errors='coerce').dt.year
raw = raw[year.between(2011, 2021)].reset_index(drop=True)          # row number = orig_idx
V = pd.DataFrame({'age':      pd.to_numeric(raw['Age'], errors='coerce'),
                  'cls':      raw['Class'],
                  'owner':    raw['Beneficial Owner'],
                  'tech_mgr': raw['Technical Manager'],
                  'ism_mgr':  raw['ISM Manager'],
                  'lli_type': raw['LLI Vessel Type']})
for c in ['cls', 'owner', 'tech_mgr', 'ism_mgr', 'lli_type']:
    V[c] = V[c].astype('string')
a = pd.concat([a, V.iloc[a['orig_idx'].values].reset_index(drop=True)], axis=1)

a['age_'] = a['age'].where(a['age'].between(0, 60))
a['agec'] = pd.cut(a['age_'], [-.1, 5, 15, 25, 60], labels=['0-5', '6-15', '16-25', '>25'])

print(f'rows {len(a):,} | age non-missing {a["age"].notna().mean()*100:.1f}%')
print(a['agec'].value_counts(dropna=False).to_string())

if CHECK_AGAINST_EXISTING:
    old = pd.read_parquet(OUT_FP)
    print(f'existing rows {len(old):,}; column sets equal: {set(old.columns) == set(a.columns)}')
    if len(old) == len(a):
        for c in ['age', 'cls', 'owner', 'tech_mgr', 'ism_mgr', 'lli_type', 'age_', 'agec']:
            x, y = a[c].astype(object), old[c].astype(object)
            same = (x.isna() & y.isna()) | (x == y)
            print(f'  {c:9s} identical share {same.mean()*100:.2f}%')
else:
    a.to_parquet(OUT_FP, index=False); print('written', OUT_FP)
