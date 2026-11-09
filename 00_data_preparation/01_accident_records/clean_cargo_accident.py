import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
# -*- coding: utf-8 -*-
# ============================================================
# clean_cargo_accident.py
# Purpose: clean accident records to match the scope of the GMTDS "Cargo Ships" traffic layer
#   keep   = GMTDS Cargo (AIS 70-79) + OBO combination carriers
#   drop   = GMTDS Tanker (AIS 80-89)  -- oil/chemical/gas/service liquid-cargo ships
# Rationale: GMTDS lists Cargo and Tanker as separate classes; the exposure layer is Cargo-only,
#       so the accident numerator must match, otherwise = tanker accidents / cargo ship traffic.
# ============================================================
import os
import pandas as pd

SRC = rf'{DATA_ROOT}/AIS_processing/accident_data/accident_cargo_2002_2021.parquet'
OUT = rf'{DATA_ROOT}/AIS_processing/accident_data/accident_cargo_clean_2002_2021.parquet'

# ── vessel type -> AIS class mapping (each of the 37 vessel_type values in SRC classified) ──
CARGO = {                                   # GMTDS Cargo (AIS 70-79): keep
    'bulk carrier',
    'general cargo with container capacity',
    'general cargo',
    'fully cellular containership',
    'Roll On Roll Off',
    'roll on roll off with container capacity',
    'reefer',
    'vehicle carrier',
    'bulk cement carrier',
    'livestock',
    'bulk carrier with container capacity',
    'bulk ore carrier',
    'wood-chip carrier',
    'fully cellular refrigerated',
    'barge container carrier',
    'barge carrier',
    'bulk aggregates carrier',
}
OBO = {                                     # combination carriers (dry bulk/oil): kept here
    'combined ore and oil carrier',
    'combined bulk and oil carrier',
}
TANKER = {                                  # GMTDS Tanker (AIS 80-89): drop
    'crude oil tanker',
    'product tanker',
    'chemical tanker',
    'Combined chemical and oil tanker',
    'asphalt tanker',
    'edible oil tanker',
    'water tanker',
    'wine tanker',
    'molasses tanker',
    'fish oil tanker',
    'fruit juice tanker',
    'Liquefied Petroleum Gas Carrier',
    'Liquefied Natural Gas Carrier',
    'Combined LNG and LPG Gas Carrier',
    'bunkering tanker',
    'floating production tanker',   # FPSO: fixed production unit, not a trading ship
    'sludge carrier',               # sludge service vessel
    'Tanker (unspecified)',
}

def classify(vt):
    if vt in CARGO:  return 'cargo'
    if vt in OBO:    return 'obo'
    if vt in TANKER: return 'tanker'
    return 'UNCLASSIFIED'

def main():
    df = pd.read_parquet(SRC)
    print(f'Source file: {df.shape[0]:,} rows')

    df['ais_class'] = df['vessel_type'].map(classify)

    # ── Completeness check: fail on any unclassified vessel type, never drop silently ──
    unk = df.loc[df['ais_class'] == 'UNCLASSIFIED', 'vessel_type'].value_counts()
    if len(unk):
        print('\n[ERROR] Unclassified vessel_type found; add it to the mapping and rerun:')
        print(unk.to_string())
        raise SystemExit(1)

    # ── Class counts ──
    print('\n=== Classification ===')
    print(df['ais_class'].value_counts().to_string())

    keep = df[df['ais_class'].isin(['cargo', 'obo'])].copy()
    drop = df[df['ais_class'] == 'tanker']
    print(f'\nKept (cargo+obo): {keep.shape[0]:,}  |  Dropped (tanker): {drop.shape[0]:,}'
          f'  |  Dropped share: {drop.shape[0]/df.shape[0]*100:.1f}%')

    # ── Dropped tanker breakdown (for manual review) ──
    print('\n=== Dropped tanker breakdown ===')
    print(drop['vessel_type'].value_counts().to_string())

    # ── Impact by region/year (if columns exist) ──
    if 'year' in df.columns:
        chk = (df.assign(kept=df['ais_class'].isin(['cargo', 'obo']))
                 .groupby('year')['kept'].agg(['sum', 'count']))
        chk['drop'] = chk['count'] - chk['sum']
        print('\n=== Kept/dropped by year ===')
        print(chk.rename(columns={'sum': 'kept', 'count': 'total'}).to_string())

    keep.to_parquet(OUT, index=False)
    print(f'\n[OK] Wrote: {OUT}')
    print(f'     {keep.shape[0]:,} rows | columns: {list(keep.columns)}')

if __name__ == '__main__':
    main()
