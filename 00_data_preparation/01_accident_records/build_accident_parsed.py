# -*- coding: utf-8 -*-
"""
build_accident_parsed.py — build the Fig. 1 accident input accident_parsed.csv (all vessel types, 2002–2022)

Reconstruction note
--------
The original generating script was not kept (existing file dated 2026-07-19). This script reproduces its structure:
  * keeps all 44 columns of the LLI Excel `data` sheet unchanged;
  * appends 5 columns: lat_dec / lon_dec (DMS -> decimal, same parser as build_full_accident.py,
    verified bit-identical against accident_cargo_2002_2021.parquet), casualty_date (YYYY-MM-DD),
    acc_year, acc_month.
Default CHECK_AGAINST_EXISTING=True: only compare with the existing file, no write; set False to regenerate once it matches.

Downstream filtering (done in 01_fig1_global_patterns/M1_results_v3.ipynb cell 3, not here):
  with coordinates and 2002–2021 -> 50,514; inside analysis domain (ocean cell, annual traffic >= FLOOR) -> 42,857 (manuscript figure).

License note: input is proprietary Lloyd's List Intelligence data; this script's outputs are not released;
the public layer contains only the 1° grid aggregate grid_1deg_cells.csv produced by M1.
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
import re, numpy as np, pandas as pd

XLSX = rf"{DATA_ROOT}/AIS_processing/accident_data/2002-2022maritime accident data.xlsx"
OUT  = rf"{DATA_ROOT}/climate_AIS_data/accident/accident_parsed.csv"
CHECK_AGAINST_EXISTING = True

DMS = re.compile(r"(-?\d+(?:\.\d+)?)\s*°\s*(\d+(?:\.\d+)?)?\s*'?\s*(\d+(?:\.\d+)?)?\s*\"?\s*([NSEWnsew])?")
def dms2dec(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    m = DMS.search(s)
    if not m:
        try: return float(s)
        except Exception: return np.nan
    d = float(m.group(1)); mi = float(m.group(2) or 0); se = float(m.group(3) or 0); h = (m.group(4) or "").upper()
    val = abs(d) + mi/60 + se/3600
    return -val if (h in ("S", "W") or d < 0) else val

df = pd.read_excel(XLSX, sheet_name="data")
df["lat_dec"] = df["Latitude"].map(dms2dec)
df["lon_dec"] = df["Longitude"].map(dms2dec)
_dt = pd.to_datetime(df["Casualty Date"], errors="coerce")
df["casualty_date"] = _dt.dt.strftime("%Y-%m-%d")
df["acc_year"]  = _dt.dt.year
df["acc_month"] = _dt.dt.month

ok = df["lat_dec"].notna() & df["lon_dec"].notna()
print(f"rows {len(df):,} | with coords {int(ok.sum()):,} | 2002–2021 with coords {int((ok & df['acc_year'].between(2002, 2021)).sum()):,}")

if CHECK_AGAINST_EXISTING:
    old = pd.read_csv(OUT, low_memory=False)
    print(f"existing rows {len(old):,}; columns equal: {list(old.columns) == list(df.columns)}")
    n = min(len(old), len(df))
    for c in ["lat_dec", "lon_dec"]:
        a, b = pd.to_numeric(old[c], errors="coerce").values[:n], df[c].values[:n]
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, atol=1e-6, equal_nan=True)
        print(f"  {c}: identical {close.mean()*100:.3f}%  (NaN in both {int(both_nan.sum()):,}, max |diff| {np.nanmax(np.abs(a-b)):.2e})")
    for c in ["acc_year", "acc_month", "casualty_date"]:
        same = (old[c].astype(str).values[:n] == df[c].astype(str).values[:n]).mean()
        print(f"  {c}: identical {same*100:.3f}%")
else:
    df.to_csv(OUT, index=False)
    print("written", OUT)
