"""
build_full_accident.py
=======================
Build the [all vessel types] accident dataset (2011-2021) from the raw xlsx; coordinates DMS->decimal, column names aligned with the cargo parquet,
keeping vessel_type for later filtering by vessel type.
Coordinate parsing verified bit-identical against accident_cargo_2002_2021.parquet (diff=0).

Output: accident_all_2011_2021.parquet
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
import re, numpy as np, pandas as pd

XLSX = rf"{DATA_ROOT}/AIS_processing/accident_data/2002-2022maritime accident data.xlsx"
OUT  = rf"{DATA_ROOT}/AIS_processing/accident_data/accident_all_2011_2021.parquet"

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
out = pd.DataFrame({
    "vessel_name":       df["Name"],
    "imo":               df["IMO"],
    "flag":              df["Flag"],
    "DWT":               df["DWT"],
    "GT":                df["GT"],
    "built_at":          df["Built At"],
    "casualty_date":     pd.to_datetime(df["Casualty Date"], errors="coerce"),
    "casualty_location": df["Casualty Location"],
    "cause":             df["Cause of Casualty"],
    "vessel_type":       df["LLI Vessel Type"],
    "vessel_type_size":  df["Vessel Type and Size"],
    "loss_type":         df["Loss Type"],
    "pollution":         df["Pollution Indicator"],
    "serious":           df["Serious Indicator"],
    "n_dead":            df["No. Dead"],
    "n_injured":         df["No. Injured"],
    "n_missing":         df["No. Missing"],
    "lat":               df["Latitude"].map(dms2dec),
    "lon":               df["Longitude"].map(dms2dec),
})
out["year"] = out["casualty_date"].dt.year
out = out[out["year"].between(2011, 2021)].reset_index(drop=True)

# text columns may contain stray datetimes/numbers (dirty xlsx) -> cast to nullable string to avoid parquet type conflicts
TEXT = ["vessel_name","flag","built_at","casualty_location","cause","vessel_type",
        "vessel_type_size","loss_type","pollution","serious"]
for c in TEXT:
    out[c] = out[c].apply(lambda x: str(x) if pd.notna(x) else None).astype("string")

out.to_parquet(OUT, index=False)
print(f"Wrote {OUT}")
print(f"2011-2021 all vessel types: {len(out)} records | with coordinates {out[['lat','lon']].notna().all(axis=1).sum()}")
print("Number of vessel types:", out["vessel_type"].nunique())
print(out["vessel_type"].value_counts(dropna=False).head(12).to_string())
