"""
match_weather_accident.py
=========================
Match cargo accidents with ERA5 weather (wave SWH + 10 m wind), 2011-2021 only.
For each accident, compute daily max / daily mean of hourly data on its casualty_date (UTC):
    swh_daily_max, swh_daily_mean          (ERA5 SWH, 0.5°)
    wind_daily_max, wind_daily_mean        (wind speed=√(u10²+v10²), ERA5 10m, 0.25°)

Performance note: the nc files use large chunks (~974/674 steps) + zlib; reading a full-year series at one point decompresses 9-13 chunks, very slow.
=> read only the 24-hour [time window] of the accident day swh[h0:h1, li, loi], touching 1-2 chunks each. ~222 ms/record, ~55 min in total.

Rules:
  - Time: the 24 hours of casualty_date (approx. UTC), valid hours only; record n_hours.
  - Space: convert longitude to 0-360, take the nearest grid point.
      * Wave is NaN over land -> search outward for the nearest ocean grid point, up to MAX_DEG_WAVE (default 1.0°), else NaN.
      * Wind has values over land too -> use the nearest grid point directly, no fallback.
  - Per-year checkpoint, to survive interruptions.
Note: the wind .grib files are actually NetCDF (wrong extension); open them with netCDF4.
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timezone

# ----------------------- Config -----------------------
ACC_PATH  = rf"{DATA_ROOT}/AIS_processing/accident_data/accident_all_2011_2021.parquet"  # all vessel types; keep vessel_type for later filtering
WAVE_DIR  = rf"{DATA_ROOT}/AIS_processing/weather_data/wave"      # era5_swh_{year}.nc
WIND_DIR  = rf"{DATA_ROOT}/AIS_processing/weather_data/wind"      # era5_wind10m_{year}.grib (actually NetCDF)
CKPT_DIR  = rf"{DATA_ROOT}/AIS_processing/accident_data/weather_ckpt_all"
OUT_CSV   = rf"{DATA_ROOT}/AIS_processing/accident_data/accident_all_weather_2011_2021.csv"
YEARS     = range(2011, 2022)
MAX_DEG_WAVE = 1.0     # max distance (deg) to nearest valid wave grid point; beyond -> NaN
# ----------------------------------------------------

os.makedirs(CKPT_DIR, exist_ok=True)
def flush(m): print(m, flush=True)
def nearest_idx(arr, val): return int(np.argmin(np.abs(arr - val)))
def to360(lon): return lon % 360.0

def build_date2hidx(time_seconds):
    """valid_time(unix seconds) -> {'YYYY-MM-DD': (h0, h1)} hour-index half-open interval [h0,h1) for that day (UTC)"""
    d = {}
    for hi, t in enumerate(time_seconds):
        key = datetime.fromtimestamp(float(t), tz=timezone.utc).strftime("%Y-%m-%d")
        if key not in d:
            d[key] = [hi, hi + 1]
        else:
            d[key][1] = hi + 1
    return d

# ── 1. Load accident data ───────────────────────────────────────────────
df = pd.read_parquet(ACC_PATH)
df["casualty_date"] = pd.to_datetime(df["casualty_date"], errors="coerce")
mask = (df["casualty_date"].notna() & df["lat"].notna() & df["lon"].notna()
        & df["year"].between(2011, 2021))
work = df[mask].copy()
work.index.name = "orig_idx"
work = work.reset_index()
flush(f"Total records {len(df)} | 2011-2021 valid (date+coordinates) {len(work)}")

OUTCOLS = ["swh_daily_max","swh_daily_mean","swh_cell_dist","swh_n_hours",
           "wind_daily_max","wind_daily_mean","wind_n_hours"]
for c in OUTCOLS:
    work[c] = np.nan
MISSING_WIND = []   # years with corrupted wind files

# ── 2. Match year by year ───────────────────────────────────────────────────
for year in YEARS:
    yr_rows = work.index[work["year"] == year].tolist()
    if not yr_rows:
        continue
    ckpt = os.path.join(CKPT_DIR, f"ckpt_{year}.csv")
    if os.path.exists(ckpt):
        flush(f"[{year}] Loaded checkpoint, skipping")
        ck = pd.read_csv(ckpt).set_index("row")
        for c in ck.columns:
            work.loc[ck.index, c] = ck[c].values
        continue

    wave_path = os.path.join(WAVE_DIR, f"era5_swh_{year}.nc")
    wind_path = os.path.join(WIND_DIR, f"era5_wind10m_{year}.grib")
    flush(f"[{year}] {len(yr_rows)} records | opening nc ...")

    # --- Open wave (on failure skip the year, no checkpoint, so it can be rerun after fixing) ---
    try:
        wv = nc.Dataset(wave_path, "r")
        wlat = np.array(wv.variables["latitude"][:]); wlon = np.array(wv.variables["longitude"][:])
        swh = wv.variables["swh"]
        wfill = getattr(swh, "_FillValue", getattr(swh, "missing_value", 9.96921e36))
        wdate2h = build_date2hidx(np.array(wv.variables["valid_time"][:]))
        wstep = abs(wlat[1] - wlat[0]); wmax_ring = int(round(MAX_DEG_WAVE / wstep))
    except Exception as e:
        flush(f"[{year}] ⚠ cannot open wave file ({str(e)[:60]}) -> skipping year, rerun after fixing"); continue

    # --- Open wind (on failure wind = NaN, wave still computed) ---
    wd = None
    try:
        wd = nc.Dataset(wind_path, "r")
        dlat = np.array(wd.variables["latitude"][:]); dlon = np.array(wd.variables["longitude"][:])
        u10 = wd.variables["u10"]; v10 = wd.variables["v10"]
        ddate2h = build_date2hidx(np.array(wd.variables["valid_time"][:]))
    except Exception as e:
        wd = None
        MISSING_WIND.append(year)
        flush(f"[{year}] ⚠ wind file corrupted/missing ({str(e)[:60]}) -> wind = NaN this year, wave as usual."
              f" After re-downloading, delete ckpt_{year}.csv and rerun")

    def wave_window(li, loi, h0, h1):
        s = np.array(swh[h0:h1, li, loi], dtype=np.float32)
        s[np.abs(s) >= abs(wfill) * 0.9] = np.nan
        if not np.all(np.isnan(s)):
            return s, 0.0
        # Land: search outward ring by ring for nearest ocean grid point (dist<=MAX_DEG_WAVE)
        for r in range(1, wmax_ring + 1):
            best = None
            for dli in range(-r, r + 1):
                for dloi in range(-r, r + 1):
                    if max(abs(dli), abs(dloi)) != r:
                        continue
                    ni, noi = li + dli, loi + dloi
                    if 0 <= ni < len(wlat) and 0 <= noi < len(wlon):
                        dist = ((wlat[ni]-wlat[li])**2 + (wlon[noi]-wlon[loi])**2) ** 0.5
                        if dist > MAX_DEG_WAVE:
                            continue
                        s2 = np.array(swh[h0:h1, ni, noi], dtype=np.float32)
                        s2[np.abs(s2) >= abs(wfill) * 0.9] = np.nan
                        if not np.all(np.isnan(s2)) and (best is None or dist < best[1]):
                            best = (s2, dist)
            if best is not None:
                return best
        return None, np.nan

    ck_rows = []
    for k, ridx in enumerate(yr_rows):
        row = work.loc[ridx]
        lat, lon = float(row["lat"]), float(row["lon"]); lon360 = to360(lon)
        date_str = row["casualty_date"].strftime("%Y-%m-%d")

        # WAVE
        smax = smean = np.nan; dist = np.nan; snh = 0
        wh = wdate2h.get(date_str)
        if wh:
            li, loi = nearest_idx(wlat, lat), nearest_idx(wlon, lon360)
            s, dist = wave_window(li, loi, wh[0], wh[1])
            if s is not None:
                v = s[~np.isnan(s)]
                if len(v): smax, smean, snh = float(v.max()), float(v.mean()), int(len(v))

        # WIND
        wmax = wmean = np.nan; wnh = 0
        dh = ddate2h.get(date_str) if wd is not None else None
        if dh:
            li2, loi2 = nearest_idx(dlat, lat), nearest_idx(dlon, lon360)
            u = np.array(u10[dh[0]:dh[1], li2, loi2], dtype=np.float32)
            vv = np.array(v10[dh[0]:dh[1], li2, loi2], dtype=np.float32)
            sp = np.sqrt(u*u + vv*vv); sp = sp[~np.isnan(sp)]
            if len(sp): wmax, wmean, wnh = float(sp.max()), float(sp.mean()), int(len(sp))

        work.at[ridx, "swh_daily_max"]  = smax
        work.at[ridx, "swh_daily_mean"] = smean
        work.at[ridx, "swh_cell_dist"]  = dist
        work.at[ridx, "swh_n_hours"]    = snh
        work.at[ridx, "wind_daily_max"]  = wmax
        work.at[ridx, "wind_daily_mean"] = wmean
        work.at[ridx, "wind_n_hours"]    = wnh
        ck_rows.append({"row": ridx, "swh_daily_max": smax, "swh_daily_mean": smean,
                        "swh_cell_dist": dist, "swh_n_hours": snh,
                        "wind_daily_max": wmax, "wind_daily_mean": wmean, "wind_n_hours": wnh})
        if (k + 1) % 500 == 0:
            flush(f"[{year}]   {k+1}/{len(yr_rows)}")

    wv.close(); wd.close()
    pd.DataFrame(ck_rows).to_csv(ckpt, index=False)
    matched = int(np.isfinite(work.loc[yr_rows, "wind_daily_max"]).sum())
    flush(f"[{year}] done | wind matched {matched}/{len(yr_rows)} | wave NaN(>{MAX_DEG_WAVE}°) "
          f"{int(work.loc[yr_rows,'swh_daily_max'].isna().sum())}")

# ── 3. Output ───────────────────────────────────────────────────────
work.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
flush(f"\n✅ Wrote {OUT_CSV}  ({len(work)} rows)")
flush("swh matched: %d | wind matched: %d" %
      (int(np.isfinite(work['swh_daily_max']).sum()), int(np.isfinite(work['wind_daily_max']).sum())))
if MISSING_WIND:
    flush(f"\n⚠ wind files corrupted for these years, wind all NaN: {MISSING_WIND}")
    flush(f"   Re-download era5_wind10m_<year>.grib for these years, then delete {CKPT_DIR}/ckpt_<year>.csv and rerun this script.")
