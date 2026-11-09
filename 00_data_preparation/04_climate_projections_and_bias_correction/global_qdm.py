#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
global_qdm.py — QDM bias correction on global 2° cells
              ACCESS-CM2 / EC-Earth3 × wave (Hs) / wind (U10) × his / ssp126 / ssp585

Method identical to the regional version (maritime accident\wind_wave_QDM_test.ipynb):
  * Model: WW3 3-hourly fields → daily maximum → max over 0.5° points within each 2° cell
  * Obs: ERA5 hourly → daily maximum → max within each 2° cell (wave 0.5° 4×4, wind 0.25° 8×8)
  * QDM: multiplicative, per cell and per calendar month (±1-month window), NQ=100 quantiles;
         calibration period REF=2002–2014; projections corrected in 10-year blocks; his in two blocks 1985–2001 / 2002–2014
  * GOOD: his coverage within REF ≥90% for both models × both variables, and ERA5 coverage ≥90% for both variables

Output directory (new): <DATA_ROOT>/new_climate/_proj_global/
  cells.csv                         all 16,200 2° cells: grid_id, clat, clon, sea_obs, sea_mod, sea
  cells_sea.csv                     ocean cells (same order as the cell dim of all .nc): + good, w_area
  cm_{kind}_{model}_{tag}.nc        uncorrected model daily maximum (time, cell)
  obs_years\obs_{kind}_{year}.nc    ERA5 per-year cache
  obs_{kind}.nc                     ERA5 daily maximum, full REF period (time, cell)
  qdm_{kind}_{model}_{tag}.nc       bias-corrected daily maximum (time, cell), non-GOOD cell = NaN
  annual_days_{kind}_{model}_{tag}.nc   annual exceedance days per cell (year, cell, cut)
  tables\annual_global_series.csv   global annual exceedance days, area-weighted (cos lat) over GOOD cells
  tables\check_ref_exceedance.csv   check: REF-period exceedance days, ERA5 / corrected / uncorrected
  tables\compare_regional.csv       consistency check against regional _proj_main on overlapping cells
  log.txt

Run (from the WRH directory):
  python global_qdm.py --stage all --workers 4
  python global_qdm.py --stage cells,cm            # stages can be run separately; each is cached for resume
  python global_qdm.py --stage obs --workers 6     # ERA5 wind is slowest; raise parallelism if G: is an SSD

Stage order: cells → cm → obs → good → qdm → check → annual → compare
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README

import os, sys, glob, time, argparse, warnings
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# On Windows, stdout redirected to a file defaults to cp1252, non-ASCII raises UnicodeEncodeError → force UTF-8
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ====================================================================
# Paths
# ====================================================================
WV_B  = rf'{DATA_ROOT}/new_climate/wave'
WN_B  = rf'{DATA_ROOT}/new_climate/criso_wind'
E5_WV = rf'{DATA_ROOT}/AIS_processing/weather_data/wave'
E5_WN = rf'{DATA_ROOT}/AIS_processing/weather_data/wind'
OLD   = rf'{DATA_ROOT}/new_climate/_proj_main'          # regional version, used only in compare stage
OUT   = rf'{DATA_ROOT}/new_climate/_proj_global'        # ★ new output directory
OBS_Y = os.path.join(OUT, 'obs_years')
TB    = os.path.join(OUT, 'tables')

MODELS = {
 'ACCESS': {'wave': {'his': rf'{WV_B}\ACCESS\CDFAC_his', 'ssp126': rf'{WV_B}\ACCESS\CDFAC1_126',
                     'ssp585': rf'{WV_B}\ACCESS\CDFAC1_585'},
            'wind': {'his': rf'{WN_B}\ACCESS-CM2\his', 'ssp126': rf'{WN_B}\ACCESS-CM2\126',
                     'ssp585': rf'{WN_B}\ACCESS-CM2\585'}},
 'ECEARTH': {'wave': {'his': rf'{WV_B}\ec_earth\cdfac1\his', 'ssp126': rf'{WV_B}\ec_earth\cdfac1\126',
                      'ssp585': rf'{WV_B}\ec_earth\cdfac1\585'},
             'wind': {'his': rf'{WN_B}\EC-EARTH3\his', 'ssp126': rf'{WN_B}\EC-EARTH3\126',
                      'ssp585': rf'{WN_B}\EC-EARTH3\585'}},
}
KINDS = ('wave', 'wind')
TAGS  = ('his', 'ssp126', 'ssp585')

# ====================================================================
# Parameters (same as regional version)
# ====================================================================
REF         = (2002, 2014)
HIS_BLOCKS  = [(1985, 2001), (2002, 2014)]
PROJ_BLOCKS = [(2015, 2024), (2025, 2034), (2035, 2044), (2045, 2054), (2055, 2064),
               (2065, 2074), (2075, 2084), (2085, 2094), (2095, 2100)]
NQ          = 100
COVER_MIN   = 0.9
CUTS        = {'wave': [1.25, 2.5, 4.0, 6.0, 9.0], 'wind': [8.0, 10.8, 13.9, 17.2, 20.8]}
NC_COMP     = 2                       # zlib compression level, 2 = favour speed

# Global 2° cell grid
CLAT = np.arange(-89, 90, 2.0)        # 90
CLON = np.arange(-179, 180, 2.0)      # 180
NLAT, NLON = len(CLAT), len(CLON)
GID  = np.array([f'{a:.3f}_{b:.3f}' for a in CLAT for b in CLON])   # same naming as regional version
GLAT = np.repeat(CLAT, NLON)
GLON = np.tile(CLON, NLAT)


# ====================================================================
# Utilities
# ====================================================================
def log(msg):
    s = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    try:
        with open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8') as f:
            f.write(s + '\n')
    except Exception:
        pass


def win(m, half=1):
    return [((m - 1 + k) % 12) + 1 for k in range(-half, half + 1)]


def cell_max(dm, lat, lon, f):
    """
    dm (nd, nlat, nlon), any latitude order, longitude 0–360 → (nd, 16200) 2° cell maxima.
    f = grid points per 2° (0.5° → 4, 0.25° → 8). The lat=+90 row is dropped.
    """
    lat = np.asarray(lat); lon = np.asarray(lon)
    if lat[0] > lat[-1]:
        dm = dm[:, ::-1, :]; lat = lat[::-1]
    assert abs(lat[0] + 90) < 1e-6, f'lat should start at -90, got {lat[0]}'
    dm = dm[:, :NLAT * f, :]
    order = np.argsort(((lon + 180) % 360) - 180, kind='stable')
    lon_s = (((lon + 180) % 360) - 180)[order]
    assert abs(lon_s[0] + 180) < 1e-6 and dm.shape[2] == NLON * f, 'lon grid mismatch'
    dm = dm[:, :, order]
    c = np.nanmax(dm.reshape(dm.shape[0], NLAT, f, NLON, f), axis=(2, 4))
    return c.reshape(dm.shape[0], -1)


def load_sel():
    df = pd.read_csv(os.path.join(OUT, 'cells_sea.csv'))
    return df['flat_idx'].values.astype(int), df['grid_id'].values


def save_da(arr, times, cells, name, path, tdim='time'):
    da = xr.DataArray(arr.astype(np.float32), dims=(tdim, 'cell'),
                      coords={tdim: times, 'cell': cells}, name=name)
    tmp = path + '.tmp'
    da.to_netcdf(tmp, encoding={name: {'zlib': True, 'complevel': NC_COMP,
                                       'chunksizes': (min(366, len(times)), len(cells))}})
    os.replace(tmp, path)


def open_da(path):
    return xr.open_dataarray(path).load()


# ====================================================================
# Stage 1: cells -- list of ocean cells
# ====================================================================
def stage_cells():
    fo = os.path.join(OUT, 'cells_sea.csv')
    if os.path.exists(fo):
        log('[CACHE] cells_sea.csv'); return
    log('cells: building ocean cell list')

    # ERA5 SWH: 24-h max on Jan 1 and Jul 1; valid on either day → ocean
    ds = xr.open_dataset(os.path.join(E5_WV, f'era5_swh_{REF[0]}.nc'))
    lat, lon = ds.latitude.values, ds.longitude.values
    t = pd.to_datetime(ds.valid_time.values)
    sea_obs = np.zeros(NLAT * NLON, bool)
    for day in (f'{REF[0]}-01-01', f'{REF[0]}-07-01'):
        k = np.where(t.normalize() == pd.Timestamp(day))[0]
        x = ds['swh'].isel(valid_time=k).values.squeeze()
        x = x.reshape(-1, len(lat), len(lon))
        sea_obs |= np.isfinite(cell_max(np.nanmax(x, 0)[None], lat, lon, 4)[0])
    ds.close()

    # WW3: his first-month + July files; finite hs → ocean (MAPSTA==1)
    sea_mod = np.zeros(NLAT * NLON, bool)
    for model in MODELS:
        files = sorted(glob.glob(os.path.join(MODELS[model]['wave']['his'], 'ww3.*_hs.nc')))
        for f in (files[0], files[6]):
            d = xr.open_dataset(f, drop_variables=['MAPSTA'])
            x = d['hs'].values.astype(np.float32)
            sea_mod |= np.isfinite(cell_max(np.nanmax(x, 0)[None], d.latitude.values,
                                            d.longitude.values, 4)[0])
            d.close()

    sea = sea_obs & sea_mod
    df = pd.DataFrame({'grid_id': GID, 'clat': GLAT, 'clon': GLON,
                       'flat_idx': np.arange(NLAT * NLON),
                       'sea_obs': sea_obs, 'sea_mod': sea_mod, 'sea': sea})
    df.to_csv(os.path.join(OUT, 'cells.csv'), index=False)
    ds_ = df[df.sea].copy()
    ds_['w_area'] = np.cos(np.deg2rad(ds_.clat))
    ds_.to_csv(fo, index=False)
    log(f'cells: ocean(ERA5) {sea_obs.sum()} | ocean(WW3) {sea_mod.sum()} | intersection {sea.sum()} / {NLAT*NLON}')


# ====================================================================
# Stage 2: cm -- model daily maximum → cell
# ====================================================================
def build_cm(model, kind, tag):
    fo = os.path.join(OUT, f'cm_{kind}_{model}_{tag}.nc')
    if os.path.exists(fo):
        return f'[CACHE] {os.path.basename(fo)}'
    SEL, cells = load_sel()
    suf = {'wave': 'hs', 'wind': 'wnd'}[kind]
    files = sorted(glob.glob(os.path.join(MODELS[model][kind][tag], f'ww3.*_{suf}.nc')))
    if not files:
        return f'[SKIP] {model} {kind} {tag}: no files'
    t0 = time.time(); A, T = [], []
    for i, f in enumerate(files):
        ds = xr.open_dataset(f, drop_variables=['MAPSTA'])
        t = pd.to_datetime(ds['time'].values)
        lat, lon = ds.latitude.values, ds.longitude.values
        if kind == 'wave':
            v = ds['hs'].values.astype(np.float32)
        else:
            u = ds['uwnd'].values.astype(np.float32); w = ds['vwnd'].values.astype(np.float32)
            v = np.sqrt(u * u + w * w); del u, w
        ds.close()
        n8 = (v.shape[0] // 8) * 8
        if n8 != v.shape[0] or t[0].hour != 0:
            log(f'  WARN {os.path.basename(f)}: steps={v.shape[0]} first hour={t[0].hour}, truncated to {n8}')
        v = v[:n8]; t = t[:n8]
        dm = np.nanmax(v.reshape(n8 // 8, 8, v.shape[1], v.shape[2]), axis=1); del v
        A.append(cell_max(dm, lat, lon, 4)[:, SEL]); T.append(t[::8].normalize().values); del dm
        if (i + 1) % 120 == 0:
            log(f'  cm {model} {kind} {tag}: {i+1}/{len(files)}  ({(time.time()-t0)/60:.0f} min)')
    arr = np.concatenate(A); tim = pd.DatetimeIndex(np.concatenate(T))
    o = np.argsort(tim.values); arr, tim = arr[o], tim[o]
    dup = tim.duplicated().sum()
    if dup:
        keep = ~tim.duplicated(); arr, tim = arr[keep], tim[keep]
    gaps = int((np.diff(tim.values).astype('timedelta64[D]').astype(int) > 1).sum())
    save_da(arr, tim.values, cells, kind, fo)
    return (f'-> {os.path.basename(fo)} {arr.shape} {tim[0].date()}..{tim[-1].date()} '
            f'duplicates {dup} gaps {gaps}  {(time.time()-t0)/60:.0f} min')


def stage_cm(workers, models, kinds):
    tasks = [(m, k, tg) for m in models for k in kinds for tg in TAGS]
    log(f'cm: {len(tasks)} tasks, {workers} workers')
    run_pool(build_cm, tasks, workers)


# ====================================================================
# Stage 3: obs -- ERA5 daily maximum → cell (REF period, per year)
# ====================================================================
def build_obs_year(kind, year):
    fo = os.path.join(OBS_Y, f'obs_{kind}_{year}.nc')
    if os.path.exists(fo):
        return f'[CACHE] {os.path.basename(fo)}'
    SEL, cells = load_sel(); t0 = time.time()
    if kind == 'wave':
        ds = xr.open_dataset(os.path.join(E5_WV, f'era5_swh_{year}.nc'), chunks={'valid_time': 24 * 15})
        blk, f = 15, 4
    else:
        ds = xr.open_dataset(os.path.join(E5_WN, f'era5_wind10m_{year}.grib'),
                             engine='netcdf4', chunks={'valid_time': 24 * 5})
        blk, f = 5, 8
    t = pd.to_datetime(ds['valid_time'].values)
    lat, lon = ds.latitude.values, ds.longitude.values
    assert len(t) % 24 == 0, f'{kind} {year}: hour count {len(t)} is not a multiple of 24'
    ndt = len(t) // 24; A, T = [], []
    for s in range(0, ndt, blk):
        n = min(blk, ndt - s); k = slice(s * 24, (s + n) * 24)
        if kind == 'wave':
            x = ds['swh'].isel(valid_time=k).values.astype(np.float32)
            x = x.reshape(n * 24, len(lat), len(lon))
        else:
            u = ds['u10'].isel(valid_time=k).values.astype(np.float32)
            v = ds['v10'].isel(valid_time=k).values.astype(np.float32)
            x = np.sqrt(u * u + v * v); del u, v
        dm = np.nanmax(x.reshape(n, 24, len(lat), len(lon)), axis=1); del x
        A.append(cell_max(dm, lat, lon, f)[:, SEL]); T.append(t[k][::24].normalize().values); del dm
    ds.close()
    arr = np.concatenate(A); tim = np.concatenate(T)
    save_da(arr, tim, cells, kind, fo)
    return f'-> {os.path.basename(fo)} {arr.shape}  {(time.time()-t0)/60:.0f} min'


def stage_obs(workers, kinds):
    os.makedirs(OBS_Y, exist_ok=True)
    years = list(range(REF[0], REF[1] + 1))
    tasks = [(k, y) for k in kinds for y in years]
    log(f'obs: {len(tasks)} (kind, year) tasks, {workers} workers')
    run_pool(build_obs_year, tasks, workers)
    for kind in kinds:
        fo = os.path.join(OUT, f'obs_{kind}.nc')
        if os.path.exists(fo):
            log(f'[CACHE] obs_{kind}.nc'); continue
        das = [xr.open_dataarray(os.path.join(OBS_Y, f'obs_{kind}_{y}.nc')).load() for y in years]
        da = xr.concat(das, dim='time').sortby('time')
        exp = pd.date_range(f'{REF[0]}-01-01', f'{REF[1]}-12-31', freq='D')
        tim = pd.DatetimeIndex(da.time.values)
        assert len(tim) == len(exp) and (tim == exp).all(), f'{kind} obs time axis incomplete {len(tim)}/{len(exp)}'
        save_da(da.values, da.time.values, da.cell.values, kind, fo)
        log(f'obs: -> obs_{kind}.nc {da.shape}')


# ====================================================================
# Stage 4: good -- coverage filter
# ====================================================================
def stage_good(models, kinds):
    fo = os.path.join(OUT, 'cells_sea.csv')
    df = pd.read_csv(fo)
    good = np.ones(len(df), bool); cov = {}
    for kind in kinds:
        o = open_da(os.path.join(OUT, f'obs_{kind}.nc'))
        c = np.isfinite(o.values).mean(0); cov[f'cov_obs_{kind}'] = c; good &= c >= COVER_MIN
        for model in models:
            h = open_da(os.path.join(OUT, f'cm_{kind}_{model}_his.nc'))
            t = pd.to_datetime(h.time.values); k = (t.year >= REF[0]) & (t.year <= REF[1])
            c = np.isfinite(h.values[k]).mean(0); cov[f'cov_{model}_{kind}'] = c; good &= c >= COVER_MIN
    for k_, v_ in cov.items():
        df[k_] = np.round(v_, 3)
    df['good'] = good
    df.to_csv(fo, index=False)
    log(f'good: {good.sum()} / {len(df)} ocean cells pass coverage ≥{COVER_MIN:.0%}')
    band = pd.cut(df.clat, [-90, -60, -30, 0, 30, 60, 90])
    log('good: passing cells by latitude band\n' + df.groupby(band, observed=True).good.agg(['sum', 'size']).to_string())


# ====================================================================
# Stage 5: qdm
# ====================================================================
QG = np.linspace(0, 1, NQ)


def qgrid(a):
    q = np.nanquantile(a, QG, axis=0).astype(np.float32)
    return np.maximum.accumulate(q, axis=0)


def inv_cdf(q, tau):
    bad = ~np.isfinite(tau); t = np.where(bad, 0., tau)
    p = np.clip(t, 0, 1) * (NQ - 1); lo = np.floor(p).astype(np.int32); np.clip(lo, 0, NQ - 2, out=lo)
    fr = (p - lo).astype(np.float32)
    a = np.take_along_axis(q, lo, 0); b = np.take_along_axis(q, lo + 1, 0)
    r = a + (b - a) * fr; r[bad] = np.nan; return r


def tau_of(x, q, BIG=1e4):
    N = x.shape[1]; off = np.arange(N, dtype=np.float64) * BIG
    ok = np.isfinite(q).all(0)
    qq = np.where(ok, q, np.linspace(0, 1, NQ)[:, None]).astype(np.float64)
    flat = (qq + off).T.ravel()
    idx = np.searchsorted(flat, (np.nan_to_num(x, nan=-1e9) + off).ravel(), side='left')
    idx = idx.reshape(x.shape) - (np.arange(N) * NQ)
    hi = np.clip(idx, 1, NQ - 1).astype(np.int32); lo = hi - 1
    a = np.take_along_axis(qq, lo, 0); b = np.take_along_axis(qq, hi, 0)
    den = b - a; fr = np.where(den > 1e-9, (x - a) / np.where(den > 1e-9, den, 1.), 0.)
    tau = np.clip((lo + np.clip(fr, 0, 1)) / (NQ - 1), 0, 1).astype(np.float32)
    tau[~np.isfinite(x)] = np.nan; tau[:, ~ok] = np.nan
    return tau


def build_qdm(model, kind, tag):
    fo = os.path.join(OUT, f'qdm_{kind}_{model}_{tag}.nc')
    if os.path.exists(fo):
        return f'[CACHE] {os.path.basename(fo)}'
    t0 = time.time()
    good = pd.read_csv(os.path.join(OUT, 'cells_sea.csv'))['good'].values.astype(bool)

    o = open_da(os.path.join(OUT, f'obs_{kind}.nc'))
    ov = o.values; om = pd.to_datetime(o.time.values).month.values
    h = open_da(os.path.join(OUT, f'cm_{kind}_{model}_his.nc'))
    ht = pd.to_datetime(h.time.values); hk = (ht.year >= REF[0]) & (ht.year <= REF[1])
    hv = h.values[hk]; hm = ht.month.values[hk]
    OQ = {m: qgrid(ov[np.isin(om, win(m))]) for m in range(1, 13)}
    HQ = {m: qgrid(hv[np.isin(hm, win(m))]) for m in range(1, 13)}
    del ov, hv

    p = open_da(os.path.join(OUT, f'cm_{kind}_{model}_{tag}.nc'))
    pt = pd.to_datetime(p.time.values); pv = p.values; cells = p.cell.values
    out = np.full_like(pv, np.nan)
    for (y0, y1) in (HIS_BLOCKS if tag == 'his' else PROJ_BLOCKS):
        bk = (pt.year >= y0) & (pt.year <= y1)
        if not bk.any():
            continue
        pb = pv[bk]; pm = pt.month.values[bk]; ob = np.full_like(pb, np.nan)
        for m in range(1, 13):
            fk = np.isin(pm, win(m)); ak = (pm == m)
            if fk.sum() < 50 or not ak.any():
                continue
            pq = qgrid(pb[fk]); x = pb[ak]; tau = tau_of(x, pq)
            hi = inv_cdf(HQ[m], tau); oi = inv_cdf(OQ[m], tau)
            with np.errstate(divide='ignore', invalid='ignore'):
                delta = x / np.where(hi > 1e-3, hi, np.nan)
            ob[ak] = np.clip(oi * delta, 0, None).astype(np.float32)
        out[bk] = ob
    out[:, ~good] = np.nan
    save_da(out, pt.values, cells, kind, fo)
    return f'-> {os.path.basename(fo)} {out.shape}  {(time.time()-t0)/60:.0f} min'


def stage_qdm(workers, models, kinds):
    tasks = [(m, k, tg) for m in models for k in kinds for tg in TAGS]
    log(f'qdm: {len(tasks)} tasks, {workers} workers')
    run_pool(build_qdm, tasks, workers)


# ====================================================================
# Stage 6: check -- REF-period exceedance days, ERA5 vs corrected vs uncorrected
# ====================================================================
def stage_check(models, kinds):
    os.makedirs(TB, exist_ok=True)
    cs = pd.read_csv(os.path.join(OUT, 'cells_sea.csv'))
    good = cs['good'].values.astype(bool); w = cs['w_area'].values[good]; w = w / w.sum()
    ny = REF[1] - REF[0] + 1; rows = []
    for kind in kinds:
        ov = open_da(os.path.join(OUT, f'obs_{kind}.nc')).values[:, good]
        for model in models:
            q = open_da(os.path.join(OUT, f'qdm_{kind}_{model}_his.nc'))
            qt = pd.to_datetime(q.time.values); qk = (qt.year >= REF[0]) & (qt.year <= REF[1])
            qv = q.values[qk][:, good]
            r = open_da(os.path.join(OUT, f'cm_{kind}_{model}_his.nc'))
            rt = pd.to_datetime(r.time.values); rk = (rt.year >= REF[0]) & (rt.year <= REF[1])
            rv = r.values[rk][:, good]
            for c in CUTS[kind]:
                rows.append(dict(variable=kind, model=model, cut=c,
                                 era5=float(((ov >= c).sum(0) / ny) @ w),
                                 qdm=float(((qv >= c).sum(0) / ny) @ w),
                                 raw=float(((rv >= c).sum(0) / ny) @ w)))
    d = pd.DataFrame(rows); d['qdm_minus_era5_pct'] = (d.qdm / d.era5 - 1) * 100
    d.to_csv(os.path.join(TB, 'check_ref_exceedance.csv'), index=False)
    log('check: REF-period global area-weighted exceedance days (d/yr)\n' + d.round(2).to_string(index=False))


# ====================================================================
# Stage 7: annual -- per-cell annual exceedance days + global area-weighted series
# ====================================================================
def build_annual(model, kind, tag):
    fo = os.path.join(OUT, f'annual_days_{kind}_{model}_{tag}.nc')
    if os.path.exists(fo):
        return f'[CACHE] {os.path.basename(fo)}'
    q = open_da(os.path.join(OUT, f'qdm_{kind}_{model}_{tag}.nc'))
    t = pd.to_datetime(q.time.values); yr = t.year.values; v = q.values
    u = np.unique(yr); idx = np.searchsorted(yr, u); ok = np.isfinite(v)
    ndays = np.add.reduceat(ok.astype(np.float32), idx, 0)
    out = np.full((len(u), v.shape[1], len(CUTS[kind])), np.nan, np.float32)
    for j, c in enumerate(CUTS[kind]):
        cnt = np.add.reduceat((ok & (v >= c)).astype(np.float32), idx, 0)
        out[:, :, j] = np.where(ndays >= 300, cnt, np.nan)     # >65 missing days in a year → NaN
    da = xr.DataArray(out, dims=('year', 'cell', 'cut'),
                      coords={'year': u, 'cell': q.cell.values, 'cut': CUTS[kind]}, name='days')
    da.to_netcdf(fo, encoding={'days': {'zlib': True, 'complevel': NC_COMP}})
    return f'-> {os.path.basename(fo)} {out.shape}'


def stage_annual(workers, models, kinds):
    os.makedirs(TB, exist_ok=True)
    tasks = [(m, k, tg) for m in models for k in kinds for tg in TAGS]
    run_pool(build_annual, tasks, workers)
    cs = pd.read_csv(os.path.join(OUT, 'cells_sea.csv'))
    good = cs['good'].values.astype(bool); w = cs['w_area'].values * good
    rows = []
    for (m, k, tg) in tasks:
        fo = os.path.join(OUT, f'annual_days_{k}_{m}_{tg}.nc')
        if not os.path.exists(fo):
            continue
        da = xr.open_dataarray(fo).load()
        for j, c in enumerate(CUTS[k]):
            a = da.values[:, :, j]; ok = np.isfinite(a)
            val = np.nansum(a * w, 1) / (ok * w).sum(1)         # missing cells get no weight
            rows.append(pd.DataFrame({'year': da.year.values, 'cut': c, 'value': val,
                                      'variable': k, 'model': m, 'scenario': tg}))
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(os.path.join(TB, 'annual_global_series.csv'), index=False)
    log(f'annual: -> tables\\annual_global_series.csv {d.shape} (value = GOOD-cell area-weighted annual exceedance days; '
        f'base/delta/pct computed per reference period in plotting code)')


# ====================================================================
# Stage 8: compare -- consistency with regional _proj_main on overlapping cells
# ====================================================================
def stage_compare(models, kinds):
    os.makedirs(TB, exist_ok=True)
    fold = os.path.join(OLD, 'obs_wave.nc')
    if not os.path.exists(fold):
        log('compare: regional output not found, skipping'); return
    rows = []
    for kind in kinds:
        for model in models:
            for tag in ('his', 'ssp585'):
                fn = os.path.join(OUT, f'qdm_{kind}_{model}_{tag}.nc')
                fo = os.path.join(OLD, f'qdm_{kind}_{model}_{tag}.nc')
                if not (os.path.exists(fn) and os.path.exists(fo)):
                    continue
                new = xr.open_dataarray(fn); old = xr.open_dataarray(fo)
                common = np.intersect1d(new.cell.values, old.cell.values)
                tsel = old.time.values[::37][:300]                  # sample 300 days
                a = new.sel(cell=common, time=tsel).values; b = old.sel(cell=common, time=tsel).values
                both = np.isfinite(a) & np.isfinite(b)
                d = np.abs(a - b)[both]
                rows.append(dict(variable=kind, model=model, tag=tag, n_cells=len(common),
                                 n_pairs=int(both.sum()),
                                 only_new_nan=int((~np.isfinite(a) & np.isfinite(b)).sum()),
                                 only_old_nan=int((np.isfinite(a) & ~np.isfinite(b)).sum()),
                                 max_abs_diff=float(d.max()) if d.size else np.nan,
                                 frac_diff_gt_1e3=float((d > 1e-3).mean()) if d.size else np.nan))
                new.close(); old.close()
    d = pd.DataFrame(rows); d.to_csv(os.path.join(TB, 'compare_regional.csv'), index=False)
    log('compare: sampled comparison with regional version on overlapping cells (max_abs_diff should be ~0; only_new_nan mostly reflects GOOD-definition differences)\n'
        + d.to_string(index=False))


# ====================================================================
# Parallel execution
# ====================================================================
def run_pool(fn, tasks, workers):
    if workers <= 1:
        for t in tasks:
            log(f'{fn.__name__}{t}: {fn(*t)}')
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, *t): t for t in tasks}
        for fu in as_completed(futs):
            t = futs[fu]
            try:
                log(f'{fn.__name__}{t}: {fu.result()}')
            except Exception as e:
                log(f'{fn.__name__}{t}: FAIL {type(e).__name__}: {e}')


# ====================================================================
# CLI
# ====================================================================
ALL = ['cells', 'cm', 'obs', 'good', 'qdm', 'check', 'annual', 'compare']


def main():
    ap = argparse.ArgumentParser(description='Global QDM bias correction')
    ap.add_argument('--stage', default='all', help='comma-separated: ' + ','.join(ALL) + ' or all')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--models', default='ACCESS,ECEARTH')
    ap.add_argument('--kinds', default='wave,wind')
    a = ap.parse_args()
    stages = ALL if a.stage == 'all' else [s.strip() for s in a.stage.split(',')]
    models = [m.strip() for m in a.models.split(',')]
    kinds = [k.strip() for k in a.kinds.split(',')]
    os.makedirs(OUT, exist_ok=True); os.makedirs(TB, exist_ok=True)
    log(f'==== global_qdm  stages={stages} workers={a.workers} models={models} kinds={kinds}')
    log(f'OUT = {OUT}')
    T0 = time.time()
    for s in stages:
        t0 = time.time(); log(f'---- stage {s}')
        if s == 'cells':   stage_cells()
        elif s == 'cm':    stage_cm(a.workers, models, kinds)
        elif s == 'obs':   stage_obs(a.workers, kinds)
        elif s == 'good':  stage_good(models, kinds)
        elif s == 'qdm':   stage_qdm(a.workers, models, kinds)
        elif s == 'check': stage_check(models, kinds)
        elif s == 'annual': stage_annual(a.workers, models, kinds)
        elif s == 'compare': stage_compare(models, kinds)
        else:
            log(f'unknown stage {s}'); continue
        log(f'---- stage {s} done {(time.time()-t0)/60:.1f} min')
    log(f'==== all done {(time.time()-T0)/3600:.2f} h')


if __name__ == '__main__':
    main()
