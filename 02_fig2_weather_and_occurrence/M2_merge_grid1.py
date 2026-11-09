import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
# Script version of the M2_merge pipeline, GRID_SIZE=1.0; writes only the grid1 cache directory, leaves grid2 untouched
import os, re, glob, warnings, time
import numpy as np, pandas as pd, xarray as xr
import rasterio
from rasterio.enums import Resampling
warnings.filterwarnings('ignore')
T0 = time.time()
def say(*a): print(f'[{(time.time()-T0)/60:6.1f} min]', *a, flush=True)

ACC_PARQUET = rf'{DATA_ROOT}/AIS_processing/accident_data/accident_cargo_clean_2002_2021.parquet'
GMTDS_ORG   = rf'{DATA_ROOT}/climate_AIS_data/GMTDS/cargo_ship/organized'
WAVE_DIR    = rf'{DATA_ROOT}/AIS_processing/weather_data/wave'
WIND_DIR    = rf'{DATA_ROOT}/AIS_processing/weather_data/wind'
CACHE_BASE  = rf'{DATA_ROOT}/AIS_processing/panel_cache_M2'
GRID_SIZE     = 1.0
YEARS         = list(range(2011, 2022))
DECIMATE      = 4
TRAFFIC_FLOOR = 0.5
REGIONS = {
    'NW_Europe': {'Iberia_Atlantic': [(36, 45, -15, -6)], 'Biscay_Celtic': [(45, 63, -15, -6), (43, 63, -6, 0)], 'North_Sea': [(48, 63, 0, 13)]},
    'NW_Atlantic': {'GoM_Caribbean': [(18, 31, -98, -60)], 'NE_Coast': [(31, 42, -82, -60), (42, 50, -67, -60)], 'Newfoundland': [(45, 55, -60, -46)]},
    'Asia': {'SE_Asia': [(-10, 20, 95, 130)], 'E_Asia': [(20, 46, 117, 146)]},
    'Med_Black': {'Med_West': [(30, 43, -6, 0), (30, 46, 0, 19)], 'Med_East': [(30, 46, 19, 26.5), (30, 41, 26.5, 37)], 'Black': [(41, 47.5, 26.5, 42)]},
    'Baltic': {'Baltic': [(53, 66, 13, 30)]},
    'Indian_Ocean_N': {'Arabian': [(5, 30, 50, 78)], 'BoB': [(5, 26, 78, 95)]},
}
FLAT_BOXES = [(r, s, b) for r, subs in REGIONS.items() for s, bs in subs.items() for b in bs]
SUBREGIONS = [s for r, subs in REGIONS.items() for s in subs]
CACHE = os.path.join(CACHE_BASE, f'grid{GRID_SIZE:g}')
assert CACHE.endswith('grid1'), CACHE
os.makedirs(CACHE, exist_ok=True)
def cpath(region, name): return os.path.join(CACHE, f'{region}__{name}.parquet')
def to_cell(lat, lon, G=GRID_SIZE):
    return np.round(np.floor(lat/G)*G + G/2, 3), np.round(np.floor(lon/G)*G + G/2, 3)
def gid(clat, clon): return f'{clat:.3f}_{clon:.3f}'
def assign_rs(lat, lon):
    for r, s, (S, N, W, E) in FLAT_BOXES:
        if S <= lat < N and W <= lon < E: return r, s
    return None, None
def boxes_of(region, sub=None):
    if sub is not None: return list(REGIONS[region][sub])
    return [b for s in REGIONS[region] for b in REGIONS[region][s]]
def region_bounds(region, sub=None):
    bs = boxes_of(region, sub)
    return (min(b[0] for b in bs), max(b[1] for b in bs), min(b[2] for b in bs), max(b[3] for b in bs))

PRIMARY_MAP = {'Machinery damage': 'Machinery', 'grounding': 'Grounding', 'collision': 'Collision', 'contact': 'Contact',
               'Fire/explosion': 'Fire_Explosion', 'Foundered': 'Foundered', 'Hull damage': 'Hull_Damage'}
NONWX_RGX = r'piracy|arrests|seizure|labour dispute|detained|port state|missing/overdue|war loss'
TYPES_WX = ['Grounding', 'Collision', 'Contact', 'Foundered', 'Hull_Damage']
TYPES_PLACEBO = ['Machinery', 'Piracy_Other']
TYPES_ALL = list(dict.fromkeys(list(PRIMARY_MAP.values()) + ['Piracy_Other', 'Misc']))
CAUSE_TOKEN = {
    'Wrecked/stranded (aground)': 'ANY_grounding', 'Collision (involving vessels)': 'ANY_collision',
    'Contact (eg. Harbour wall)': 'ANY_contact', 'Hull damage (holed, cracks, structural failure)': 'ANY_hull',
    'Foundered (sunk, submerged)': 'ANY_founder', 'Fire/explosion': 'ANY_fire',
    'Machinery damage/failure (e.g. lost rudder, fouled propellor)': 'ANY_machinery', 'Piracy': 'ANY_piracy'}
def split_causes(s):
    out, buf, depth = [], '', 0
    for ch in str(s):
        if ch == '(': depth += 1
        elif ch == ')': depth -= 1
        if ch == ',' and depth == 0:
            if buf.strip(): out.append(buf.strip())
            buf = ''
        else: buf += ch
    if buf.strip(): out.append(buf.strip())
    return out
def label_acc_type(df):
    t = df['cause_subcategory'].fillna('other').map(PRIMARY_MAP)
    is_nw = df['cause'].fillna('').str.lower().str.contains(NONWX_RGX, regex=True)
    return t.fillna(pd.Series(np.where(is_nw, 'Piracy_Other', 'Misc'), index=df.index))
def add_any_flags(df):
    sets = df['cause'].fillna('').map(lambda s: set(split_causes(s)))
    for tok, col in CAUSE_TOKEN.items(): df[col] = sets.map(lambda S, t=tok: int(t in S))
    df['n_causes'] = sets.map(len).astype(int)
    return df
CNT_COLS = (['A_it', 'A_wx', 'A_placebo'] + [f'A_{t}' for t in TYPES_ALL] + list(CAUSE_TOKEN.values()) + ['n_causes']
            + ['A_serious', 'A_wx_serious', 'A_tloss', 'A_tloss_official', 'A_actual_loss', 'A_poll']
            + ['n_dead_sum', 'n_injured_sum', 'n_missing_sum', 'n_lifelost_sum', 'A_fatal', 'A_injury', 'A_cas_reported', 'A_cas_zero']
            + ['dwt_sum', 'gt_sum', 'A_dwt_known'])
say('setup ok', CACHE)

# ── tif index ──
mani = pd.read_csv(os.path.join(GMTDS_ORG, '_manifest.csv')); mani.columns = [c.strip().lstrip('\ufeff') for c in mani.columns]
pat = re.compile(r'Cargo_b(\d+)_(\d{4})_(\d{1,2})\.tif$', re.I); rows = []
for p in glob.glob(os.path.join(GMTDS_ORG, '**', '*.tif'), recursive=True):
    m = pat.search(os.path.basename(p))
    if m and int(m.group(2)) in YEARS: rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), p))
tif_idx = pd.DataFrame(rows, columns=['box_id','year','month','path']).drop_duplicates(['box_id','year','month'])
def boxes_of_region(region):
    out = []
    for t in mani.itertuples():
        if any(min(t.N, N) > max(t.S, S) and min(t.E, E) > max(t.W, W) for (S, N, W, E) in boxes_of(region)): out.append(int(t.box_id))
    return sorted(set(out))
say('tif index', len(tif_idx))

# ── sea cells ──
def build_sea_cells():
    cp = os.path.join(CACHE, '_sea_cells.parquet')
    if os.path.exists(cp): return pd.read_parquet(cp)
    ds = xr.open_dataset(os.path.join(WAVE_DIR, f'era5_swh_{YEARS[0]}.nc'))
    la, lo = ds['latitude'].values, ds['longitude'].values
    lo180 = np.where(lo > 180, lo - 360, lo)
    ocean = ~np.isnan(ds['swh'].isel(valid_time=0).values); ds.close()
    LO, LA = np.meshgrid(lo180, la); CLA, CLO = to_cell(LA, LO)
    cells = pd.DataFrame({'grid_lat': CLA[ocean], 'grid_lon': CLO[ocean]}).drop_duplicates()
    rs = [assign_rs(a, o) for a, o in zip(cells.grid_lat, cells.grid_lon)]
    cells['region'] = [x[0] for x in rs]; cells['subregion'] = [x[1] for x in rs]
    cells = cells[cells.region.notna()].copy()
    cells['grid_id'] = cells.grid_lat.map('{:.3f}'.format) + '_' + cells.grid_lon.map('{:.3f}'.format)
    cells = cells[['region','subregion','grid_id','grid_lat','grid_lon']].reset_index(drop=True)
    assert cells.grid_id.is_unique and cells.subregion.nunique() == len(SUBREGIONS)
    cells.to_parquet(cp, index=False); return cells
sea_cells = build_sea_cells()
CELL2RS = sea_cells.set_index('grid_id')[['region','subregion']]
SUBMAP = dict(zip(sea_cells.grid_id, sea_cells.subregion))
say('sea cells', len(sea_cells)); print(sea_cells.groupby('region').size().to_string(), flush=True)

# ── Ait ──
def build_Ait():
    cp = os.path.join(CACHE, '_Ait.parquet')
    if os.path.exists(cp): return pd.read_parquet(cp)
    df = pd.read_parquet(ACC_PARQUET).dropna(subset=['lat', 'lon']); df = df[df.year.between(YEARS[0], YEARS[-1])].copy()
    clat, clon = to_cell(df.lat.values, df.lon.values); df['grid_id'] = [gid(a, o) for a, o in zip(clat, clon)]
    n0 = len(df); df = df.join(CELL2RS, on='grid_id'); df = df[df.region.notna()].copy()
    say(f'  accidents {n0:,} -> in study sea cells {len(df):,}')
    df['year_month'] = pd.to_datetime(df['casualty_date']).dt.strftime('%Y-%m')
    df['acc_type'] = label_acc_type(df); df = add_any_flags(df)
    df['A_it'] = 1; df['A_wx'] = df.acc_type.isin(TYPES_WX).astype(int); df['A_placebo'] = df.acc_type.isin(TYPES_PLACEBO).astype(int)
    for t in TYPES_ALL: df[f'A_{t}'] = (df.acc_type == t).astype(int)
    df['A_serious'] = df.serious.astype(int); df['A_wx_serious'] = (df.A_wx.astype(bool) & df.serious).astype(int)
    df['A_poll'] = df.pollution.astype(int); df['A_tloss_official'] = df.loss_type.notna().astype(int)
    df['A_actual_loss'] = (df.loss_type == 'Actual total loss').astype(int)
    df['A_tloss'] = (df.loss_type.notna() | (df.acc_type == 'Foundered')).astype(int)
    dead, inj, miss = df.n_dead, df.n_injured, df.n_missing
    df['n_dead_sum'] = dead.fillna(0); df['n_injured_sum'] = inj.fillna(0); df['n_missing_sum'] = miss.fillna(0)
    df['n_lifelost_sum'] = dead.fillna(0) + miss.fillna(0)
    df['A_fatal'] = ((dead.fillna(0) + miss.fillna(0)) > 0).astype(int); df['A_injury'] = (inj.fillna(0) > 0).astype(int)
    _rep = dead.notna() | inj.notna() | miss.notna(); df['A_cas_reported'] = _rep.astype(int)
    df['A_cas_zero'] = (_rep & (dead.fillna(0)+inj.fillna(0)+miss.fillna(0) == 0)).astype(int)
    df['dwt_sum'] = df.DWT.clip(lower=0); df['gt_sum'] = df.GT.clip(lower=0); df['A_dwt_known'] = (df.DWT > 0).astype(int)
    Ait = df.groupby(['region', 'subregion', 'grid_id', 'year_month'])[CNT_COLS].sum().reset_index()
    Ait.to_parquet(cp, index=False); return Ait
Ait = build_Ait()
assert sum(int(Ait[f'A_{t}'].sum()) for t in TYPES_ALL) == int(Ait.A_it.sum())
say('Ait', len(Ait), 'accidents', int(Ait.A_it.sum()))

# ── traffic E_it ──
def _bin_tif_to_cells(path, G=GRID_SIZE, dec=DECIMATE):
    with rasterio.open(path) as src:
        oh, ow = max(1, src.height//dec), max(1, src.width//dec)
        arr = src.read(1, out_shape=(oh, ow), resampling=Resampling.average, masked=True)
        T = src.transform * src.transform.scale(src.width/ow, src.height/oh)
    valid = ~arr.mask if np.ma.isMaskedArray(arr) else np.isfinite(arr)
    data = arr.data if np.ma.isMaskedArray(arr) else arr
    ny, nx = arr.shape
    xs = T.c + (np.arange(nx)+0.5)*T.a; ys = T.f + (np.arange(ny)+0.5)*T.e
    clon = np.floor(xs/G)*G + G/2; clat = np.floor(ys/G)*G + G/2
    d = pd.DataFrame({'grid_lat': np.round(np.broadcast_to(clat[:,None], arr.shape)[valid], 3),
                      'grid_lon': np.round(np.broadcast_to(clon, arr.shape)[valid], 3), 'v': data[valid]})
    g = d.groupby(['grid_lat','grid_lon'])['v'].agg(psum='sum', pcount='count').reset_index()
    g['grid_id'] = [gid(a, o) for a, o in zip(g.grid_lat, g.grid_lon)]
    return g
def build_Eit(region):
    cp = cpath(region, 'Eit')
    if os.path.exists(cp): return pd.read_parquet(cp)
    sea_ids = set(sea_cells[sea_cells.region == region].grid_id)
    sub = tif_idx[tif_idx.box_id.isin(boxes_of_region(region))]; out = []
    for (y, mo), grp in sub.groupby(['year','month']):
        acc = {}
        for p in grp.path:
            try: g = _bin_tif_to_cells(p)
            except Exception as e: print('skip', os.path.basename(p), e, flush=True); continue
            for r in g.itertuples(index=False):
                if r.grid_id not in sea_ids: continue
                s, c = acc.get(r.grid_id, (0.0, 0)); acc[r.grid_id] = (s + r.psum, c + r.pcount)
        for k, (s, c) in acc.items(): out.append((region, k, f'{y}-{mo:02d}', s/c if c else np.nan))
    Eit = pd.DataFrame(out, columns=['region','grid_id','year_month','E_it'])
    Eit.to_parquet(cp, index=False); say(f'  [{region}] E_it {len(Eit)}'); return Eit
Eit_all = pd.concat([build_Eit(r) for r in REGIONS], ignore_index=True)
say('Eit done', len(Eit_all))

# ── ERA5 daily cell max ──
def _open_var(year, var):
    if var == 'wave':
        ds = xr.open_dataset(os.path.join(WAVE_DIR, f'era5_swh_{year}.nc'), chunks={'valid_time': 240}); da = ds['swh']
    else:
        ds = xr.open_dataset(os.path.join(WIND_DIR, f'era5_wind10m_{year}.grib'), engine='netcdf4', chunks={'valid_time': 240})
        da = np.sqrt(ds['u10']**2 + ds['v10']**2)
    da = da.squeeze(drop=True)
    da = da.assign_coords(longitude=(((da['longitude']+180) % 360) - 180)).sortby('longitude')
    return ds, da
def build_daily_cellmax(region, var):
    cp = cpath(region, f'dmax_{var}')
    if os.path.exists(cp): return pd.read_parquet(cp)
    S, N, W, E = region_bounds(region); sea_ids = set(sea_cells[sea_cells.region == region].grid_id); frames = []
    for y in YEARS:
        ds, da = _open_var(y, var)
        da = da.sel(latitude=slice(N, S), longitude=slice(W, E))
        dd = da.resample(valid_time='1D').max().load()
        clat = np.round(np.floor(dd.latitude.values /GRID_SIZE)*GRID_SIZE + GRID_SIZE/2, 3)
        clon = np.round(np.floor(dd.longitude.values/GRID_SIZE)*GRID_SIZE + GRID_SIZE/2, 3)
        dd = dd.assign_coords(clat=('latitude', clat), clon=('longitude', clon))
        cell = dd.groupby('clat').max().groupby('clon').max()
        df = cell.to_dataframe(name='dmax').reset_index()
        df['grid_id'] = df.clat.map('{:.3f}'.format) + '_' + df.clon.map('{:.3f}'.format)
        df = df[df.grid_id.isin(sea_ids)].copy(); df['date'] = pd.to_datetime(df['valid_time']).dt.normalize()
        frames.append(df[['grid_id','date','dmax']]); ds.close(); say(f'  {region}/{var} {y}')
    out = pd.concat(frames, ignore_index=True); out.to_parquet(cp, index=False); return out

PCTS = [('P95', 0.95), ('P99', 0.99)]
def build_extrdays(region):
    cp = cpath(region, 'extrdays')
    if os.path.exists(cp): return pd.read_parquet(cp)
    wave = build_daily_cellmax(region, 'wave').rename(columns={'dmax': 'wave'})
    wind = build_daily_cellmax(region, 'wind').rename(columns={'dmax': 'wind'})
    d = wave.merge(wind, on=['grid_id', 'date'], how='inner')
    d = d[np.isfinite(d.wave) & np.isfinite(d.wind)].copy()
    d['subregion'] = d.grid_id.map(SUBMAP); d = d[d.subregion.notna()].copy()
    d['cmonth'] = pd.to_datetime(d['date']).dt.month; d['year_month'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m')
    ccols, thr_rec = [], []
    for tag, q in PCTS:
        thr = d.groupby(['subregion', 'cmonth'])[['wave', 'wind']].quantile(q); thr.columns = ['thr_wave', 'thr_wind']
        idx = pd.MultiIndex.from_arrays([d.subregion, d.cmonth])
        tw = thr['thr_wave'].reindex(idx).values; ti = thr['thr_wind'].reindex(idx).values
        ew, ei = d.wave.values >= tw, d.wind.values >= ti
        d[f'wind_only_{tag}'] = (ei & ~ew).astype('int8'); d[f'wave_only_{tag}'] = (ew & ~ei).astype('int8')
        d[f'compound_{tag}'] = (ew & ei).astype('int8'); d[f'wind_or_wave_{tag}'] = (ew | ei).astype('int8')
        ccols += [f'{c}_{tag}' for c in ['wind_only', 'wave_only', 'compound', 'wind_or_wave']]
        t = thr.reset_index(); t['pct'] = tag; t['region'] = region; thr_rec.append(t)
    nd = d.groupby(['grid_id', 'year_month']).size().rename('n_days_extr')
    agg = pd.concat([nd, d.groupby(['grid_id', 'year_month'])[ccols].sum()], axis=1).reset_index()
    agg['region'] = region; agg = agg.merge(sea_cells[['grid_id', 'subregion']], on='grid_id', how='left')
    agg.to_parquet(cp, index=False); pd.concat(thr_rec).to_parquet(cpath(region, 'thresholds'), index=False)
    say(f'  [{region}] extrdays {len(agg):,}'); return agg
Extr_all = pd.concat([build_extrdays(r) for r in REGIONS], ignore_index=True)
say('extrdays done')

BINS_VER = 'v4'
WAVE_BINS = [(0, 1.25), (1.25, 2.5), (2.5, 4), (4, 6), (6, 9), (9, np.inf)]
WIND_BINS = [(0, 8.0), (8.0, 10.8), (10.8, 13.9), (13.9, 17.2), (17.2, 20.8), (20.8, np.inf)]
def _bl(pre, lo, hi): return f"{pre}{lo}_{'p' if np.isinf(hi) else hi}".replace('.', 'p')
WV_COLS = [_bl('Wv_', *b) for b in WAVE_BINS]; WD_COLS = [_bl('Wd_', *b) for b in WIND_BINS]
WV_EDGES = [WAVE_BINS[0][0]] + [hi for _, hi in WAVE_BINS]; WD_EDGES = [WIND_BINS[0][0]] + [hi for _, hi in WIND_BINS]
def build_bins(region):
    cp = cpath(region, f'bins_{BINS_VER}')
    if os.path.exists(cp): return pd.read_parquet(cp)
    wave = build_daily_cellmax(region, 'wave').rename(columns={'dmax': 'wave'})
    wind = build_daily_cellmax(region, 'wind').rename(columns={'dmax': 'wind'})
    d = wave.merge(wind, on=['grid_id', 'date'], how='inner'); d = d[np.isfinite(d.wave) & np.isfinite(d.wind)].copy()
    d['year_month'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m')
    d['wv'] = pd.cut(d.wave, bins=WV_EDGES, right=False, labels=WV_COLS); d['wd'] = pd.cut(d.wind, bins=WD_EDGES, right=False, labels=WD_COLS)
    d = d.dropna(subset=['wv', 'wd']); k = ['grid_id', 'year_month']
    nd = d.groupby(k, observed=True).size().rename('n_days')
    A = d.groupby(k, observed=True)['wv'].value_counts().unstack(fill_value=0).reindex(columns=WV_COLS, fill_value=0).reindex(nd.index, fill_value=0)
    B = d.groupby(k, observed=True)['wd'].value_counts().unstack(fill_value=0).reindex(columns=WD_COLS, fill_value=0).reindex(nd.index, fill_value=0)
    agg = pd.concat([nd, A, B], axis=1).reset_index()
    agg[WV_COLS + WD_COLS + ['n_days']] = agg[WV_COLS + WD_COLS + ['n_days']].astype('int16')
    assert (agg[WV_COLS].sum(1) == agg.n_days).all() and (agg[WD_COLS].sum(1) == agg.n_days).all()
    agg['region'] = region; agg = agg.merge(sea_cells[['grid_id', 'subregion']], on='grid_id', how='left')
    agg.to_parquet(cp, index=False); say(f'  [{region}] bins {len(agg):,}'); return agg
Bins_all = pd.concat([build_bins(r) for r in REGIONS], ignore_index=True)
Bins_all.to_parquet(os.path.join(CACHE, f'bins_global_{BINS_VER}.parquet'), index=False)
say('bins done')

# ── panel ──
KEY = ['region', 'grid_id', 'year_month']
for nm, t in [('Eit', Eit_all), ('Extr', Extr_all), ('Bins', Bins_all), ('Ait', Ait)]:
    assert t.duplicated(KEY).sum() == 0, nm
n_acc_total = int(Ait.A_it.sum())
panel = Eit_all[Eit_all.E_it.notna()].copy(); say(f'1 E_it non-null {len(panel):,} / {len(Eit_all):,}')
n0 = len(panel); panel = panel.merge(Extr_all, on=KEY, how='inner'); say(f'2 merge Extr {len(panel):,} (drop {n0-len(panel):,})')
n0 = len(panel); panel = panel.merge(Bins_all.drop(columns=['subregion']), on=KEY, how='inner'); say(f'3 merge Bins {len(panel):,} (drop {n0-len(panel):,})')
panel = panel.merge(Ait.drop(columns=['subregion']), on=KEY, how='left')
n_acc_in = int(panel['A_it'].fillna(0).sum()); say(f'4 accidents in panel {n_acc_in:,} / {n_acc_total:,}')
panel[CNT_COLS] = panel[CNT_COLS].fillna(0)
for c in CNT_COLS: panel[c] = panel[c].astype('float32' if c in ['dwt_sum', 'gt_sum'] else 'int32')
for c in WV_COLS + WD_COLS + ['n_days']: panel[c] = panel[c].astype('int16')
panel['log_E'] = np.log(panel['E_it'] + TRAFFIC_FLOOR); panel['E_zero'] = (panel['E_it'] == 0).astype('int8')
panel = panel.merge(sea_cells[['grid_id','grid_lat','grid_lon']], on='grid_id', how='left')
panel['year'] = panel['year_month'].str[:4].astype('int16'); panel['month'] = panel['year_month'].str[-2:]; panel['month_num'] = panel['month'].astype('int8')
outp = os.path.join(CACHE, 'panel_global.parquet'); panel.to_parquet(outp, index=False)
say(f'PANEL DONE {panel.shape} -> {outp}')
print(panel.groupby(['region']).agg(obs=('A_it','size'), cells=('grid_id','nunique'), acc=('A_it','sum'), wx=('A_wx','sum')).to_string(), flush=True)
print('zero-accident cell-months %.1f%%' % ((panel.A_it == 0).mean()*100), flush=True)
