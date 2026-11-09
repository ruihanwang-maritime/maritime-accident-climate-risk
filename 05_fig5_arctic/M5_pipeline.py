import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
# M5 pipeline — wind-wave exposure in newly navigable Arctic waters (mainly non-ice-class ships)
# See M5_design.md for the design. Run cells (# %%) in order; after setting MODEL = 'ECEARTH', rerun from Cell 0.
# Rule: each cache is validated first (all keys, all files, aligned dims); if valid it is skipped, else rebuilt and saved.

# %% Cell 0 — configuration
import os, glob, warnings, numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib import cm
import cartopy.crs as ccrs, cartopy.feature as cfeature
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')
pd.set_option('display.width', 230); pd.set_option('display.max_columns', 40)

MODEL     = 'ACCESS'                                 # ← 'ACCESS' or 'ECEARTH'
BASE      = rf'{DATA_ROOT}/new_climate'
ARC_OUT   = rf'{BASE}\wave\_arctic'                  # wave/wind cache per grid cell per month
DAILY_OUT = rf'{BASE}\wave\_daily'                   # daily max Hs / U10 cache per grid cell (one file per year)
NAV_OUT   = rf'{BASE}\siconc\_nav'                   # sea ice / navigability cache
M5_OUT    = rf'{BASE}\_M5'                           # output tables
for d in (ARC_OUT, DAILY_OUT, NAV_OUT, M5_OUT): os.makedirs(d, exist_ok=True)

CFG = {
 'ACCESS':  dict(label='ACCESS-CM2',
                 wave={'his': rf'{BASE}\wave\ACCESS\CDFAC_his', 'ssp126': rf'{BASE}\wave\ACCESS\CDFAC1_126', 'ssp585': rf'{BASE}\wave\ACCESS\CDFAC1_585'},
                 wind={'his': rf'{BASE}\criso_wind\ACCESS-CM2\his', 'ssp126': rf'{BASE}\criso_wind\ACCESS-CM2\126', 'ssp585': rf'{BASE}\criso_wind\ACCESS-CM2\585'},
                 siday=rf'{BASE}\siconc\SIday\ACCESS-CM2', sit_mode='daily', sit_monthly=None),
 'ECEARTH': dict(label='EC-Earth3',
                 wave={'his': rf'{BASE}\wave\ec_earth\cdfac1\his', 'ssp126': rf'{BASE}\wave\ec_earth\cdfac1\126', 'ssp585': rf'{BASE}\wave\ec_earth\cdfac1\585'},
                 wind={'his': rf'{BASE}\criso_wind\EC-EARTH3\his', 'ssp126': rf'{BASE}\criso_wind\EC-EARTH3\126', 'ssp585': rf'{BASE}\criso_wind\EC-EARTH3\585'},
                 siday=rf'{BASE}\siconc\SIday\EC-Earth3', sit_mode='monthly',
                 sit_monthly={'his': rf'{BASE}\siconc\sithick_ECEARTH_historical.nc',
                              'ssp126': rf'{BASE}\siconc\sithick_ECEARTH_ssp126.nc',
                              'ssp585': rf'{BASE}\siconc\sithick_ECEARTH_ssp585.nc'}),
}[MODEL]
MDL_LABEL = CFG['label']
TAGS      = ['his', 'ssp126', 'ssp585']
SSPS      = ['ssp126', 'ssp585']
EXP       = {'his': 'historical', 'ssp126': 'ssp126', 'ssp585': 'ssp585'}
YEARS     = {'his': (1985, 2014), 'ssp126': (2015, 2100), 'ssp585': (2015, 2100)}
SSP_LABEL = {'ssp126': 'SSP1-2.6', 'ssp585': 'SSP5-8.5'}
PERIODS   = [('1985–2014', 'his', 1985, 2014), ('2021–2040', None, 2021, 2040), ('2041–2060', None, 2041, 2060),
             ('2061–2080', None, 2061, 2080), ('2081–2100', None, 2081, 2100)]
LAT0 = 60.0
# Sectors approximate IHO sea areas; NEP/NWP boundary at Point Barrow, Alaska, 156°W (Beaufort Sea assigned to NWP)
SECTORS = {'Barents-Kara': (66.5, 82, 15, 100), 'Laptev-E.Siberian': (66.5, 82, 100, 180),
           'Chukchi': (66.5, 82, -180, -156), 'Beaufort-Canadian Arctic': (66.5, 82, -156, -45),
           'Greenland-Fram': (66.5, 82, -45, 15), 'Central Arctic': (82, 90, -180, 180)}
PASSAGES = {'NEP (Northeast Passage)':   ['Barents-Kara', 'Laptev-E.Siberian', 'Chukchi'],
            'NWP (Northwest Passage)':   ['Beaufort-Canadian Arctic'],
            'TSR (Transpolar Sea Route)': ['Central Arctic'],
            'Fram Strait approach':      ['Greenland-Fram']}
PASS_COLOR = {'NEP (Northeast Passage)': '#c8452b', 'NWP (Northwest Passage)': '#2e9e78',
              'TSR (Transpolar Sea Route)': '#7b5aa6', 'Fram Strait approach': '#2c6e91'}
SHOW = list(PASSAGES)                                # main text uses only the four passages; 'Pan-Arctic' still usable in pass_sel for totals
WAYPOINTS = {
 'NEP (Northeast Passage)':   [(69.5, 35), (71.5, 55), (73.5, 68), (74.0, 80), (77.7, 104), (75.5, 115), (74.5, 128), (74.7, 140), (72.5, 155), (70.5, 170), (69.5, 180), (66.5, -169)],
 'NWP (Northwest Passage)':   [(74.0, -70), (74.2, -82), (74.3, -95), (74.5, -108), (74.8, -120), (72.0, -135), (71.5, -156), (66.5, -169)],
 'TSR (Transpolar Sea Route)': [(78.0, 0), (86.0, 0), (89.5, 0), (89.5, 180), (86.0, -175), (76.0, -170), (66.5, -169)],
}
PASS_LABEL_POS = {'NEP (Northeast Passage)': (75, 110), 'NWP (Northwest Passage)': (75, -95),
                  'TSR (Transpolar Sea Route)': (86, -60), 'Fram Strait approach': (75, -12)}

WMO = ['hs_lt1p25', 'hs_1p25_2p5', 'hs_2p5_4', 'hs_4_6', 'hs_6_9', 'hs_ge9']        # same Hs bins as M2 Table 1
WMO_EDGES = [0, 1.25, 2.5, 4.0, 6.0, 9.0, np.inf]; WMO_LAB = ['<1.25 m', '1.25–2.5 m', '2.5–4 m', '4–6 m', '6–9 m', '≥9 m']
BFT = ['u_lt8', 'u_8_10p8', 'u_10p8_13p9', 'u_13p9_17p2', 'u_17p2_20p8', 'u_ge20p8']
BFT_EDGES = [0, 8.0, 10.8, 13.9, 17.2, 20.8, np.inf]; BFT_LAB = ['<Bft 5', 'Bft 5', 'Bft 6', 'Bft 7', 'Bft 8', '≥Bft 9']
COMP = {'cp_hs2p5_u13p9': (2.5, 13.9), 'cp_hs2p5_u17p2': (2.5, 17.2), 'cp_hs4_u20p8': (4.0, 20.8)}
HS_ROUGH  = ['hs_2p5_4', 'hs_4_6', 'hs_6_9', 'hs_ge9']   # main wave threshold Hs ≥ 2.5 m (lowest significant M2 bin)
U_STRONG  = ['u_13p9_17p2', 'u_17p2_20p8', 'u_ge20p8'] # main wind threshold U10 ≥ 13.9 m/s (Bft 7, lowest significant M2 bin)
U_GALE    = ['u_17p2_20p8', 'u_ge20p8']               # sensitivity: Bft 8 gale-warning level

# POLARIS (IMO MSC.1/Circ.1519 Table 1.3); thickness -> ice type by WMO stages; for thick ice, thickness proxies ice age
SIT_EDGES = [0.10, 0.15, 0.30, 0.50, 0.70, 1.00, 1.20, 2.00, 2.50, 3.00]
RIV_TABLE = {'NOICE': [1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -8],
             'PC7':   [2, 2,  2,  1,  1,  0, -1, -2, -3, -3, -3],
             'PC6':   [2, 2,  2,  2,  1,  1,  0, -1, -2, -3, -3]}
CRIT = ['OW_polaris', 'PC6_polaris', 'PC7_polaris', 'sic_lt15', 'icefree_1pct']
LABEL = {'OW_polaris': 'No ice class (POLARIS RIO≥0)', 'PC6_polaris': 'PC6 (POLARIS RIO≥0)',
         'PC7_polaris': 'PC7 (POLARIS RIO≥0)', 'sic_lt15': 'SIC < 15%', 'icefree_1pct': 'SIC < 1%',
         'wave_active': 'WW3 wave-active'}

# M2 Table 1 Hs-bin IRRs (A_it specification) and 95% CI, used in Section 5
IRR_M2 = {'hs_lt1p25': (1.0, 1.0, 1.0), 'hs_1p25_2p5': (1.0036, 0.995, 1.013), 'hs_2p5_4': (1.0227, 1.010, 1.036),
          'hs_4_6': (1.0462, 1.024, 1.069), 'hs_6_9': (1.0523, 1.007, 1.100), 'hs_ge9': (1.1989, 1.071, 1.342)}
BETA = {which: {k: np.log(v[i]) for k, v in IRR_M2.items()} for i, which in enumerate(['main', 'lo', 'hi'])}

# thresholds: new/traditional navigable areas, index sample size, coverage denominator
THR_NEW_FUT, THR_NEW_HIS, THR_HIS_NAV, MIN_WAVE_DAYS, MIN_NAV_FRAC = 90, 30, 90, 10, 0.05

# ── Checks ──
print(f'MODEL = {MODEL} ({MDL_LABEL})')
for tag in TAGS:
    nw = len(glob.glob(os.path.join(CFG['wave'][tag], 'ww3.*_hs.nc'))); nu = len(glob.glob(os.path.join(CFG['wind'][tag], 'ww3.*_wnd.nc')))
    nc = len(glob.glob(os.path.join(CFG['siday'], EXP[tag], 'siconc_*.nc')))
    nt = len(glob.glob(os.path.join(CFG['siday'], EXP[tag], 'sithick_*.nc'))) if CFG['sit_mode'] == 'daily' else int(os.path.exists(CFG['sit_monthly'][tag]))
    print(f'  {tag:7s} wave {nw:4d} monthly files | wind {nu:4d} | siconc {nc:3d} files | sithick {nt} ({CFG["sit_mode"]})')
    assert nw >= 360 and nu == nw and nc > 0 and nt > 0, f'{tag} data incomplete'
print('Check passed: data directories complete')

# %% Cell 1 — target grid and nearest-neighbour index
_f0 = sorted(glob.glob(os.path.join(CFG['wave']['his'], 'ww3.*_hs.nc')))[0]
_d  = xr.open_dataset(_f0); LAT, LON = _d.latitude.values, _d.longitude.values; MAP = _d['MAPSTA'].values; _d.close()
lon180 = ((LON + 180) % 360) - 180
LA, LO = np.meshgrid(LAT, lon180, indexing='ij')
OCEAN2D = np.isin(MAP, [1, 7]); INDOM = OCEAN2D & (LA >= LAT0)
FLAT  = np.flatnonzero(INDOM.ravel()); NP_ = len(FLAT)
P_LAT = LA.ravel()[FLAT]; P_LON = LO.ravel()[FLAT]
CELL_KM2 = (3090.0 * np.cos(np.deg2rad(P_LAT))).astype(np.float32)        # 0.5°×0.5° cell area
sec = np.full(LA.shape, '', dtype=object)
for nm, (S, N, W, E) in SECTORS.items(): sec[(LA >= S) & (LA < N) & (LO >= W) & (LO < E) & OCEAN2D] = nm
P_SEC  = sec.ravel()[FLAT]
P_PASS = np.full(NP_, '', dtype=object)
for p, secs in PASSAGES.items(): P_PASS[np.isin(P_SEC, secs)] = p

def xyz(la, lo):
    la, lo = np.deg2rad(la), np.deg2rad(lo)
    return np.stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)], -1)

_s = xr.open_dataset(sorted(glob.glob(os.path.join(CFG['siday'], 'historical', 'siconc_*.nc')))[0])
S_LAT = _s.latitude.values.ravel(); S_LON = ((_s.longitude.values.ravel() + 180) % 360) - 180
S_OK  = np.isfinite(S_LAT) & np.isfinite(S_LON) & np.isfinite(_s.siconc.isel(time=0).values.ravel())
S_SHAPE = _s.siconc.isel(time=0).shape; _s.close()
DIST, NN = cKDTree(xyz(S_LAT[S_OK], S_LON[S_OK])).query(xyz(P_LAT, P_LON), k=1); DIST *= 6371.0
FAR = DIST > 100.0

# grid back-fill helper (for maps)
ORD  = np.argsort(lon180); LONP = lon180[ORD]; LONW = np.concatenate([LONP, [LONP[0] + 360]])
KI   = LAT >= LAT0 - 1; LATP = LAT[KI]
def to_grid(v):
    g = np.full(LA.size, np.nan, np.float32); g[FLAT] = v
    G = g.reshape(LA.shape)[:, ORD][KI]
    return np.concatenate([G, G[:, :1]], axis=1)

# ── Checks ──
print(f'Target grid cells {NP_} (0.5° wave grid, ocean points ≥{LAT0:.0f}N); study domain area {CELL_KM2.sum()/1e6:.2f}×10⁶ km²')
print('Grid cells per passage:', pd.Series(np.where(P_PASS == '', '(outside)', P_PASS)).value_counts().to_dict())
print(f'Sea-ice grid {S_SHAPE}, valid ocean cells {S_OK.sum()} | nearest-neighbour distance median {np.median(DIST):.1f} km, p95 {np.percentile(DIST, 95):.1f} km, >100 km {FAR.sum()} points')
assert NP_ == 27375, 'Target grid cell count unexpected; check wave files'
assert np.median(DIST) < 30, 'Nearest-neighbour distance abnormal; check sea-ice coordinates'

# %% Cell 2 — extract daily sea ice to target grid cells (cache check: keys, days, cell count)
def monthly_sit_interpolator(tag):
    m = xr.open_dataset(CFG['sit_monthly'][tag])
    mlat = m.latitude.values.ravel(); mlon = ((m.longitude.values.ravel() + 180) % 360) - 180
    V = m.sithick.values.reshape(m.sizes['time'], -1); ok = np.isfinite(mlat) & np.isfinite(mlon)
    _, nn_m = cKDTree(xyz(mlat[ok], mlon[ok])).query(xyz(P_LAT, P_LON), k=1)
    Vm = np.nan_to_num(V[:, ok][:, nn_m], nan=0.0)
    tm = pd.to_datetime([f'{str(t)[:4]}-{str(t)[5:7]}-15' for t in m.time.values]); m.close()
    xm = (tm - pd.Timestamp('1900-01-01')).days.values.astype(float)
    f = interp1d(xm, Vm, axis=0, bounds_error=False, fill_value=(Vm[0], Vm[-1]))
    return lambda dates: f((dates - pd.Timestamp('1900-01-01')).days.values.astype(float))

def expected_days(tag):
    y0, y1 = YEARS[tag]; return (pd.Timestamp(f'{y1}-12-31') - pd.Timestamp(f'{y0}-01-01')).days + 1

def extract_exp(tag):
    fo = os.path.join(NAV_OUT, f'siday_{MODEL}_{tag}.npz')
    if os.path.exists(fo):
        z = np.load(fo)
        if {'date', 'sic', 'sit'} <= set(z.files) and z['sic'].shape[1] == NP_ and abs(z['sic'].shape[0] - expected_days(tag)) <= 1:
            print(f'[CACHE] {os.path.basename(fo)}'); return z
        z.close(); print(f'[REBUILD] {os.path.basename(fo)} failed validation')
    y0, y1 = YEARS[tag]
    fc = sorted(glob.glob(os.path.join(CFG['siday'], EXP[tag], 'siconc_*.nc')))
    ft = sorted(glob.glob(os.path.join(CFG['siday'], EXP[tag], 'sithick_*.nc'))) if CFG['sit_mode'] == 'daily' else None
    sit_of = monthly_sit_interpolator(tag) if CFG['sit_mode'] == 'monthly' else None
    dates, SIC, SIT, n_trace, n_miss = [], [], [], 0, 0
    for i, f_c in enumerate(fc):
        dc = xr.open_dataset(f_c); assert dc.latitude.shape == S_SHAPE, 'Sea-ice grid changed'
        t = pd.to_datetime(dc.time.values.astype('datetime64[D]')); k = (t.year >= y0) & (t.year <= y1)
        if not k.any(): dc.close(); continue
        c = dc.siconc.values[k].reshape(k.sum(), -1)[:, S_OK][:, NN] / 100.0; dc.close()
        if ft is not None:
            dt = xr.open_dataset(ft[i]); h = dt.sithick.values[k].reshape(k.sum(), -1)[:, S_OK][:, NN]; dt.close()
            tr = np.isnan(h) & (c <= 0.001); n_trace += int(tr.sum()); h = np.where(tr, 0.0, h)     # trace ice → thickness 0
        else:
            h = sit_of(t[k]).astype(np.float32)
        h = np.where(c <= 0, 0.0, h)
        n_miss += int((np.isnan(h) & np.isfinite(c)).sum())
        c[:, FAR] = np.nan; h[:, FAR] = np.nan
        SIC.append(np.where(np.isnan(c), 65535, np.round(c * 10000)).astype(np.uint16))
        SIT.append(np.where(np.isnan(h), 65535, np.round(np.clip(h, 0, 600) * 100)).astype(np.uint16))
        dates.append(t[k].strftime('%Y%m%d').astype(np.int32))
        if (i + 1) % 10 == 0: print(f'  {i+1}/{len(fc)}', flush=True)
    out = dict(date=np.concatenate(dates), sic=np.concatenate(SIC), sit=np.concatenate(SIT))
    np.savez_compressed(fo, **out)
    print(f'  -> {os.path.basename(fo)} {out["sic"].shape} | trace ice set to zero {n_trace:,} | ice present but thickness missing {n_miss:,}')
    return np.load(fo)

SID = {tag: extract_exp(tag) for tag in TAGS}

# ── Checks ──
for tag in TAGS:
    z = SID[tag]; nd = z['sic'].shape[0]; nan_share = (z['sic'] == 65535).mean()
    print(f'{tag:7s} days {nd} (expected {expected_days(tag)}±1) | cells {z["sic"].shape[1]} | missing share {nan_share:.4f}')
    assert abs(nd - expected_days(tag)) <= 1 and z['sic'].shape[1] == NP_ and nan_share < 0.02
print('Check passed: daily sea-ice cache complete')

# %% Cell 3 — POLARIS monthly/annual navigable days and longest continuous period (cache check: all keys)
def riv(T, ic):
    tab = np.array(RIV_TABLE[ic], np.float32)
    return np.where(T <= 0, 3.0, tab[np.clip(np.digitize(T, SIT_EDGES), 0, 10)])
def rio(sic, sit, ic):
    C = sic * 10.0
    return (10 - C) * 3 + C * riv(sit, ic)
def decode(z):
    sic = z['sic'].astype(np.float32); sic[sic == 65535] = np.nan; sic /= 10000.0
    sit = z['sit'].astype(np.float32); sit[sit == 65535] = np.nan; sit /= 100.0
    trace = np.isnan(sit) & np.isfinite(sic) & (sic <= 0.001)      # trace-ice safeguard: GCMs output no thickness where siconc≤0.1%
    sit[trace] = 0.0
    rest = int((np.isnan(sit) & np.isfinite(sic)).sum())
    print(f'  decode: trace ice set to zero {int(trace.sum()):,} | remaining ice present but thickness missing {rest:,}')
    return sic, sit
def longest_run(mask):
    run = np.zeros(mask.shape[1], np.int16); best = np.zeros(mask.shape[1], np.int16)
    for d in range(mask.shape[0]):
        run = np.where(mask[d], run + 1, 0); best = np.maximum(best, run)
    return best

NAV_KEYS = ['ym', 'year', 'ndays_m'] + [k + s for k in CRIT for s in ('_m', '_y')] + ['OW_maxrun_y', 'PC7_maxrun_y']
def navigable(tag):
    fo = os.path.join(NAV_OUT, f'nav_{MODEL}_{tag}.npz')
    if os.path.exists(fo):
        z = np.load(fo)
        if set(NAV_KEYS) <= set(z.files) and z['OW_polaris_m'].shape[1] == NP_: print(f'[CACHE] {os.path.basename(fo)}'); return z
        z.close(); print(f'[REBUILD] {os.path.basename(fo)} failed validation')
    sic, sit = decode(SID[tag]); ym = SID[tag]['date'] // 100; yr = ym // 100
    ok = np.isfinite(sic) & np.isfinite(sit)
    crit = {'OW_polaris': (rio(sic, sit, 'NOICE') >= 0) & ok, 'PC6_polaris': (rio(sic, sit, 'PC6') >= 0) & ok,
            'PC7_polaris': (rio(sic, sit, 'PC7') >= 0) & ok, 'sic_lt15': (sic < 0.15) & ok, 'icefree_1pct': (sic < 0.01) & ok}
    months, years = np.unique(ym), np.unique(yr)
    out = {'ym': months, 'year': years, 'ndays_m': np.array([(ym == m).sum() for m in months], np.int16)}
    for k, v in crit.items():
        out[k + '_m'] = np.stack([v[ym == m].sum(0) for m in months]).astype(np.int8)
        out[k + '_y'] = np.stack([v[yr == y].sum(0) for y in years]).astype(np.int16)
    out['OW_maxrun_y']  = np.stack([longest_run(crit['OW_polaris'][yr == y]) for y in years])
    out['PC7_maxrun_y'] = np.stack([longest_run(crit['PC7_polaris'][yr == y]) for y in years])
    np.savez_compressed(fo, **out); print(f'  -> {os.path.basename(fo)}'); return np.load(fo)

NAV = {tag: navigable(tag) for tag in TAGS}

# ── Checks ──
for tag in TAGS:
    z = NAV[tag]
    assert (z['icefree_1pct_m'] <= z['OW_polaris_m']).all(), f'{tag}: RIO<0 in ice-free grid cells'
    assert (z['OW_polaris_m'] <= z['PC7_polaris_m']).all() and (z['PC7_polaris_m'] <= z['PC6_polaris_m']).all(), f'{tag}: ship-class ordering violated'
    ysum = np.stack([z['OW_polaris_m'][z['ym'] // 100 == y].astype(int).sum(0) for y in z['year']])
    assert np.array_equal(ysum, z['OW_polaris_y']), f'{tag}: monthly/annual mismatch'
    print(f'{tag:7s} months {len(z["ym"])} years {len(z["year"])} | OW mean annual navigable days (domain mean) {z["OW_polaris_y"].mean():.1f}')
print('Check passed: ice-free ⊂ OW ⊂ PC7 ⊂ PC6, monthly/annual consistent')

# %% Cell 4 — wave cache: WMO bins of daily max Hs (cache check: all keys, aligned with NAV month series)
def wave_month(path):
    ds = xr.open_dataset(path, drop_variables=['MAPSTA'])
    v = ds['hs'].values.astype(np.float32); ds.close()
    nd = v.shape[0] // 8
    dm = np.nanmax(v.reshape(nd, 8, 361, 720), axis=1).reshape(nd, -1)[:, FLAT]
    ok = np.isfinite(dm)
    r = {'nvalid': ok.sum(0).astype(np.int8), 'maxhs': np.where(ok.any(0), np.nanmax(np.where(ok, dm, -9), 0), np.nan).astype(np.float32)}
    for i, nm in enumerate(WMO): r[nm] = (ok & (dm >= WMO_EDGES[i]) & (dm < WMO_EDGES[i+1])).sum(0).astype(np.int8)
    return r, nd

WAVE_KEYS = ['nvalid', 'maxhs', 'ym', 'ndays'] + WMO
def daily_file(tag, y): return os.path.join(DAILY_OUT, f'daily_{MODEL}_{tag}_{y}.npz')
def split_ge6(z, tag):
    """Old cache has only hs_ge6: split into 6–9 m / ≥9 m using Cell 5 daily Hs (cm), avoiding re-reading WW3"""
    out = {k: np.array(z[k]) for k in z.files}; ym = out['ym']; ge9 = np.zeros_like(out['hs_ge6'])
    for y in np.unique(ym // 100):
        d = np.load(daily_file(tag, y)); dm = d['date'] // 100; hs = d['hs']; d.close()
        for m in np.unique(dm):
            h = hs[dm == m]; ge9[np.flatnonzero(ym == m)[0]] = ((h != 65535) & (h >= 900)).sum(0)
    assert (ge9 <= out['hs_ge6']).all(), f'{tag}: ≥9 m days exceed ≥6 m days; daily and monthly caches inconsistent'
    out['hs_ge9'] = ge9.astype(np.int8); out['hs_6_9'] = (out.pop('hs_ge6') - ge9).astype(np.int8)
    return out

def build_wave(tag):
    fo = os.path.join(ARC_OUT, f'arctic_{MODEL}_{tag}.npz'); y0, y1 = YEARS[tag]
    if os.path.exists(fo):
        z = np.load(fo)
        if set(WAVE_KEYS) <= set(z.files) and np.array_equal(z['ym'], NAV[tag]['ym']): print(f'[CACHE] {os.path.basename(fo)}'); return z
        if ('hs_ge6' in z.files and np.array_equal(z['ym'], NAV[tag]['ym'])
                and all(os.path.exists(daily_file(tag, y)) for y in range(y0, y1 + 1))):
            out = split_ge6(z, tag); z.close(); np.savez_compressed(fo, **out)
            print(f'[UPGRADE] {os.path.basename(fo)}: hs_ge6 → hs_6_9 + hs_ge9 (from daily cache)'); return np.load(fo)
        z.close(); print(f'[REBUILD] {os.path.basename(fo)} failed validation')
    y0, y1 = YEARS[tag]
    files = [f for f in sorted(glob.glob(os.path.join(CFG['wave'][tag], 'ww3.*_hs.nc'))) if y0 <= int(os.path.basename(f)[4:8]) <= y1]
    keys = ['nvalid', 'maxhs'] + WMO; acc = {k: [] for k in keys}; ym, nds = [], []
    for i, f in enumerate(files):
        r, nd = wave_month(f)
        for k in keys: acc[k].append(r[k])
        ym.append(int(os.path.basename(f)[4:10])); nds.append(nd)
        if (i + 1) % 120 == 0: print(f'  {i+1}/{len(files)}', flush=True)
    out = {k: np.stack(v) for k, v in acc.items()}; out['ym'] = np.array(ym); out['ndays'] = np.array(nds)
    np.savez_compressed(fo, **out); print(f'  -> {os.path.basename(fo)} {out["nvalid"].shape}'); return np.load(fo)

WAV = {tag: build_wave(tag) for tag in TAGS}

# ── Checks ──
for tag in TAGS:
    z = WAV[tag]
    assert np.array_equal(z['ym'], NAV[tag]['ym']), f'{tag}: wave month series inconsistent with sea ice'
    assert np.array_equal(sum(z[k].astype(int) for k in WMO), z['nvalid'].astype(int)), f'{tag}: WMO bins do not sum to total'
    print(f'{tag:7s} {z["nvalid"].shape} | domain mean annual wave days {z["nvalid"].astype(float).sum() / len(NAV[tag]["year"]) / NP_:.1f}')
print('Check passed: wave cache aligned, bins sum to total')

# %% Cell 5 — wind cache + daily cache (six Beaufort bins, three compound-day bins, daily max Hs/U10 per-year files; 3–4 h per model)
def wind_month(fw, fh):
    w = xr.open_dataset(fw, drop_variables=['MAPSTA'])
    u = np.hypot(w['uwnd'].values, w['vwnd'].values).astype(np.float32); w.close()
    nd = u.shape[0] // 8
    um = np.nanmax(u.reshape(nd, 8, 361, 720), axis=1).reshape(nd, -1)[:, FLAT]
    h = xr.open_dataset(fh, drop_variables=['MAPSTA'])
    hm = np.nanmax(h['hs'].values.astype(np.float32).reshape(nd, 8, 361, 720), axis=1).reshape(nd, -1)[:, FLAT]; h.close()
    ok = np.isfinite(um) & np.isfinite(hm)
    r = {nm: (ok & (um >= BFT_EDGES[i]) & (um < BFT_EDGES[i+1])).sum(0).astype(np.int8) for i, nm in enumerate(BFT)}
    for nm, (ht, ut) in COMP.items(): r[nm] = (ok & (hm >= ht) & (um >= ut)).sum(0).astype(np.int8)
    r['nvalid_w'] = ok.sum(0).astype(np.int8); r['maxu'] = np.where(ok.any(0), np.nanmax(np.where(ok, um, -9), 0), np.nan).astype(np.float32)
    daily = {'hs': np.where(ok, np.round(np.clip(hm, 0, 600) * 100), 65535).astype(np.uint16),     # cm
             'u10': np.where(ok, np.round(np.clip(um, 0, 600) * 10), 65535).astype(np.uint16)}     # 0.1 m/s
    return r, nd, daily

WIND_KEYS = BFT + list(COMP) + ['nvalid_w', 'maxu', 'ym']
def build_wind(tag):
    fo = os.path.join(ARC_OUT, f'arctic_wind_{MODEL}_{tag}.npz'); y0, y1 = YEARS[tag]
    have_daily = all(os.path.exists(daily_file(tag, y)) for y in range(y0, y1 + 1))
    if os.path.exists(fo):
        z = np.load(fo)
        if set(WIND_KEYS) <= set(z.files) and np.array_equal(z['ym'], WAV[tag]['ym']) and have_daily:
            print(f'[CACHE] {os.path.basename(fo)} + {y1 - y0 + 1} daily files'); return z
        z.close(); print(f'[REBUILD] {os.path.basename(fo)}: ' + ('missing compound bins/keys' if not set(WIND_KEYS) <= set(z.files) else 'missing daily files'))
    fws = [f for f in sorted(glob.glob(os.path.join(CFG['wind'][tag], 'ww3.*_wnd.nc'))) if y0 <= int(os.path.basename(f)[4:8]) <= y1]
    keys = BFT + list(COMP) + ['nvalid_w', 'maxu']; acc = {k: [] for k in keys}; ym = []
    ybuf, ycur = {'hs': [], 'u10': [], 'date': []}, None
    def flush(y):
        np.savez_compressed(daily_file(tag, y), date=np.concatenate(ybuf['date']), hs=np.concatenate(ybuf['hs']), u10=np.concatenate(ybuf['u10']))
    for i, fw in enumerate(fws):
        yr_, mo_ = int(os.path.basename(fw)[4:8]), int(os.path.basename(fw)[8:10])
        fh = os.path.join(CFG['wave'][tag], os.path.basename(fw).replace('_wnd', '_hs'))
        r, nd, daily = wind_month(fw, fh)
        for k in keys: acc[k].append(r[k])
        ym.append(yr_ * 100 + mo_)
        if ycur is not None and yr_ != ycur: flush(ycur); ybuf = {'hs': [], 'u10': [], 'date': []}
        ycur = yr_
        ybuf['hs'].append(daily['hs']); ybuf['u10'].append(daily['u10'])
        ybuf['date'].append(np.array([yr_ * 10000 + mo_ * 100 + d for d in range(1, nd + 1)], np.int32))
        if (i + 1) % 60 == 0: print(f'  {i+1}/{len(fws)}', flush=True)
    flush(ycur)
    out = {k: np.stack(v) for k, v in acc.items()}; out['ym'] = np.array(ym)
    np.savez_compressed(fo, **out); print(f'  -> {os.path.basename(fo)} {out["nvalid_w"].shape} + {y1 - y0 + 1} daily files'); return np.load(fo)

WND = {tag: build_wind(tag) for tag in TAGS}

# ── Checks ──
for tag in TAGS:
    z = WND[tag]
    assert np.array_equal(z['ym'], WAV[tag]['ym']), f'{tag}: wind month series inconsistent'
    assert np.array_equal(z['nvalid_w'], WAV[tag]['nvalid']), f'{tag}: wind and wave valid days differ (ice mask should be identical)'
    assert np.array_equal(sum(z[k].astype(int) for k in BFT), z['nvalid_w'].astype(int)), f'{tag}: Beaufort bins do not sum to total'
    d = np.load(daily_file(tag, YEARS[tag][1])); assert d['hs'].shape[1] == NP_ and d['hs'].shape[0] >= 365, f'{tag}: daily file abnormal'
    print(f'{tag:7s} domain mean annual ≥Bft7 days {sum(z[k].astype(float) for k in U_STRONG).sum() / len(NAV[tag]["year"]) / NP_:.2f} | daily files OK')
print('Check passed: same wind/wave mask, bins sum to total, daily cache complete')

# %% Cell 6 — remove artifact grid cells, unify the valid grid cell set
Wh = WAV['his']['nvalid'].astype(np.float32).reshape(-1, 12, NP_).sum(1).mean(0)         # historical mean annual wave days
Nh = NAV['his']['OW_polaris_y'].astype(np.float32).mean(0)
ARTIFACT = (Wh >= 300) & (Nh <= 60)
EXCL  = FAR | ARTIFACT; VALID = ~EXCL
DOM_KM2 = CELL_KM2[VALID].sum()

def _mask_npz(z, keys2d):
    out = {k: np.array(z[k]) for k in z.files}
    for k in keys2d:
        if k in out and out[k].ndim == 2: out[k][:, EXCL] = 0
    return out
NAV = {t: _mask_npz(NAV[t], [k for k in NAV[t].files if k.endswith(('_m', '_y'))]) for t in TAGS}
WAV = {t: _mask_npz(WAV[t], ['nvalid'] + WMO) for t in TAGS}
WND = {t: _mask_npz(WND[t], BFT + list(COMP) + ['nvalid_w']) for t in TAGS}

def pass_sel(p):
    return (P_PASS == p) & VALID if p != 'Pan-Arctic' else (P_LAT >= 66.5) & VALID

# ── Checks ──
print(f'Artifact grid cells {ARTIFACT.sum()}, by sector: {pd.Series(np.where(P_SEC == "", "(outside)", P_SEC)[ARTIFACT]).value_counts().to_dict()}')
print(f'Valid grid cells {VALID.sum()} / {NP_}, valid area {DOM_KM2/1e6:.2f}×10⁶ km²')
for tag in TAGS:
    ex = np.clip(WAV[tag]['nvalid'].astype(int) - NAV[tag]['OW_polaris_m'].astype(int), 0, None)
    share = ex.sum() / WAV[tag]['nvalid'].sum()
    print(f'{tag:7s} after removal, cell-days with wave>navigable as share of wave cell-days {share:.2%} (monthly scale; ice-edge interpolation differences, expected 2–5%)')
    assert share < 0.05, f'{tag}: remaining discrepancy too large (>5%); first confirm trace-ice handling in decode is active, then check artifact criterion'
fig = plt.figure(figsize=(6.5, 6.5), dpi=110)
ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=90))
ax.set_extent([-180, 180, LAT0, 90], ccrs.PlateCarree()); ax.add_feature(cfeature.LAND, facecolor='#d9d3c7'); ax.add_feature(cfeature.COASTLINE, lw=.4)
ax.scatter(P_LON[ARTIFACT], P_LAT[ARTIFACT], s=5, c='#c8452b', transform=ccrs.PlateCarree(), label=f'artifact (n={ARTIFACT.sum()})')
ax.scatter(P_LON[FAR], P_LAT[FAR], s=5, c='#555', transform=ccrs.PlateCarree(), label=f'>100 km from ice grid (n={FAR.sum()})')
ax.legend(loc='lower left', fontsize=8); ax.set_title(f'Excluded cells, {MDL_LABEL}'); plt.show()

# %% Cell 7 — Section 1: passage definition and visualisation
OCEAN_C, LAND_C = '#dfe9f0', '#bfbfbf'
PROJ = ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=90)
def base_map(ax):
    ax.set_extent([-180, 180, LAT0, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_C, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_C, edgecolor='none', zorder=3)
    ax.add_feature(cfeature.COASTLINE, lw=.35, edgecolor='#666', zorder=4)
    ax.gridlines(lw=.3, color='#9aa0a6', alpha=.6, zorder=5, xlocs=range(-180, 180, 60), ylocs=[60, 70, 80])
    for lon, lab in [(0, '0°'), (90, '90°E'), (180, '180°'), (-90, '90°W')]:
        ax.text(lon, LAT0 + 1.2, lab, transform=ccrs.PlateCarree(), fontsize=7.5, ha='center', va='bottom', color='#444', zorder=9)
def draw_sectors(ax, labels=True):
    for s, (S, N, W, E) in SECTORS.items():
        if N < 90:
            for x in (W, E): ax.plot([x, x], [S, N], color='#222', lw=.9, ls=(0, (4, 2)), transform=ccrs.PlateCarree(), zorder=7)
    for latc in (66.5, 82.0):
        ax.plot(np.linspace(-180, 180, 361), np.full(361, latc), color='#222', lw=.9, ls=(0, (4, 2)), transform=ccrs.PlateCarree(), zorder=7)
    if labels:
        for p, (la, lo) in PASS_LABEL_POS.items():
            ax.text(lo, la, p.split(' (')[0], transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold', color=PASS_COLOR[p],
                    ha='center', va='center', zorder=9, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
def draw_routes(ax):
    for p, pts in WAYPOINTS.items():
        ax.plot([q[1] for q in pts], [q[0] for q in pts], color='#111', lw=1.1, transform=ccrs.Geodetic(), zorder=6)

fig = plt.figure(figsize=(9, 9), dpi=120); ax = plt.axes(projection=PROJ); base_map(ax)
for p, col in PASS_COLOR.items():
    s = pass_sel(p); ax.pcolormesh(LONW, LATP, to_grid(np.where(s, 1.0, np.nan)), cmap=ListedColormap([col]), vmin=0, vmax=1, alpha=.35,
                                   transform=ccrs.PlateCarree(), shading='auto', zorder=2)
draw_sectors(ax); draw_routes(ax)
ax.pcolormesh(LONW, LATP, to_grid(np.where(EXCL, 1.0, np.nan)), cmap=ListedColormap(['#222']), vmin=0, vmax=1, transform=ccrs.PlateCarree(), shading='auto', zorder=6)
ax.legend(handles=[mpatches.Patch(color=c, alpha=.5, label=p) for p, c in PASS_COLOR.items()] + [mpatches.Patch(color='#222', label='excluded cells')],
          loc='lower left', fontsize=8.5, framealpha=.9)
ax.set_title(f'Arctic passages (sector-based) and reference routes — {MDL_LABEL} grid', fontsize=12); plt.show()

# ── Checks ──
tab = pd.DataFrame([{'passage': p, 'cells': int(pass_sel(p).sum()), 'area_1e6km2': CELL_KM2[pass_sel(p)].sum() / 1e6} for p in SHOW + ['Pan-Arctic']])
print(tab.round(2).to_string(index=False))
assert sum(pass_sel(p).sum() for p in PASSAGES) + ((P_PASS == '') & VALID).sum() == VALID.sum()

# %% Cell 8 — Section 2: monthly navigable area share (heatmap) and navigation season length
def monthly_frac(tag, key, p):
    s = pass_sel(p)
    if key == 'wave_active': d = WAV[tag]['nvalid'].astype(np.float32)[:, s]; nd = WAV[tag]['ndays'][:, None]
    else: d = NAV[tag][key + '_m'].astype(np.float32)[:, s]; nd = NAV[tag]['ndays_m'][:, None]
    return 100.0 * ((d / nd) * CELL_KM2[s]).sum(1) / CELL_KM2[s].sum()

def period_month_table(key, ssp, p):
    out = {}
    for lab, tg, y0, y1 in PERIODS:
        tag = tg or ssp; ym = NAV[tag]['ym']; k = (ym // 100 >= y0) & (ym // 100 <= y1)
        f = monthly_frac(tag, key, p)[k]; out[lab] = pd.Series(f).groupby(ym[k] % 100).mean().values
    return pd.DataFrame(out, index=range(1, 13))

MON = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
HEATMAP_KEYS = ['OW_polaris', 'wave_active']          # main text: OW; PC7 only in tables
for key in HEATMAP_KEYS:
    fig, axes = plt.subplots(2, len(SHOW), figsize=(3.3 * len(SHOW), 7.2), dpi=110)
    for i, ssp in enumerate(SSPS):
        for j, p in enumerate(SHOW):
            ax = axes[i, j]; M = period_month_table(key, ssp, p)
            im = ax.imshow(M.values, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
            for a in range(12):
                for b in range(M.shape[1]):
                    ax.text(b, a, f'{M.values[a, b]:.0f}', ha='center', va='center', fontsize=7, color='white' if M.values[a, b] > 60 else '#222')
            ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, fontsize=7, rotation=35)
            ax.set_yticks(range(12)); ax.set_yticklabels(MON, fontsize=7)
            if i == 0: ax.set_title(p, fontsize=9)
            if j == 0: ax.set_ylabel(SSP_LABEL[ssp], fontsize=9)
    fig.colorbar(im, ax=axes, shrink=.6, pad=.01, label='% of passage ocean area')
    fig.suptitle(f'{LABEL[key]} — monthly area fraction, {MDL_LABEL}', y=.995, fontsize=12); plt.show()

rows = []
for ssp in SSPS:
    for lab, tg, y0, y1 in PERIODS:
        tag = tg or ssp; yrs = NAV[tag]['year']; k = (yrs >= y0) & (yrs <= y1)
        for p in SHOW + ['Pan-Arctic']:
            s = pass_sel(p); w = CELL_KM2[s]
            for ic in ['OW', 'PC7']:
                nd = NAV[tag][f'{ic}_polaris_y'][k].astype(np.float32).mean(0)[s]; mr = NAV[tag][f'{ic}_maxrun_y'][k].astype(np.float32).mean(0)[s]
                rows.append({'ssp': ssp, 'period': lab, 'passage': p, 'ice_class': ic,
                             'nav_days': (nd * w).sum() / w.sum(), 'max_run': (mr * w).sum() / w.sum(),
                             'area_share_ge90d': 100 * w[nd >= 90].sum() / w.sum(), 'area_share_ge183d': 100 * w[nd >= 183].sum() / w.sum()})
SEASON = pd.DataFrame(rows)
print('=== Navigation season length: area-weighted mean annual navigable days ===')
print(SEASON.pivot_table(index=['passage', 'ice_class'], columns=['ssp', 'period'], values='nav_days').round(0).to_string())
print('\n=== Longest continuous navigable period (area-weighted mean annual days) ===')
print(SEASON.pivot_table(index=['passage', 'ice_class'], columns=['ssp', 'period'], values='max_run').round(0).to_string())

# ── Checks ──
for key in ['OW_polaris', 'wave_active']:
    A = period_month_table(key, 'ssp126', 'Pan-Arctic'); B = period_month_table(key, 'ssp585', 'Pan-Arctic')
    assert (A.values >= 0).all() and (A.values <= 100.001).all() and np.allclose(A['1985–2014'], B['1985–2014'])
print('Check passed: shares in range, historical period identical across scenarios')

# %% Cell 8b — polar sector plot (angle = longitude sector, ring = month, colour = navigable area share)
def sec_sel(s): return (P_SEC == s) & VALID
def sector_month_table(key, ssp, s):
    out = {}
    for lab, tg, y0, y1 in PERIODS:
        tag = tg or ssp; ym = NAV[tag]['ym']; k = (ym // 100 >= y0) & (ym // 100 <= y1); c = sec_sel(s)
        if key == 'wave_active': d = WAV[tag]['nvalid'].astype(np.float32)[:, c]; nd = WAV[tag]['ndays'][:, None]
        else: d = NAV[tag][key + '_m'].astype(np.float32)[:, c]; nd = NAV[tag]['ndays_m'][:, None]
        f = 100.0 * ((d / nd) * CELL_KM2[c]).sum(1) / CELL_KM2[c].sum()
        out[lab] = pd.Series(f[k]).groupby(ym[k] % 100).mean().values
    return pd.DataFrame(out, index=range(1, 13))

R_IN, R_MID, R_OUT = 0.0, 0.42, 1.0
def polar_sector_panel(ax, key, ssp, lab, cmap='YlGnBu', norm=Normalize(0, 100)):
    ax.set_theta_zero_location('S'); ax.set_theta_direction(1)
    ax.set_ylim(0, R_OUT); ax.set_yticks([]); ax.set_xticks([]); ax.spines['polar'].set_visible(False)
    cmo = cm.get_cmap(cmap)
    for s, (S, N, W, E) in SECTORS.items():
        M = sector_month_table(key, ssp, s)[lab].values
        if N >= 90:
            dr = (R_MID - R_IN) / 12
            for m in range(12): ax.bar(0, dr, width=2 * np.pi, bottom=R_IN + m * dr, color=cmo(norm(M[m])), edgecolor='white', lw=.3)
        else:
            th0, th1 = np.deg2rad(W), np.deg2rad(E); dr = (R_OUT - R_MID) / 12
            for m in range(12): ax.bar((th0 + th1) / 2, dr, width=th1 - th0, bottom=R_MID + m * dr, color=cmo(norm(M[m])), edgecolor='white', lw=.3)
    for s, (S, N, W, E) in SECTORS.items():
        if N < 90:
            for x in (W, E): ax.plot([np.deg2rad(x)] * 2, [R_MID, R_OUT], color='#333', lw=.8)
    ax.plot(np.linspace(0, 2 * np.pi, 361), [R_MID] * 361, color='#333', lw=.8)
    for p, ss in PASSAGES.items():
        if 'TSR' in p: continue
        W = min(SECTORS[s][2] for s in ss); E = max(SECTORS[s][3] for s in ss)
        th = np.linspace(np.deg2rad(W), np.deg2rad(E), 60); ax.plot(th, [R_OUT + .06] * 60, color=PASS_COLOR[p], lw=3, clip_on=False)
    ax.set_title(lab, fontsize=9, pad=12)

KEY = 'OW_polaris'
fig, axes = plt.subplots(2, len(PERIODS), figsize=(3.2 * len(PERIODS), 7.4), dpi=120, subplot_kw={'projection': 'polar'})
for i, ssp in enumerate(SSPS):
    for j, (lab, tg, y0, y1) in enumerate(PERIODS):
        polar_sector_panel(axes[i, j], KEY, ssp, lab)
        if j == 0: axes[i, j].text(-0.25, .5, SSP_LABEL[ssp], transform=axes[i, j].transAxes, rotation=90, va='center', fontsize=11)
sm = cm.ScalarMappable(norm=Normalize(0, 100), cmap='YlGnBu'); sm.set_array([])
cb = fig.colorbar(sm, ax=axes, shrink=.55, pad=.02); cb.set_label('Navigable area fraction (%)')
fig.legend(handles=[mpatches.Patch(color=c, label=p) for p, c in PASS_COLOR.items()], loc='lower center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.45, -.01))
fig.suptitle(f'{LABEL[KEY]} — angle = longitude sector, rings = Jan (inner) → Dec (outer), {MDL_LABEL}', y=.995, fontsize=11)
plt.show()

# %% Cell 9 — Section 3: in-ice navigation share (wave days / OW navigable days; not in main text, goes to stackplot grey layer and SI)
def coverage_table(ssp, p):
    out = {}
    for lab, tg, y0, y1 in PERIODS:
        tag = tg or ssp; ym = NAV[tag]['ym']; k = (ym // 100 >= y0) & (ym // 100 <= y1); s = pass_sel(p)
        nd = NAV[tag]['ndays_m'][k].astype(np.float32)
        wv = (WAV[tag]['nvalid'].astype(np.float32)[k][:, s] * CELL_KM2[s]).sum(1)
        nv = (NAV[tag]['OW_polaris_m'].astype(np.float32)[k][:, s] * CELL_KM2[s]).sum(1)
        cap = nd * CELL_KM2[s].sum()
        df = pd.DataFrame({'m': ym[k] % 100, 'wv': wv, 'nv': nv, 'cap': cap}).groupby('m').sum()
        r = df.wv / df.nv.replace(0, np.nan); r[df.nv / df.cap < MIN_NAV_FRAC] = np.nan
        out[lab] = r.values
    return pd.DataFrame(out, index=range(1, 13)).clip(upper=1.0)

COV = {(ssp, p): coverage_table(ssp, p) for ssp in SSPS for p in SHOW}
vals = np.concatenate([v.values.ravel() for v in COV.values()]); vals = vals[np.isfinite(vals)]
print(f'Units with defined coverage {len(vals)}, undefined (navigable area <{MIN_NAV_FRAC:.0%}){sum(int(np.isnan(v.values).sum()) for v in COV.values())}')

fig, axes = plt.subplots(2, len(SHOW), figsize=(3.3 * len(SHOW), 7.2), dpi=110)
for i, ssp in enumerate(SSPS):
    for j, p in enumerate(SHOW):
        ax = axes[i, j]; M = COV[(ssp, p)]
        im = ax.imshow(M.values, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
        for a in range(12):
            for b in range(M.shape[1]):
                if np.isfinite(M.values[a, b]): ax.text(b, a, f'{M.values[a, b]:.2f}', ha='center', va='center', fontsize=6.5)
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, fontsize=7, rotation=35); ax.set_yticks(range(12)); ax.set_yticklabels(MON, fontsize=7)
        if i == 0: ax.set_title(p, fontsize=9)
        if j == 0: ax.set_ylabel(SSP_LABEL[ssp], fontsize=9)
fig.colorbar(im, ax=axes, shrink=.6, pad=.01, label='wave-active / OW-navigable cell-days')
fig.suptitle(f'Share of navigable days with wave simulation (rest = navigable in ice), {MDL_LABEL}', y=.995, fontsize=12); plt.show()

# %% Cell 10 — Section 4a: wind-wave composition of traditional vs newly navigable areas (stackplot + sea-state table)
def yearly(tag, arr_m):
    ym = NAV[tag]['ym']; return np.stack([arr_m[ym // 100 == y].astype(np.int32).sum(0) for y in NAV[tag]['year']])
def hist_mean(key_y): return NAV['his'][key_y].astype(np.float32).mean(0)
def fut_mean(ssp, key_y, y0, y1):
    yrs = NAV[ssp]['year']; return NAV[ssp][key_y][(yrs >= y0) & (yrs <= y1)].astype(np.float32).mean(0)
HIS_NAV = (hist_mean('OW_polaris_y') >= THR_HIS_NAV) & VALID
def new_nav_mask(ssp, y0=2081, y1=2100): return (fut_mean(ssp, 'OW_polaris_y', y0, y1) >= THR_NEW_FUT) & (hist_mean('OW_polaris_y') < THR_NEW_HIS) & VALID
GROUPS = {'newly navigable (by 2081–2100)': lambda ssp: new_nav_mask(ssp), 'historically navigable': lambda ssp: HIS_NAV}

HSY = {t: {k: yearly(t, WAV[t][k]) for k in WMO + ['nvalid']} for t in TAGS}
WDY = {t: {k: yearly(t, WND[t][k]) for k in BFT + list(COMP)} for t in TAGS}

def annual_series(ssp, sel):
    w = CELL_KM2[sel]; W = w.sum(); rec = []
    for tag in ['his', ssp]:
        for i, y in enumerate(NAV[tag]['year']):
            r = {'year': y}
            for k in WMO: r[k] = HSY[tag][k][i, sel] @ w / W
            for k in BFT + list(COMP): r[k] = WDY[tag][k][i, sel] @ w / W
            r['wave'] = HSY[tag]['nvalid'][i, sel] @ w / W
            r['nav'] = NAV[tag]['OW_polaris_y'][i, sel].astype(np.float32) @ w / W
            r['nowave'] = max(r['nav'] - r['wave'], 0.0); rec.append(r)
    return pd.DataFrame(rec).sort_values('year')

STK = WMO + ['nowave']; STK_LAB = WMO_LAB + ['navigable in ice (no wave simulation)']
STK_C = ['#c9e3f0', '#7fb8d8', '#3d86b5', '#e08a5b', '#a8322a', '#5e1914', '#dfe6ea']
for gname, gfun in GROUPS.items():
    fig, axes = plt.subplots(len(SHOW), 2, figsize=(12, 2.4 * len(SHOW)), sharex=True, sharey=True, dpi=110)
    for j, ssp in enumerate(SSPS):
        for i, p in enumerate(SHOW):
            ax = axes[i, j]; sel = gfun(ssp) & pass_sel(p)
            if sel.sum() < 20: ax.text(.5, .5, f'n={sel.sum()}', ha='center', transform=ax.transAxes); continue
            d = annual_series(ssp, sel); d[STK] = d[STK].rolling(11, center=True, min_periods=6).mean()
            ax.stackplot(d.year, *[d[k] for k in STK], colors=STK_C, labels=STK_LAB if (i == 0 and j == 0) else ())
            ax.axvline(2014.5, color='#444', lw=.8, ls=':'); ax.set_xlim(1985, 2100); ax.set_ylim(0, 366)
            ax.text(.02, .87, f'{p}  n={sel.sum()}  {CELL_KM2[sel].sum()/1e6:.2f}×10⁶ km²', transform=ax.transAxes, fontsize=8)
            if j == 0: ax.set_ylabel('days / yr', fontsize=8)
            if i == 0: ax.set_title(SSP_LABEL[ssp], fontsize=10)
    fig.legend(loc='upper center', ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, 1.0))
    fig.suptitle(f'{gname} — navigable window and sea-state composition (OW ships), {MDL_LABEL}', y=1.03, fontsize=11); plt.tight_layout(); plt.show()

rows = []
for ssp in SSPS:
    for lab, tg, y0, y1 in PERIODS:
        tag = tg or ssp; yrs = NAV[tag]['year']; k = (yrs >= y0) & (yrs <= y1)
        for gname, gfun in GROUPS.items():
            for p in SHOW + ['Pan-Arctic']:
                sel = gfun(ssp) & pass_sel(p)
                if sel.sum() < 20: continue
                w = CELL_KM2[sel]; W = w.sum(); r = {'ssp': ssp, 'period': lab, 'group': gname, 'passage': p, 'cells': int(sel.sum())}
                for kk in WMO + ['nvalid']: r[kk] = HSY[tag][kk][k][:, sel].mean(0) @ w / W
                for kk in BFT + list(COMP): r[kk] = WDY[tag][kk][k][:, sel].mean(0) @ w / W
                r['nav'] = NAV[tag]['OW_polaris_y'][k].astype(np.float32).mean(0)[sel] @ w / W
                N = r['nvalid']
                r['coverage'] = N / r['nav'] if r['nav'] > 0 else np.nan
                r['share_hs_ge2p5'] = 100 * sum(r[x] for x in HS_ROUGH) / N if N > 0 else np.nan
                r['share_u_ge13p9'] = 100 * sum(r[x] for x in U_STRONG) / N if N > 0 else np.nan
                r['share_u_ge17p2'] = 100 * sum(r[x] for x in U_GALE) / N if N > 0 else np.nan
                r['share_cp_2p5_13p9'] = 100 * r['cp_hs2p5_u13p9'] / N if N > 0 else np.nan
                rows.append(r)
SEA = pd.DataFrame(rows)
print('=== Navigable wave days: Hs≥2.5 m share of wave days (%) ===')
print(SEA.pivot_table(index=['group', 'passage'], columns=['ssp', 'period'], values='share_hs_ge2p5').round(1).to_string())
print('\n=== Navigable wave days: U10≥13.9 m/s share of wave days (%) ===')
print(SEA.pivot_table(index=['group', 'passage'], columns=['ssp', 'period'], values='share_u_ge13p9').round(1).to_string())
print('\n=== Navigable wave days: compound days (Hs≥2.5 & U10≥13.9) share of wave days (%) ===')
print(SEA.pivot_table(index=['group', 'passage'], columns=['ssp', 'period'], values='share_cp_2p5_13p9').round(1).to_string())

# ── Checks ──
for ssp in SSPS: assert not (new_nav_mask(ssp) & HIS_NAV).any(), 'Groups not mutually exclusive'
assert np.allclose(SEA[WMO].sum(1), SEA['nvalid']) and np.allclose(SEA[BFT].sum(1), SEA['nvalid']), 'Bins do not sum to total'
assert (SEA['coverage'].dropna() <= 1.05).all(); print('Check passed: groups exclusive, bins sum to total, coverage ≤ 1')

# %% Cell 10b — map of first-opening decade + opened area per passage (OW ships)
DECADES  = [(2021, 2030), (2031, 2040), (2041, 2050), (2051, 2060), (2061, 2070), (2071, 2080), (2081, 2090), (2091, 2100)]
DEC_LAB  = ['2020s', '2030s', '2040s', '2050s', '2060s', '2070s', '2080s', '2090s']
def first_open_decade(ssp):
    yrs = NAV[ssp]['year']
    dec = np.stack([NAV[ssp]['OW_polaris_y'][(yrs >= a) & (yrs <= b)].astype(np.float32).mean(0) >= THR_NEW_FUT for a, b in DECADES])
    persist = np.flip(np.cumprod(np.flip(dec, 0), 0), 0).astype(bool)
    idx = np.where(persist.any(0), persist.argmax(0), -1)
    out = np.where(hist_mean('OW_polaris_y') >= THR_HIS_NAV, 99, idx); out[~VALID] = -2
    return out
FOD = {ssp: first_open_decade(ssp) for ssp in SSPS}

ALREADY_C, NEVER_C = '#a9c4d6', '#f2f2f2'
cmap_dec = ListedColormap(['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026', '#7a0177', '#49006a'])
norm_dec = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap_dec.N)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), dpi=130, subplot_kw={'projection': PROJ})
for ax, ssp in zip(axes, SSPS):
    base_map(ax); f = FOD[ssp]
    ax.pcolormesh(LONW, LATP, to_grid(np.where(f == 99, 1.0, np.nan)), cmap=ListedColormap([ALREADY_C]), vmin=0, vmax=1, transform=ccrs.PlateCarree(), shading='auto', zorder=1)
    ax.pcolormesh(LONW, LATP, to_grid(np.where(f == -1, 1.0, np.nan)), cmap=ListedColormap([NEVER_C]), vmin=0, vmax=1, transform=ccrs.PlateCarree(), shading='auto', zorder=1)
    fd = f.astype(np.float32); fd[(fd < 0) | (fd == 99)] = np.nan
    pm = ax.pcolormesh(LONW, LATP, to_grid(fd), cmap=cmap_dec, norm=norm_dec, transform=ccrs.PlateCarree(), shading='auto', zorder=2)
    draw_sectors(ax); draw_routes(ax); ax.set_title(f'{MDL_LABEL} · {SSP_LABEL[ssp]}', fontsize=11)
cb = fig.colorbar(pm, ax=axes, ticks=range(8), shrink=.7, pad=.02); cb.ax.set_yticklabels(DEC_LAB)
cb.set_label(f'First-open decade (OW ships, ≥{THR_NEW_FUT} navigable d/yr, persistent)')
fig.legend(handles=[mpatches.Patch(color=ALREADY_C, label=f'Historically open (1985–2014, ≥{THR_HIS_NAV} d/yr)'),
                    mpatches.Patch(facecolor=NEVER_C, edgecolor='#bbb', label='Not open by 2100'),
                    plt.Line2D([], [], color='#111', lw=1.1, label='Reference routes'),
                    plt.Line2D([], [], color='#222', lw=.9, ls=(0, (4, 2)), label='Passage boundaries')],
           loc='lower center', ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(.45, -.02))
plt.show()

rows = []
for ssp in SSPS:
    for p in SHOW:
        s = pass_sel(p); f = FOD[ssp]
        rec = {'ssp': ssp, 'passage': p, 'already_open': CELL_KM2[s & (f == 99)].sum() / 1e6,
               'never_open': CELL_KM2[s & (f == -1)].sum() / 1e6, 'total': CELL_KM2[s].sum() / 1e6}
        for d in range(8): rec[DEC_LAB[d]] = CELL_KM2[s & (f == d)].sum() / 1e6
        rows.append(rec)
FODT = pd.DataFrame(rows)
print('=== Area by first-opening decade per passage (10⁶ km²) ===')
print(FODT.set_index(['passage', 'ssp'])[DEC_LAB + ['already_open', 'never_open', 'total']].round(2).to_string())

# ── Checks ──
for ssp in SSPS:
    f = FOD[ssp]; assert ((f == 99) | (f == -1) | ((f >= 0) & (f <= 7))).sum() == VALID.sum(), f'{ssp}: classification does not cover all valid grid cells'
print('Check passed: historically open / first-opening decade / never open are exclusive and complete')

# %% Cell 11 — Section 4b: window vs sea-state effect decomposition (traditional route own change; new area own change; new area vs traditional route)
def _agg(tag, cells, y0, y1, ekeys):
    yrs = NAV[tag]['year']; k = (yrs >= y0) & (yrs <= y1); w = CELL_KM2[cells]; W = w.sum()
    N = HSY[tag]['nvalid'][k][:, cells].mean(0) @ w / W
    E = sum((HSY if ek.startswith('hs') else WDY)[tag][ek][k][:, cells].mean(0) @ w / W for ek in ekeys)
    return N, E
def decomp(ssp, p, ekeys, group, ref):
    """group: 'his'|'new'; ref: ('his',1985,2014) traditional route historical period, or (ssp,y0,y1) own early period"""
    sel = (new_nav_mask(ssp) if group == 'new' else HIS_NAV) & pass_sel(p)
    ref_tag, r0, r1 = ref; ref_cells = (HIS_NAV & pass_sel(p)) if ref_tag == 'his' else sel
    if sel.sum() < 20 or ref_cells.sum() < 20: return None
    Nr, Er = _agg(ref_tag, ref_cells, r0, r1, ekeys); Nf, Ef = _agg(ssp, sel, 2081, 2100, ekeys)
    if Nr < MIN_WAVE_DAYS or Nf < MIN_WAVE_DAYS: return None
    pr, pf = Er / Nr, Ef / Nf; dN, dp = Nf - Nr, pf - pr
    return dict(ssp=ssp, passage=p, group=group, ref=f'{ref_tag} {r0}–{r1}', N_ref=Nr, E_ref=Er, N_fut=Nf, E_fut=Ef,
                window=dN * pr, roughness=Nr * dp, interaction=dN * dp, dE=Ef - Er)

HAZ = [('Hs≥2.5 m', HS_ROUGH), ('U10≥13.9 m/s', U_STRONG)]
rows = []
for ssp in SSPS:
    for p in SHOW:
        for hz, ek in HAZ:
            for r in [decomp(ssp, p, ek, 'his', ('his', 1985, 2014)),          # traditional route: own change
                      decomp(ssp, p, ek, 'new', (ssp, 2041, 2060)),           # new area: own change
                      decomp(ssp, p, ek, 'new', ('his', 1985, 2014))]:        # new area vs traditional route historical (contrast, not change)
                if r: r['hazard'] = hz; rows.append(r)
DEC = pd.DataFrame(rows)
DEC['kind'] = np.where(DEC.group == 'his', 'traditional: change vs 1985–2014',
               np.where(DEC.ref.str.startswith('his'), 'new vs traditional (comparison)', 'new: change 2041–2060 → 2081–2100'))
print('=== Decomposition (days/yr) ===')
print(DEC[['kind', 'ssp', 'hazard', 'passage', 'N_ref', 'E_ref', 'N_fut', 'E_fut', 'window', 'roughness', 'interaction', 'dE']].round(1).to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), dpi=110)
for i, kind in enumerate(['traditional: change vs 1985–2014', 'new: change 2041–2060 → 2081–2100']):
    for j, (hz, _) in enumerate(HAZ):
        ax = axes[i, j]; d = DEC[(DEC.kind == kind) & (DEC.hazard == hz)]; x = np.arange(len(d))
        ax.bar(x, d.window, .6, color='#2f7ab4', label='window effect ΔN·p̄'); ax.bar(x, d.roughness, .6, bottom=d.window, color='#e08a5b', label='roughness effect N̄·Δp')
        ax.bar(x, d.interaction, .6, bottom=d.window + d.roughness, color='#9aa0a6', label='interaction'); ax.plot(x, d.dE, 'k_', ms=18, mew=2, label='total ΔE')
        ax.set_xticks(x); ax.set_xticklabels([f'{r.passage.split(" (")[0]}\n{SSP_LABEL[r.ssp]}' for r in d.itertuples()], fontsize=7.5)
        ax.axhline(0, color='#333', lw=1); ax.set_title(f'{kind} — {hz}', fontsize=9.5); ax.grid(axis='y', alpha=.25)
        if j == 0: ax.set_ylabel('Δ days per year')
axes[0, 0].legend(fontsize=8, frameon=False); plt.tight_layout(); plt.show()

# ── Checks ──
assert np.allclose(DEC.window + DEC.roughness + DEC.interaction, DEC.dE, atol=1e-6); print('Check passed: decomposition sums to total')

# %% Cell 12 — Section 5: M2 hazard-weighted exposure index (monthly cumulative + per wave day; M2 CI bounds; sample threshold)
def hazard_index(tag, cells, y0, y1, which='main'):
    ym = WAV[tag]['ym']; k = (ym // 100 >= y0) & (ym // 100 <= y1)
    nv = WAV[tag]['nvalid'][k][:, cells].astype(np.float32); nyr = len(np.unique(ym[k] // 100))
    if (nv.sum(0) @ CELL_KM2[cells]) / CELL_KM2[cells].sum() / nyr < MIN_WAVE_DAYS: return np.nan, np.nan
    logm = sum(BETA[which][b] * WAV[tag][b][k][:, cells].astype(np.float32) for b in WMO)
    w = nv * CELL_KM2[cells]
    monthly = float((w * np.exp(logm)).sum() / w.sum())
    per_day = float(np.exp((logm * CELL_KM2[cells]).sum() / w.sum()))
    return monthly, per_day

rows = []
for ssp in SSPS:
    for lab, tg, y0, y1 in PERIODS:
        for gname, gfun in GROUPS.items():
            for p in SHOW + ['Pan-Arctic']:
                sel = gfun(ssp) & pass_sel(p)
                if sel.sum() < 20: continue
                r = {'ssp': ssp, 'period': lab, 'group': gname, 'passage': p}
                for which in ['main', 'lo', 'hi']: r[which], r[which + '_perday'] = hazard_index(tg or ssp, sel, y0, y1, which)
                rows.append(r)
HZ = pd.DataFrame(rows)
HZ['excess_month_%'] = (HZ['main'] - 1) * 100; HZ['excess_day_%'] = (HZ['main_perday'] - 1) * 100
print('=== Monthly cumulative index: monthly accident rate ratio vs an all-calm month (M2 Table 1 coefficients) ===')
print(HZ.pivot_table(index=['group', 'passage'], columns=['ssp', 'period'], values='main').round(3).to_string())
print('\n=== Excess accident rate per wave day (%) ===')
print(HZ.pivot_table(index=['group', 'passage'], columns=['ssp', 'period'], values='excess_day_%').round(2).to_string())
print('\n=== Monthly cumulative index, M2 95% CI upper bound ===')
print(HZ.pivot_table(index=['group', 'passage'], columns=['ssp', 'period'], values='hi').round(3).to_string())

# ── Checks ──
ok_ = HZ.dropna(subset=['main'])
assert (ok_['main'] >= 1 - 1e-9).all() and (ok_['lo'] <= ok_['main'] + 1e-9).all() and (ok_['main'] <= ok_['hi'] + 1e-9).all()
print('Check passed: index ≥ 1, interval contains point estimate')

# %% Cell 12b — three exposure layers in new areas: daily hazard / annual days per location / regional total exposure (incl. weighted)
def layers(ssp, p, y0, y1):
    sel = new_nav_mask(ssp) & pass_sel(p)
    if sel.sum() < 20: return None
    tag = ssp; yrs = NAV[tag]['year']; k = (yrs >= y0) & (yrs <= y1); w = CELL_KM2[sel]; W = w.sum()
    r = {'ssp': ssp, 'passage': p, 'period': f'{y0}–{y1}', 'cells': int(sel.sum()), 'area_1e6km2': W / 1e6}
    days = {kk: HSY[tag][kk][k][:, sel].mean(0) for kk in WMO + ['nvalid']}
    days.update({kk: WDY[tag][kk][k][:, sel].mean(0) for kk in BFT + list(COMP)})
    nav = NAV[tag]['OW_polaris_y'][k].astype(np.float32).mean(0)[sel]
    N = days['nvalid'] @ w / W
    for kk in WMO + BFT + list(COMP):
        r[f'days_{kk}'] = days[kk] @ w / W
        r[f'share_{kk}'] = 100 * (days[kk] @ w) / (days['nvalid'] @ w) if N > 0 else np.nan
        r[f'tot_{kk}'] = days[kk] @ w / 1e6
    r['nav_days'] = nav @ w / W; r['wave_days'] = N; r['coverage'] = N / r['nav_days'] if r['nav_days'] > 0 else np.nan
    ym = WAV[tag]['ym']; km = (ym // 100 >= y0) & (ym // 100 <= y1); nyr = len(np.unique(ym[km] // 100))
    for which in ['main', 'lo', 'hi']:
        if N < MIN_WAVE_DAYS: r[f'idx_month_{which}'] = r[f'idx_day_{which}'] = r[f'wexp_{which}'] = np.nan; continue
        logm = sum(BETA[which][b] * WAV[tag][b][km][:, sel].astype(np.float32) for b in WMO)
        nv = WAV[tag]['nvalid'][km][:, sel].astype(np.float32); ww = nv * w
        r[f'idx_month_{which}'] = float((ww * np.exp(logm)).sum() / ww.sum())
        r[f'idx_day_{which}']   = float(np.exp((logm * w).sum() / ww.sum()))
        r[f'wexp_{which}'] = float(((np.exp(logm) - 1) * ww).sum() / 1e6 / nyr)
    return r

NEWH = pd.DataFrame([x for ssp in SSPS for (lab, tg, y0, y1) in PERIODS[1:] for p in SHOW + ['Pan-Arctic'] if (x := layers(ssp, p, y0, y1))])
NEWH['excess_day_%'] = (NEWH['idx_day_main'] - 1) * 100
for col, ttl in [('share_hs_2p5_4', 'Layer 1: Hs 2.5–4 m share of wave days (%)'), ('days_hs_2p5_4', 'Layer 2: Hs 2.5–4 m mean annual days'),
                 ('tot_hs_2p5_4', 'Layer 3: Hs 2.5–4 m total exposure (10⁶ km²·d/yr)'), ('excess_day_%', 'Excess accident rate per wave day (%)'),
                 ('wexp_main', 'Weighted total exposure (10⁶ km²·d/yr)')]:
    print(f'\n=== New areas · {ttl} ===')
    print(NEWH.pivot_table(index='passage', columns=['ssp', 'period'], values=col).round(2).to_string())

# %% Cell 13 — export
OUT = {'season': SEASON, 'seastate': SEA, 'decomp': DEC, 'hazard_index': HZ, 'newarea_layers': NEWH, 'first_open_decade': FODT}
for nm, df in OUT.items():
    fo = os.path.join(M5_OUT, f'M5_{nm}_{MODEL}.csv'); df.to_csv(fo, index=False); print(f'{fo}  {len(df)} rows')
def _slug(p): return p.split(' (')[0].replace(' ', '_')
for ssp in SSPS:
    for p in SHOW:
        COV[(ssp, p)].to_csv(os.path.join(M5_OUT, f'M5_coverage_{MODEL}_{ssp}_{_slug(p)}.csv'))
        for key in ['OW_polaris', 'PC7_polaris', 'wave_active']:
            period_month_table(key, ssp, p).to_csv(os.path.join(M5_OUT, f'M5_monthfrac_{MODEL}_{ssp}_{key}_{_slug(p)}.csv'))
np.savez_compressed(os.path.join(M5_OUT, f'M5_masks_{MODEL}.npz'), VALID=VALID, EXCL=EXCL, ARTIFACT=ARTIFACT, HIS_NAV=HIS_NAV,
                    NEW_126=new_nav_mask('ssp126'), NEW_585=new_nav_mask('ssp585'), FOD_126=FOD['ssp126'], FOD_585=FOD['ssp585'],
                    P_LAT=P_LAT, P_LON=P_LON, P_PASS=P_PASS.astype(str))
assert all(os.path.getsize(os.path.join(M5_OUT, f'M5_{nm}_{MODEL}.csv')) > 0 for nm in OUT); print('Export complete')
