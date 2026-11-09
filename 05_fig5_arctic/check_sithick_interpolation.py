# -*- coding: utf-8 -*-
"""
check_sithick_interpolation.py — tests how "linear interpolation of monthly mean ice thickness to daily" affects POLARIS navigability

Source: ad-hoc script run in the 2026-09-06 analysis session (results recorded in M5_design.md §3), restored verbatim with comments.
On the ACCESS-CM2 historical period (1985–2014, 27,375 target grid cells × 10,957 days), with the same daily siconc,
computes POLARIS RIO ≥ 0 navigable cell-days using (a) SIday daily sithick and (b) SImon monthly mean sithick interpolated linearly to daily,
and compares totals, per-cell-day agreement and per-cell annual navigable-day error. EC-Earth3 has no future daily thickness; the main text uses (b).

Printed output (2026-09-06):
  thickness |Δ| median 0.048 m, p90 0.261 m (ice-covered cell-days only)
  NOICE: 59.57M vs 62.40M (+4.76%) | agreement 0.9879 | annual-day error median 0.1 d, p95 1.9 d
  PC7  : 79.99M vs 81.61M (+2.02%) | agreement 0.9810 | median 0.7 d, p95 5.9 d
  PC6  : 88.22M vs 90.00M (+2.02%) | agreement 0.9793 | median 0.8 d, p95 5.2 d
Requires: siday_ACCESS_his.npz (produced by M5_pipeline.py Cell 2), sithick_ACCESS_historical.nc (SImon).
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
import numpy as np, pandas as pd, xarray as xr, glob, warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree

# ── Target grid cells and nearest neighbours (same as M5_pipeline.py Cell 1) ──
f0 = sorted(glob.glob(f'{DATA_ROOT}/new_climate/wave/ACCESS/CDFAC_his/ww3.*_hs.nc'))[0]
d = xr.open_dataset(f0); LAT, LON = d.latitude.values, d.longitude.values; MAP = d.MAPSTA.values; d.close()
lon180 = ((LON + 180) % 360) - 180; LA, LO = np.meshgrid(LAT, lon180, indexing='ij')
FLAT = np.flatnonzero((np.isin(MAP, [1, 7]) & (LA >= 60)).ravel()); P_LAT = LA.ravel()[FLAT]; P_LON = LO.ravel()[FLAT]
def xyz(la, lo):
    la, lo = np.deg2rad(la), np.deg2rad(lo); return np.stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)], -1)

# ── Daily cache (siconc and daily sithick) ──
z = np.load(f'{DATA_ROOT}/new_climate/siconc/_nav/siday_ACCESS_his.npz')
sic = z['sic'].astype(np.float32); sic[sic == 65535] = np.nan; sic /= 10000
sitd = z['sit'].astype(np.float32); sitd[sitd == 65535] = np.nan; sitd /= 100
sitd[np.isnan(sitd) & np.isfinite(sic) & (sic <= 0.001)] = 0                   # trace ice → thickness 0
dates = pd.to_datetime(z['date'].astype(str)); yr = dates.year.values

# ── Monthly thickness (SImon) → nearest neighbour to target cells → linear interpolation to daily ──
m = xr.open_dataset(f'{DATA_ROOT}/new_climate/siconc/sithick_ACCESS_historical.nc')
slat = m.latitude.values.ravel(); slon = ((m.longitude.values.ravel() + 180) % 360) - 180
V = m.sithick.values.reshape(m.sizes['time'], -1); good = np.isfinite(slat) & np.isfinite(slon)
dist, nn = cKDTree(xyz(slat[good], slon[good])).query(xyz(P_LAT, P_LON), k=1)
Vm = V[:, good][:, nn]
t_m = pd.to_datetime([f'{int(str(t)[:4])}-{int(str(t)[5:7]):02d}-15' for t in m.time.values])
Vm = np.where(np.isnan(Vm), 0.0, Vm)                                             # ice-free month → 0, then interpolate
xm = (t_m - pd.Timestamp('1985-01-01')).days.values.astype(float); xd = (dates - pd.Timestamp('1985-01-01')).days.values.astype(float)
sitm = np.empty_like(sitd)
for j in range(0, sitd.shape[1], 2000):
    sitm[:, j:j+2000] = np.stack([np.interp(xd, xm, Vm[:, c]) for c in range(j, min(j+2000, sitd.shape[1]))], 1)
sitm[np.isnan(sic)] = np.nan

# ── POLARIS (IMO MSC.1/Circ.1519 Table 1.3, single ice type, thickness proxies ice age) ──
SIT_EDGES = [0.10, 0.15, 0.30, 0.50, 0.70, 1.00, 1.20, 2.00, 2.50, 3.00]
RIV = {'NOICE': [1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -8], 'PC7': [2, 2, 2, 1, 1, 0, -1, -2, -3, -3, -3], 'PC6': [2, 2, 2, 2, 1, 1, 0, -1, -2, -3, -3]}
def rio(s, t, ic):
    tab = np.array(RIV[ic], np.float32); r = np.where(t <= 0, 3.0, tab[np.clip(np.digitize(t, SIT_EDGES), 0, 10)])
    C = s * 10; return (10 - C) * 3 + C * r

print(f'ACCESS-CM2 historical 1985–2014, {sitd.shape[1]} cells × {sitd.shape[0]} days')
print(f'thickness |Δ| daily vs interpolated: median {np.nanmedian(np.abs(sitd-sitm)):.3f} m, p90 {np.nanpercentile(np.abs(sitd-sitm), 90):.3f} m (ice-covered cell-days)')
rows = []
for ic in ['NOICE', 'PC7', 'PC6']:
    nd = (rio(sic, sitd, ic) >= 0) & np.isfinite(sic); nm = (rio(sic, sitm, ic) >= 0) & np.isfinite(sic)
    tot_d = nd.sum(); tot_m = nm.sum(); agree = (nd == nm)[np.isfinite(sic)].mean()
    yd = np.stack([nd[yr == y].sum(0) for y in np.unique(yr)]).mean(0); ym_ = np.stack([nm[yr == y].sum(0) for y in np.unique(yr)]).mean(0)
    rows.append(dict(vessel_class=ic, navdays_daily_M=tot_d/1e6, navdays_interp_M=tot_m/1e6, rel_diff_pct=(tot_m-tot_d)/tot_d*100,
                     cell_day_agreement=agree, annual_err_median_d=np.median(np.abs(ym_-yd)), annual_err_p95_d=np.percentile(np.abs(ym_-yd), 95)))
    print(f'{ic:6s}: navigable cell-days daily {tot_d/1e6:.2f}M  interpolated {tot_m/1e6:.2f}M  ({(tot_m-tot_d)/tot_d:+.2%}) | agreement {agree:.4f} | '
          f'annual-days error median {np.median(np.abs(ym_-yd)):.1f} d, p95 {np.percentile(np.abs(ym_-yd), 95):.1f} d')
pd.DataFrame(rows).to_csv(f'{DATA_ROOT}/new_climate/_M5/M5_sithick_interpolation_check_ACCESS.csv', index=False)
