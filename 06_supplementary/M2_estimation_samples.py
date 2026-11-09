# -*- coding: utf-8 -*-
"""
M2_estimation_samples.py — SI table: actual estimation samples of the occurrence models
Refits with the specification of 02_fig2_weather_and_occurrence/M2_update_v3.ipynb (reads caches only, does not modify data),
and reads observations, accidents, clusters, fixed effects and separation exclusions from the pyfixest model objects.
Output: SI_occurrence_estimation_samples.csv
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
import os, numpy as np, pandas as pd, pyfixest as pf, warnings; warnings.filterwarnings('ignore')
import pyarrow.parquet as pq

CACHE   = rf'{DATA_ROOT}/AIS_processing/panel_cache_M2/grid2'
OUT     = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SI_occurrence_estimation_samples.csv')]
Y_MAIN  = 'A_it'
FE_ABS  = 'grid_id^month + region^year_month'
FE_REL_WI = ['grid_id + year_month', 'grid_id^month + year_month', 'grid_id']
TYPES_ALL = ['Machinery','Grounding','Collision','Contact','Fire_Explosion','Foundered','Hull_Damage','Piracy_Other','Misc']
MIN_EVENTS = 60
WAVE_CUTS = [1.25, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
WIND_CUTS = [8.0, 10.0, 12.0, 13.9, 15.0, 17.2, 19.0, 20.8, 22.0]
ABS_LADDER = [(1.25, 8.0), (2.5, 10.8), (4.0, 13.9), (6.0, 17.2), (9.0, 20.8)]
ABS_GRID   = ABS_LADDER + [(4.0, 17.2), (6.0, 13.9), (6.0, 20.8)]
ABS_D      = [(4.0, 13.9), (6.0, 17.2)]
def cutname(pre, c): return f'{pre}_ge{c:g}'.replace('.', 'p')
def abstag(hs, u): return f'{hs:g}_{u:g}'.replace('.', 'p')
def abs3(hs, u):  t = abstag(hs, u); return [f'wo_{t}', f'wv_{t}', f'cp_{t}']

# ── panel + cached sweep/compound variables (merged the same way as notebook S2, S3) ──
panel = pd.read_parquet(rf'{CACHE}\panel_global.parquet')
n_panel_rows = len(panel); n_panel_cells = panel.grid_id.nunique()
for fn in ['_scan_days.parquet', '_abs_compound_days.parquet']:
    S = pd.read_parquet(rf'{CACHE}\{fn}'); cols = [c for c in S.columns if c not in ('grid_id', 'year_month')]
    panel = panel.drop(columns=[c for c in cols if c in panel.columns]).merge(S, on=['grid_id', 'year_month'], how='left')
    panel[cols] = panel[cols].fillna(0).astype('int16')
PA = panel[panel.groupby('grid_id')[Y_MAIN].transform('sum') >= 1].copy()          # active(): manually drop grid cells with zero accidents over the whole period
WV_REG = [c for c in panel.columns if c.startswith('Wv_')][1:]
WD_REG = [c for c in panel.columns if c.startswith('Wd_')][1:]
print(f'panel {n_panel_rows:,} rows / {n_panel_cells} cells -> active {len(PA):,} rows / {PA.grid_id.nunique()} cells / {int(PA[Y_MAIN].sum()):,} accidents')

def fitm(y, xs, data, fe):
    return pf.fepois(f"{y} ~ {' + '.join(xs)} | {fe}", data=data, vcov={'CRV1': 'grid_id'}, offset='log_E')
def fit_ladder(y, xs, data, ladder):
    for fe in ladder:
        try:
            m = fitm(y, xs, data, fe)
            if all(x in set(m.coef().index) for x in xs): return m, fe
        except Exception: continue
    return None, None

def record(model_group, model, y, thr, m, fe, data_in, source):
    d = m._data
    n_sep = getattr(m, 'n_separation_na', np.nan)
    row = dict(Model=model_group, Model_detail=model, Outcome=y, Threshold_category=thr,
               Input_rows=len(data_in), Input_clusters=data_in.grid_id.nunique(), Input_accidents=int(data_in[y].sum()),
               Observations=int(m._N), Accident_records=int(d[y].sum()), Clusters=int(d['grid_id'].nunique()),
               Fixed_effects=m._fixef, Separation_exclusions=int(n_sep) if pd.notna(n_sep) else 'not reported by estimator',
               Separation_check='pyfixest separation_check=["fe"] (default): drops observations in fixed-effect groups with all-zero outcome',
               Manual_exclusion=f'grid cells with zero {Y_MAIN} over 2011–2021 removed before estimation ({n_panel_rows - len(PA):,} of {n_panel_rows:,} cell-months)',
               Ladder_fallback=('yes' if fe != FE_ABS else 'no'), Source=source)
    assert row['Input_rows'] - row['Observations'] == row['Separation_exclusions'], 'separation count mismatch'
    return row

rows = []
# 1–2 Hs / wind speed category models (Table 1, Fig. 2b left)
for cols, name, thr in [(WV_REG, 'Wave-height category model', 'WMO classes (ref. <1.25 m)'), (WD_REG, 'Wind-speed category model', 'Beaufort classes (ref. <8 m/s)')]:
    m = fitm(Y_MAIN, cols, PA, FE_ABS)
    rows.append(record(name, name, Y_MAIN, thr, m, FE_ABS, PA, 'M2_update_v3 Cell 6 (Table 1); refit, model object'))
    print('done', name, m._N)
# 3 single-threshold models (threshold sweep, Fig. 2b right)
for pre, cuts, unit in [('wv', WAVE_CUTS, 'm'), ('wd', WIND_CUTS, 'm/s')]:
    for c in cuts:
        col = cutname(pre, c)
        if col not in PA.columns: print('missing', col); continue
        m = fitm(Y_MAIN, [col], PA, FE_ABS)
        rows.append(record('Single-threshold model', f'{"Hs" if pre == "wv" else "U10"} ≥ {c:g} {unit}', Y_MAIN, f'{"Hs" if pre == "wv" else "U10"} ≥ {c:g} {unit}', m, FE_ABS, PA, 'M2_update_v3 Cell 7 (threshold sweep); refit, model object'))
    print('done scan', pre)
# 4 absolute-threshold three-state compound models (Fig. 2c and appendix table A1)
for hs, u in ABS_GRID:
    xs = abs3(hs, u); m = fitm(Y_MAIN, xs, PA, FE_ABS)
    rows.append(record('Compound three-state model', f'Hs ≥ {hs:g} m & U10 ≥ {u:g} m/s', Y_MAIN, f'wind-only / wave-only / compound days at Hs ≥ {hs:g} m, U10 ≥ {u:g} m/s', m, FE_ABS, PA,
                       'M2_update_v3 Cell 8 (Table 2, kind=compound); refit, model object' + ('' if (hs, u) in ABS_LADDER else ' [appendix table A1 only]')))
print('done compound')
# 5 Fig. 2d accident-type models (three-state compound, FE ladder)
for hs, u in ABS_D:
    xs = abs3(hs, u)
    for t in TYPES_ALL:
        col = f'A_{t}'; n = int(PA[col].sum())
        if n < MIN_EVENTS:
            rows.append(dict(Model='Accident-type compound model', Model_detail=f'{t}, Hs ≥ {hs:g} m & U10 ≥ {u:g} m/s', Outcome=col, Threshold_category=f'Hs ≥ {hs:g} m, U10 ≥ {u:g} m/s',
                             Input_rows=len(PA), Input_clusters=PA.grid_id.nunique(), Input_accidents=n, Observations='not estimated (< 60 events)', Accident_records=n,
                             Clusters='', Fixed_effects='', Separation_exclusions='', Separation_check='', Manual_exclusion='', Ladder_fallback='', Source='M2_update_v3 Cell 9 (Table 4)'))
            continue
        m, fe = fit_ladder(col, xs, PA, [FE_ABS] + FE_REL_WI)
        if m is None:
            rows.append(dict(Model='Accident-type compound model', Model_detail=f'{t}, Hs ≥ {hs:g} m & U10 ≥ {u:g} m/s', Outcome=col, Threshold_category=f'Hs ≥ {hs:g} m, U10 ≥ {u:g} m/s',
                             Input_rows=len(PA), Input_clusters=PA.grid_id.nunique(), Input_accidents=n, Observations='did not converge', Accident_records=n,
                             Clusters='', Fixed_effects='', Separation_exclusions='', Separation_check='', Manual_exclusion='', Ladder_fallback='', Source='M2_update_v3 Cell 9 (Table 4)'))
            continue
        rows.append(record('Accident-type compound model', f'{t}, Hs ≥ {hs:g} m & U10 ≥ {u:g} m/s', col, f'wind-only / wave-only / compound days at Hs ≥ {hs:g} m, U10 ≥ {u:g} m/s', m, fe, PA,
                           'M2_update_v3 Cell 9 (Table 4, spec=three, fit_ladder); refit, model object'))
    print('done types', hs, u)

T = pd.DataFrame(rows)
for p in OUT:
    os.makedirs(os.path.dirname(p), exist_ok=True); T.to_csv(p, index=False, encoding='utf-8-sig')
print('\nwritten:', *OUT, sep='\n  ')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(T[['Model', 'Model_detail', 'Outcome', 'Input_rows', 'Observations', 'Accident_records', 'Clusters', 'Separation_exclusions', 'Fixed_effects']].to_string(index=False))
