import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
# grid1 vs grid2: Table 1 binned dose-response + absolute-threshold compound ladder, both grids side by side in one script
import os, sys, time, numpy as np, pandas as pd, pyfixest as pf, warnings; warnings.filterwarnings('ignore')
from scipy.stats import norm
pd.set_option('display.width', 250)
T0 = time.time()
def say(*a): print(f'[{(time.time()-T0)/60:6.1f} min]', *a, flush=True)
BASE = rf'{DATA_ROOT}/AIS_processing/panel_cache_M2'
FE_ABS = 'grid_id^month + region^year_month'
ABS_LADDER = [(1.25, 8.0), (2.5, 10.8), (4.0, 13.9), (6.0, 17.2), (9.0, 20.8)]
def abstag(hs, u): return f'{hs:g}_{u:g}'.replace('.', 'p')
def abs3(hs, u):  t = abstag(hs, u); return [f'wo_{t}', f'wv_{t}', f'cp_{t}']
def stars(p): return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '†' if p<.1 else ''
def active(p, col='A_it'): return p[p.groupby('grid_id')[col].transform('sum') >= 1].copy()
def fitm(y, xs, data, fe, cluster):
    return pf.fepois(f"{y} ~ {' + '.join(xs)} | {fe}", data=data, vcov={'CRV1': cluster}, offset='log_E')
def nobs(m):
    for a in ("_N", "N", "nobs"):
        v = getattr(m, a, None)
        if v is not None: return int(v)
    return -1
def row(m, x):
    b, se = float(m.coef()[x]), float(m.se()[x]); return np.exp(b), np.exp(b-1.96*se), np.exp(b+1.96*se), float(m.pvalue()[x])

def load(grid):
    CACHE = rf'{BASE}\grid{grid}'
    panel = pd.read_parquet(rf'{CACHE}\panel_global.parquet')
    if grid == 1:                                        # cluster on 2° blocks
        lat = panel['grid_id'].str.split('_').str[0].astype(float); lon = panel['grid_id'].str.split('_').str[1].astype(float)
        panel['block2'] = (np.floor(lat/2)*2+1).map('{:.3f}'.format) + '_' + (np.floor(lon/2)*2+1).map('{:.3f}'.format)
        cluster = 'block2'
    else:
        cluster = 'grid_id'
    fr = []
    for r in panel['region'].unique():
        w = pd.read_parquet(rf'{CACHE}\{r}__dmax_wave.parquet').rename(columns={'dmax':'swh'})
        i = pd.read_parquet(rf'{CACHE}\{r}__dmax_wind.parquet').rename(columns={'dmax':'wsp'})
        fr.append(w.merge(i, on=['grid_id','date'], how='inner'))
    d = pd.concat(fr, ignore_index=True).drop_duplicates(['grid_id','date'])
    d = d[np.isfinite(d.swh) & np.isfinite(d.wsp) & d.grid_id.isin(set(panel.grid_id))].copy()
    d['year_month'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m')
    need = []
    for hs, u in ABS_LADDER:
        ew, ei = d.swh >= hs, d.wsp >= u; wo, wv, cp = abs3(hs, u)
        d[wo] = (ei & ~ew).astype('int8'); d[wv] = (ew & ~ei).astype('int8'); d[cp] = (ew & ei).astype('int8'); need += [wo, wv, cp]
    S = d.groupby(['grid_id','year_month'])[need].sum().reset_index()
    panel = panel.drop(columns=[c for c in need if c in panel.columns]).merge(S, on=['grid_id','year_month'], how='left')
    panel[need] = panel[need].fillna(0).astype('int16')
    PA = active(panel)
    say(f'grid{grid}: panel {len(panel):,} rows / {panel.grid_id.nunique():,} cells | active {len(PA):,} rows / {PA.grid_id.nunique():,} cells | '
        f'accidents {int(PA.A_it.sum()):,} | clusters {PA[cluster].nunique():,} | zero-acc cell-months {(panel.A_it==0).mean()*100:.1f}%')
    return PA, cluster

OUT = {}
for grid in (2, 1):
    if not os.path.exists(rf'{BASE}\grid{grid}\panel_global.parquet'):
        say(f'grid{grid} panel missing, skip'); continue
    PA, cluster = load(grid)
    WV_REG = [c for c in PA.columns if c.startswith('Wv_')][1:]; WD_REG = [c for c in PA.columns if c.startswith('Wd_')][1:]
    rows = []
    for y in ('A_it', 'A_wx'):
        for cols, hz in [(WV_REG, 'wave'), (WD_REG, 'wind')]:
            m = fitm(y, cols, PA, FE_ABS, cluster)
            for x in cols:
                if x in set(m.coef().index):
                    irr, lo, hi, pv = row(m, x); rows.append(dict(grid=grid, y=y, block='bins', hz=hz, term=x, IRR=irr, lo=lo, hi=hi, p=pv, n=nobs(m)))
        say(f'grid{grid} {y} bins done')
        for hs, u in ABS_LADDER:
            xs = abs3(hs, u); cp = xs[2]; days = int(PA[cp].sum())
            try:
                m = fitm(y, xs, PA, FE_ABS, cluster); irr, lo, hi, pv = row(m, cp)
                rows.append(dict(grid=grid, y=y, block='compound', hz='cp', term=abstag(hs, u), IRR=irr, lo=lo, hi=hi, p=pv, n=nobs(m), days=days))
            except Exception as e:
                rows.append(dict(grid=grid, y=y, block='compound', hz='cp', term=abstag(hs, u), IRR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=np.nan, days=days))
                say(f'  compound {hs}/{u} FAIL {type(e).__name__}')
        say(f'grid{grid} {y} compound done')
    OUT[grid] = pd.DataFrame(rows)

R = pd.concat(OUT.values(), ignore_index=True)
R.to_csv(rf'{BASE}\grid1\M2_grid1_vs_grid2.csv', index=False)
R['cell'] = R.apply(lambda r: f'{r.IRR:.3f} [{r.lo:.3f},{r.hi:.3f}]{stars(r.p)}' if np.isfinite(r.IRR) else '—', axis=1)
for y in ('A_it', 'A_wx'):
    print(f'\n================ {y} ================')
    t = R[(R.y == y)].pivot_table(index=['block','hz','term'], columns='grid', values='cell', aggfunc='first', sort=False)
    print(t.to_string())
    if 'days' in R:
        dd = R[(R.y == y) & (R.block == 'compound')].pivot_table(index='term', columns='grid', values='days', aggfunc='first', sort=False)
        print('\ncompound days by grid:'); print(dd.to_string())
say('ALL DONE')
