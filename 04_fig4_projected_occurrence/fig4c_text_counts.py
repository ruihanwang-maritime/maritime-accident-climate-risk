# -*- coding: utf-8 -*-
"""
fig4c_text_counts.py — numbers quoted in the Results text for Fig. 4c

Reads the same re-based tables and filters as the Fig. 4c heat maps in M4_v2.ipynb
(2081–2100 vs 2021–2040, traffic-weighted, seasonal level, coefficients RR_rq1_wave / RR_rq1_wind)
and reports, per scenario, over the 5 regions x 4 seasons x 2 hazards = 40 region–season–weather cells:
  * how many two-model means are negative,
  * in how many cells ACCESS-CM2 and EC-Earth3 agree on the sign (the dots in Fig. 4c),
  * the five-region aggregate per season.
Read-only; writes nothing.
"""
import os
import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
TB = f'{DATA_ROOT}/new_climate/_proj_main/tables'
SLICE, WEIGHT, LEVEL = '2081-2100', 'w_trf', 'SEA'
REGIONS = ['NW_Europe', 'NW_Atlantic', 'Med_Black', 'Asia', 'Indian_Ocean_N']
SEASONS = ['DJF', 'MAM', 'JJA', 'SON']
COEF = {'wave': 'RR_rq1_wave', 'wind': 'RR_rq1_wind'}


def select(d, kind):
    return d[(d.slice == SLICE) & (d.weight == WEIGHT) & (d.level == LEVEL) & (d.coef == COEF[kind])]


def main():
    B1 = pd.read_csv(os.path.join(TB, 'B1_risk_region_ref2021_2040.csv'))
    B2 = pd.read_csv(os.path.join(TB, 'B2_risk_total_ref2021_2040.csv'))
    for ssp in ['ssp126', 'ssp585']:
        rows = []
        for kind in COEF:
            d = select(B1, kind)
            for reg in REGIONS:
                for sea in SEASONS:
                    v = d[(d.region == reg) & (d.season == sea) & (d.scenario == ssp)].set_index('model')['pct']
                    if len(v) != 2:
                        continue
                    rows.append(dict(hazard=kind, region=reg, season=sea, mean=v.mean(),
                                     agree=np.sign(v.iloc[0]) == np.sign(v.iloc[1])))
        t = pd.DataFrame(rows)
        print(f'\n== {ssp}: {len(t)} region-season-weather cells')
        print(f'   two-model mean < 0       : {(t["mean"] < 0).sum()}')
        print(f'   models agree on the sign : {t.agree.sum()}')
        print('   cells with mean >= 0:')
        print(t[t['mean'] >= 0].round(2).to_string(index=False))
        for kind in COEF:
            d = select(B2, kind)
            agg = {sea: d[(d.season == sea) & (d.scenario == ssp)]['pct'].mean() for sea in SEASONS}
            print(f'   five-region aggregate, {kind}: ' + ', '.join(f'{s} {x:+.2f}%' for s, x in agg.items()))


if __name__ == '__main__':
    main()
