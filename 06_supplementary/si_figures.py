import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
# ═══════════════════════════════════════════════════════════════════════════════
#  SI figures — Cell 0: shared style and paths (used by all SI figures; run this cell first)
# ═══════════════════════════════════════════════════════════════════════════════
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm

SAVE_DIR = os.environ.get('SI_SAVE_DIR') or None                     # e.g. 'SI_figs'; None = display only
DPI      = 300
FS = dict(base=17, tick=20, label=24, legend=18, letter=22, annot=18)   # ★ all font sizes, uniformly enlarged
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': FS['base'], 'axes.labelsize': FS['label'], 'xtick.labelsize': FS['tick'],
    'ytick.labelsize': FS['tick'], 'legend.fontsize': FS['legend'], 'axes.titlesize': FS['base'],
    'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#444444', 'axes.linewidth': 1.0, 'xtick.color': '#333333', 'ytick.color': '#333333',
    'legend.frameon': False, 'figure.dpi': 110, 'savefig.dpi': DPI, 'mathtext.default': 'regular'})
INK, INK2, GREY = '#1a1a1a', '#5c6166', '#9aa0a6'
WAVE_C, WIND_C, CP_C = '#2c6e91', '#d1603d', '#7b5aa6'
MODEL_LAB = {'ACCESS': 'ACCESS-CM2', 'ECEARTH': 'EC-Earth3'}
SSP_LAB   = {'ssp126': 'SSP1-2.6', 'ssp585': 'SSP5-8.5'}
SSP_C     = {'ssp126': '#3b7dd8', 'ssp585': '#e0453f'}
REG_LAB   = {'NW_Europe': 'NWE', 'NW_Atlantic': 'NWA', 'Med_Black': 'MBS', 'Asia': 'ESA', 'Indian_Ocean_N': 'NIO', 'Baltic': 'BALT'}

G      = rf'{DATA_ROOT}'
P_M2   = rf'{G}\AIS_processing\panel_cache_M2'
P_M3   = os.environ.get('M3_OUT_DIR', os.path.join('..', '03_fig3_accident_consequences', 'M3_reanalysis_outputs'))  # folder written by M3_reanalysis_complete.py
P_PROJ = rf'{G}\new_climate\_proj_main\tables'
P_GLOB = rf'{G}\new_climate\_proj_global\tables'
P_M5   = rf'{G}\new_climate\_M5'

def letter(ax, s, *_):
    fig = ax.figure; fig._letters = getattr(fig, '_letters', []) + [(ax, s)]
def place_letters(fig, dx=0.004, dy=0.006):
    fig.canvas.draw(); ren = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, s in getattr(fig, '_letters', []):
        bb = ax.get_tightbbox(ren); x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(x0 - dx, y1 + dy, s, fontsize=FS['letter'], fontweight='bold', color=INK, ha='right', va='bottom')
    fig._letters = []
def place_fig_legends(fig, gap=0.012):
    if not fig.legends: return
    fig.canvas.draw(); ren = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    ymin = min(inv.transform((0, ax.get_tightbbox(ren).y0))[1] for ax in fig.axes)
    for leg in list(fig.legends):
        handles = leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles
        labels = [t.get_text() for t in leg.texts]; ncol = getattr(leg, '_ncols', 1)
        leg.remove()
        fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(.5, ymin - gap), ncol=ncol, frameon=False)
def grid(ax, axis='y'):
    ax.grid(axis=axis, lw=.5, color='#e6e9ec', zorder=0); ax.set_axisbelow(True)
def is_sig(lo, hi): return not (lo <= 1 <= hi)
import textwrap
def wrap_axis_labels(fig, frac=0.92):
    fig.canvas.draw(); ren = fig.canvas.get_renderer()
    for ax in fig.axes:
        for axis, setter in ((ax.xaxis, ax.set_xlabel), (ax.yaxis, ax.set_ylabel)):
            lab = axis.get_label_text()
            if not lab or chr(10) in lab: continue
            bb = ax.get_window_extent(ren); avail = (bb.width if axis is ax.xaxis else bb.height) * frac
            need = axis.label.get_window_extent(ren); need = need.width if axis is ax.xaxis else need.height
            if need > avail:
                n = max(10, int(len(lab) * avail / need))
                setter(chr(10).join(textwrap.wrap(lab, n, break_long_words=False)))
def finish(fig, name):
    wrap_axis_labels(fig)
    place_fig_legends(fig)
    place_letters(fig)
    if SAVE_DIR:
        os.makedirs(SAVE_DIR, exist_ok=True)
        fig.savefig(os.path.join(SAVE_DIR, f'{name}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(SAVE_DIR, f'{name}.pdf'), bbox_inches='tight')
    plt.show()
print('SI style loaded')


# ═══════════════════════════════════════════════════════════════════════════════
#  Supplementary Fig. 2 — resolution sensitivity of Fig. 2: 2° main setting vs 1° (reads M2_grid1_vs_grid2.csv)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_S2_resolution(y='A_it'):
    R = pd.read_csv(rf'{P_M2}\grid1\M2_grid1_vs_grid2.csv'); R = R[R.y == y]
    LAB = {'Wv_1p25_2p5': 'Moderate (1.25–2.5 m)', 'Wv_2p5_4': 'Rough (2.5–4 m)', 'Wv_4_6': 'Very rough (4–6 m)',
           'Wv_6_9': 'High (6–9 m)', 'Wv_9_p': 'Very high (≥9 m)',
           'Wd_8p0_10p8': 'Bft 5 (8–10.8 m s$^{-1}$)', 'Wd_10p8_13p9': 'Bft 6 (10.8–13.9)', 'Wd_13p9_17p2': 'Bft 7 (13.9–17.2)',
           'Wd_17p2_20p8': 'Bft 8 (17.2–20.8)', 'Wd_20p8_p': 'Bft ≥9 (≥20.8)',
           '1p25_8': 'Moderate + Bft 5', '2p5_10p8': 'Rough + Bft 6', '4_13p9': 'Very rough + Bft 7',
           '6_17p2': 'High + Bft 8', '9_20p8': 'Very high + Bft 9'}
    panels = [('bins', 'wave', WAVE_C, 'a'), ('bins', 'wind', WIND_C, 'b'), ('compound', 'cp', CP_C, 'c')]
    STY = {2: dict(marker='o', off=+0.18), 1: dict(marker='s', off=-0.18)}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6))
    for ax, (blk, hz, col, pl) in zip(axes, panels):
        d = R[(R.block == blk) & (R.hz == hz)]
        terms = [t for t in LAB if t in set(d.term)]; n = len(terms)
        for i, t in enumerate(terms):
            yy = n - 1 - i
            if i % 2 == 0: ax.axhspan(yy-.5, yy+.5, color='#f4f5f6', lw=0, zorder=0)
            for g, st in STY.items():
                r = d[(d.term == t) & (d.grid == g)]
                if not len(r): continue
                r = r.iloc[0]; c, l, h = (r.IRR-1)*100, (r.lo-1)*100, (r.hi-1)*100
                sig = is_sig(r.lo, r.hi)
                ax.errorbar(c, yy+st['off'], xerr=[[c-l], [h-c]], fmt=st['marker'], ms=9,
                            mfc=col if sig else 'white', mec=col, mew=1.8,
                            ecolor=col, elinewidth=1.6, capsize=3.5, capthick=1.4, zorder=4)
        ax.set_yticks(range(n)); ax.set_yticklabels([LAB[t] for t in terms][::-1])
        ax.axvline(0, color='#4a4a4a', lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.set_ylim(-.6, n-.4); ax.set_xlabel('Accident-rate change per day (%)')
        ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False); grid(ax, 'x')
        letter(ax, pl, -0.55 if pl != 'c' else -0.5)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', ms=9, mfc='#4a4a4a', mec='#4a4a4a', label='2° × 2° (main)'),
                        Line2D([], [], marker='s', ls='', ms=9, mfc='#4a4a4a', mec='#4a4a4a', label='1° × 1° (clustered on 2° blocks)'),
                        Line2D([], [], marker='o', ls='', ms=9, mfc='#4a4a4a', mec='#4a4a4a', label='95% CI excludes 0'),
                        Line2D([], [], marker='o', ls='', ms=9, mfc='white', mec='#4a4a4a', mew=1.8, label='95% CI includes 0')],
               loc='lower center', bbox_to_anchor=(.5, -.02), ncol=4)
    fig.tight_layout(rect=(0, .06, .985, 1), w_pad=3); finish(fig, 'FigS2_resolution_sensitivity')


# ═══════════════════════════════════════════════════════════════════════════════
#  Not in final SI (X1) — compound-day threshold sensitivity (run inside M2_update_v3; uses in-memory T2A, ABS_GRID, abslab)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_X1_compound_sensitivity(T2A, ABS_GRID, y='A_it'):
    d = T2A[T2A.y == y]
    combos = ABS_GRID[::-1]; n = len(combos)
    KIND = [('compound', CP_C, 'o', +0.18, 'Compound vs. days below both thresholds'),
            ('cp_vs_all', '#b59ad6', 's', -0.18, 'Compound vs. all other days')]
    fig, ax = plt.subplots(figsize=(10, 0.85*n + 2))
    for i, (hs, u) in enumerate(combos):
        if i % 2 == 0: ax.axhspan(i-.5, i+.5, color='#f4f5f6', lw=0, zorder=0)
        for kind, col, mk, off, _ in KIND:
            r = d[(d.hs == hs) & (d.u == u) & (d.kind == kind)]
            if not len(r): continue
            r = r.iloc[0]; c, l, h = (r.IRR-1)*100, (r.lo-1)*100, (r.hi-1)*100
            ax.errorbar(c, i+off, xerr=[[c-l], [h-c]], fmt=mk, ms=9.5, mfc=col if is_sig(r.lo, r.hi) else 'white',
                        mec=col, mew=1.8, ecolor=col, elinewidth=1.6, capsize=3.5, zorder=4)
    ax.set_yticks(range(n)); ax.set_yticklabels([f'Hs ≥ {hs:g} m & U$_{{10}}$ ≥ {u:g} m s$^{{-1}}$' for hs, u in combos])
    ax.axvline(0, color='#4a4a4a', lw=1.1, ls=(0, (4, 3)), zorder=1); ax.set_ylim(-.6, n-.4)
    ax.set_xlabel('Change in accident rate per compound-extreme day (%)')
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False); grid(ax, 'x')
    ax.legend(handles=[Line2D([], [], marker=mk, ls='', ms=9.5, mfc=col, mec=col, label=lab) for _, col, mk, _, lab in KIND]
                      + [Line2D([], [], marker='o', ls='', ms=9.5, mfc='white', mec='#4a4a4a', mew=1.8, label='95% CI includes 0')],
              loc='lower right')
    fig.tight_layout(); finish(fig, 'FigX1_compound_threshold_sensitivity')


# ═══════════════════════════════════════════════════════════════════════════════
#  Not in final SI (X2) — relative-threshold (P95/P99) version: a single vs compound; b compound-day effect by region; c accident type
#            call inside M2_update_v3 after appendix A2–A4 (uses in-memory T2, T3, T4)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_X2_relative(T2, T3, T4, y='A_it', MIN_CLUSTERS=50):
    TAGS = {'P95': ('o', +0.18, 9.5), 'P99': ('D', -0.18, 8.5)}
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), gridspec_kw={'width_ratios': [1, 1.15, 1.15]})
    # a  single vs compound
    ax = axes[0]; HAZ = [('wind_only', 'Wind only', WIND_C), ('wave_only', 'Wave only', WAVE_C), ('compound', 'Compound', CP_C)]
    for i, (key, lab, col) in enumerate(HAZ):
        yy = len(HAZ) - 1 - i
        for tag, (mk, off, ms) in TAGS.items():
            r = T2[(T2.y == y) & (T2.hazard == f'{key}_{tag}')]
            if not len(r): continue
            r = r.iloc[0]; c, l, h = (r.IRR-1)*100, (r.lo-1)*100, (r.hi-1)*100
            ax.errorbar(c, yy+off, xerr=[[c-l], [h-c]], fmt=mk, ms=ms, mfc=col if is_sig(r.lo, r.hi) else 'white',
                        mec=col, mew=1.8, ecolor=col, elinewidth=1.6, capsize=3.5, zorder=4)
    ax.set_yticks(range(len(HAZ))); ax.set_yticklabels([h[1] for h in HAZ][::-1]); ax.set_ylim(-.6, len(HAZ)-.4)
    letter(ax, 'a', -0.3)
    # b  by region (compound)
    ax = axes[1]; d = T3[T3.y == y]
    order = d[d.tag == 'P95'].sort_values('IRR')['region'].tolist()
    for i, reg in enumerate(order):
        yy = len(order) - 1 - i
        if i % 2 == 0: ax.axhspan(yy-.5, yy+.5, color='#f4f5f6', lw=0, zorder=0)
        for tag, (mk, off, ms) in TAGS.items():
            r = d[(d.region == reg) & (d.tag == tag)]
            if not len(r): continue
            r = r.iloc[0]; c, l, h = (r.IRR-1)*100, (r.lo-1)*100, (r.hi-1)*100
            ax.errorbar(c, yy+off, xerr=[[c-l], [h-c]], fmt=mk, ms=ms, mfc=CP_C if is_sig(r.lo, r.hi) else 'white',
                        mec=CP_C, mew=1.8, ecolor=CP_C, elinewidth=1.6, capsize=3.5,
                        alpha=.45 if r.cells < MIN_CLUSTERS else 1, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'{REG_LAB.get(r, r)} ({int(d[d.region == r].n.iloc[0]):,})' for r in order][::-1]); ax.set_ylim(-.6, len(order)-.4)
    letter(ax, 'b', -0.3)
    # c  accident type (compound)
    ax = axes[2]; NAME = {'Fire_Explosion': 'Fire / explosion', 'Hull_Damage': 'Hull damage', 'Piracy_Other': 'Piracy / other', 'Misc': 'Miscellaneous'}
    types = T4.drop_duplicates('type').sort_values('IRR', ascending=False)['type'].tolist()
    for i, t in enumerate(types):
        yy = len(types) - 1 - i
        if i % 2 == 0: ax.axhspan(yy-.5, yy+.5, color='#f4f5f6', lw=0, zorder=0)
        for tag, (mk, off, ms) in TAGS.items():
            r = T4[(T4.type == t) & (T4.tag == tag)]
            if not len(r): continue
            r = r.iloc[0]; c, l, h = (r.IRR-1)*100, (r.lo-1)*100, (r.hi-1)*100
            ax.errorbar(c, yy+off, xerr=[[c-l], [h-c]], fmt=mk, ms=ms, mfc=CP_C if is_sig(r.lo, r.hi) else 'white',
                        mec=CP_C, mew=1.8, ecolor=CP_C, elinewidth=1.6, capsize=3.5, zorder=4)
    ax.set_yticks(range(len(types))); ax.set_yticklabels([NAME.get(t, t) for t in types][::-1]); ax.set_ylim(-.6, len(types)-.4)
    letter(ax, 'c', -0.42)
    for ax in axes:
        ax.axvline(0, color='#4a4a4a', lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.set_xlabel('Change in accident rate per extreme day (%)')
        ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False); grid(ax, 'x')
    fig.legend(handles=[Line2D([], [], marker='o', ls='', ms=9.5, mfc='#4a4a4a', mec='#4a4a4a', label='P95 (sub-region × calendar month)'),
                        Line2D([], [], marker='D', ls='', ms=8.5, mfc='#4a4a4a', mec='#4a4a4a', label='P99'),
                        Line2D([], [], marker='o', ls='', ms=9.5, mfc='white', mec='#4a4a4a', mew=1.8, label='95% CI includes 0')],
               loc='lower center', bbox_to_anchor=(.5, -.02), ncol=3)
    fig.tight_layout(rect=(0, .06, 1, 1)); finish(fig, 'FigX2_relative_thresholds')


# ═══════════════════════════════════════════════════════════════════════════════
#  Supplementary Figs. 3–4 — Fig. 3 SI: weather effects at representative ship age / tonnage (reads M3_reanalysis_outputs)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_S3_S4_effects_at_values(modifier='age'):
    f = 'table_weather_effects_at_representative_ages.csv' if modifier == 'age' else 'table_weather_effects_at_representative_GT.csv'
    T = pd.read_csv(os.path.join(P_M3, f))
    vals = T.drop_duplicates('value_label')['value_label'].tolist()
    cm = plt.get_cmap('viridis'); cols = [cm(x) for x in np.linspace(.15, .85, len(vals))]
    HZ = {'wv': ('Hs (m)', ['<1.25', '1.25–2.5', '2.5–4', '4–6', '≥6']), 'wd': ('U$_{10}$ (m s$^{-1}$)', ['<8', '8–10.8', '10.8–13.9', '13.9–17.2', '≥17.2'])}
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), sharex='col')
    for r, (y, ylab) in enumerate([('ser', 'Serious outcome'), ('tloss', 'Total loss')]):
        for c, (hz, (xl, xt)) in enumerate(HZ.items()):
            ax = axes[r, c]; d = T[(T.outcome == y) & (T.hazard == hz)]
            x = np.arange(5); w = 0.8 / len(vals)
            for k, (v, col) in enumerate(zip(vals, cols)):
                s = d[d.value_label == v].sort_values('category_index')
                xx = x + (k - (len(vals)-1)/2) * w
                ax.bar(xx, s.b, width=w*.95, color=col, edgecolor='white', lw=.5, zorder=3, label=v.replace('GT p', 'GT percentile '))
                ax.errorbar(xx, s.b, yerr=[s.b - s.lo, s.hi - s.b], fmt='none', ecolor=INK, elinewidth=1, capsize=2.5, zorder=4)
            ax.axhline(0, color='#4a4a4a', lw=1); ax.set_xticks(x); ax.set_xticklabels(xt); grid(ax)
            if r == 1: ax.set_xlabel(xl)
            if c == 0: ax.set_ylabel(f'{ylab} (pp)')
            letter(ax, 'abcd'[r*2+c], -0.14 if c == 0 else -0.08)
    axes[0, 0].legend(loc='upper left')
    fig.tight_layout(); finish(fig, f"FigS{3 if modifier == 'age' else 4}_effects_by_{modifier}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Supplementary Fig. 5 — Fig. 3 SI: separate models vs wave and wind mutually adjusted (reads M3_reanalysis_outputs)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_S5_joint_vs_separate():
    A = pd.read_csv(os.path.join(P_M3, 'table_main_separate_hazards.csv')).assign(spec='Separate models (main)')
    B = pd.read_csv(os.path.join(P_M3, 'table_sensitivity_joint_weather.csv')).assign(spec='Wave and wind jointly adjusted')
    T = pd.concat([A, B]); T = T[T.category_index > 0]
    HZ = {'wv': ['1.25–2.5', '2.5–4', '4–6', '≥6'], 'wd': ['8–10.8', '10.8–13.9', '13.9–17.2', '≥17.2']}
    STY = {'Separate models (main)': ('o', -0.16), 'Wave and wind jointly adjusted': ('s', +0.16)}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    for ax, (hz, xt) in zip(axes, HZ.items()):
        col = WAVE_C if hz == 'wv' else WIND_C
        for y, mfc in [('ser', col), ('tloss', 'white')]:
            for spec, (mk, off) in STY.items():
                d = T[(T.hazard == hz) & (T.outcome == y) & (T.spec == spec)].sort_values('category_index')
                x = np.arange(4) + off + (0.06 if y == 'tloss' else -0.06)
                ax.errorbar(x, d.b, yerr=[d.b - d.lo, d.hi - d.b], fmt=mk, ms=9, mfc=mfc, mec=col, mew=1.8,
                            ecolor=col, elinewidth=1.5, capsize=3.5, ls='none', zorder=4)
        ax.axhline(0, color='#4a4a4a', lw=1); ax.set_xticks(range(4)); ax.set_xticklabels(xt); grid(ax)
        ax.set_xlabel('Hs (m)' if hz == 'wv' else 'U$_{10}$ (m s$^{-1}$)')
        letter(ax, 'ab'[list(HZ).index(hz)], -0.2 if hz == 'wv' else -0.1)
    axes[0].set_ylabel('Difference from reference' + chr(10) + 'category (pp)')
    fig.legend(handles=[Line2D([], [], marker='o', ls='', ms=9, mfc='#4a4a4a', mec='#4a4a4a', label='Separate models (main), serious outcome'),
                        Line2D([], [], marker='o', ls='', ms=9, mfc='white', mec='#4a4a4a', mew=1.8, label='Separate models (main), total loss'),
                        Line2D([], [], marker='s', ls='', ms=9, mfc='#4a4a4a', mec='#4a4a4a', label='Jointly adjusted, serious outcome'),
                        Line2D([], [], marker='s', ls='', ms=9, mfc='white', mec='#4a4a4a', mew=1.8, label='Jointly adjusted, total loss')],
               loc='lower center', bbox_to_anchor=(.5, -.03), ncol=2)
    fig.tight_layout(rect=(0, .1, 1, 1)); finish(fig, 'FigS5_joint_vs_separate')


# ═══════════════════════════════════════════════════════════════════════════════
#  Supplementary Fig. 6 — bias-correction evaluation: annual exceedance days in calibration period (2002–2014), ERA5 vs raw model vs QDM (global)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_S6_qdm_evaluation():
    Q = pd.read_csv(os.path.join(P_GLOB, 'check_ref_exceedance.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    SRC = [('era5', 'ERA5', '#4a4a4a'), ('raw', 'Uncorrected model', '#b9bfc4'), ('qdm', 'Bias-corrected (QDM)', None)]
    for ax, (var, unit, col) in zip(axes, [('wave', 'Hs ≥ (m)', WAVE_C), ('wind', 'U$_{10}$ ≥ (m s$^{-1}$)', WIND_C)]):
        d = Q[Q.variable == var]; cuts = sorted(d.cut.unique()); x = np.arange(len(cuts))
        groups = [(m, s) for m in ['ACCESS', 'ECEARTH'] for s in SRC]; w = 0.8 / len(groups)
        for k, (m, (key, lab, c)) in enumerate(groups):
            v = [float(d[(d.model == m) & (d.cut == cut)][key].iloc[0]) for cut in cuts]
            xx = x + (k - (len(groups)-1)/2) * w
            hatch = '' if m == 'ACCESS' else '//'
            ax.bar(xx, v, width=w*.95, color=c or col, edgecolor='white' if not hatch else INK, lw=.6, hatch=hatch, zorder=3)
        ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([f'{c:g}' for c in cuts]); ax.set_xlabel(unit)
        ax.set_ylabel('Exceedance days per year' + chr(10) + '(2002–2014, ocean-area mean)'); grid(ax)
        letter(ax, 'ab'[axes.tolist().index(ax)], -0.12)
    fig.legend(handles=[Patch(facecolor='#4a4a4a', label='ERA5'), Patch(facecolor='#b9bfc4', label='Uncorrected model'),
                        Patch(facecolor=WAVE_C, label='Bias-corrected (QDM), wave panel'), Patch(facecolor=WIND_C, label='Bias-corrected (QDM), wind panel'),
                        Patch(facecolor='white', edgecolor=INK, label='ACCESS-CM2'), Patch(facecolor='white', edgecolor=INK, hatch='//', label='EC-Earth3')],
               loc='lower center', bbox_to_anchor=(.5, -.06), ncol=3)
    fig.tight_layout(rect=(0, .1, 1, 1)); finish(fig, 'FigS6_qdm_evaluation')


# ═══════════════════════════════════════════════════════════════════════════════
#  Supplementary Figs. 7–8 — relative change in annual exceedance days by region and threshold (vs 2021–2040; reads A1_hazard_annual.csv)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_S7_S8_regional_threshold_changes():
    A = pd.read_csv(os.path.join(P_PROJ, 'A1_hazard_annual.csv'))
    A = A[A.metric.str.startswith('N_')].copy(); A['cut'] = A.metric.str[2:].astype(float)
    KEEP = {'wave': [1.25, 2.5, 4.0, 6.0], 'wind': [8.0, 10.8, 13.9, 17.2]}
    RAMP = {'wave': plt.cm.Blues(np.linspace(.45, .95, 4)), 'wind': plt.cm.Oranges(np.linspace(.45, .95, 4))}
    regs = ['NW_Europe', 'NW_Atlantic', 'Med_Black', 'Asia', 'Indian_Ocean_N']
    slices = ['2021-2040', '2041-2060', '2061-2080', '2081-2100']; x = np.arange(len(slices))
    ref = A[A['slice'] == '2021-2040'][['model', 'scenario', 'region', 'variable', 'cut', 'future']].rename(columns={'future': 'ref'})
    A = A.merge(ref, on=['model', 'scenario', 'region', 'variable', 'cut']); A['pct_ref'] = (A.future / A.ref - 1) * 100
    for ssp in ['ssp126', 'ssp585']:
        fig, axes = plt.subplots(2, 5, figsize=(26, 11), sharex=True, sharey='row')
        for r, var in enumerate(['wave', 'wind']):
            for c, reg in enumerate(regs):
                ax = axes[r, c]
                for cut, col in zip(KEEP[var], RAMP[var]):
                    for m, ls in [('ACCESS', '-'), ('ECEARTH', (0, (4, 2)))]:
                        d = A[(A.scenario == ssp) & (A.variable == var) & (A.region == reg) & (A.cut == cut) & (A.model == m)].set_index('slice').reindex(slices)
                        ax.plot(x, d.pct_ref, ls=ls, color=col, lw=2, marker='o', ms=5)
                ax.axhline(0, color='#4a4a4a', lw=.9); grid(ax)
                ax.set_xticks(x); ax.set_xticklabels(['2021–\n2040', '2041–\n2060', '2061–\n2080', '2081–\n2100'])
                if r == 0: ax.text(.5, 1.02, REG_LAB[reg], transform=ax.transAxes, ha='center', va='bottom', fontsize=FS['base'], color=INK)
                if c == 0: ax.set_ylabel(('Hs' if var == 'wave' else 'U$_{10}$') + ' exceedance days' + chr(10) + 'change (%)')
        h = [Line2D([], [], color=RAMP['wave'][i], lw=2, label=f'Hs ≥ {c:g} m') for i, c in enumerate(KEEP['wave'])] + \
            [Line2D([], [], color=RAMP['wind'][i], lw=2, label=f'U$_{{10}}$ ≥ {c:g} m s$^{{-1}}$') for i, c in enumerate(KEEP['wind'])] + \
            [Line2D([], [], color='#4a4a4a', lw=2, ls='-', label='ACCESS-CM2'), Line2D([], [], color='#4a4a4a', lw=2, ls=(0, (4, 2)), label='EC-Earth3')]
        fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, -.01), ncol=5)
        fig.tight_layout(rect=(0, .08, 1, 1), h_pad=3); finish(fig, f"FigS{7 if ssp == 'ssp126' else 8}_regional_threshold_changes_{ssp}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Supplementary Figs. 9–10 — intermediate-period version of Fig. 4c (2041–2060, 2061–2080; reads B1/B2 rebased tables)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_S9_S10_seasonal_heat(slice_='2041-2060', weight='w_trf'):
    B1 = pd.read_csv(os.path.join(P_PROJ, 'B1_risk_region_ref2021_2040.csv')); B2 = pd.read_csv(os.path.join(P_PROJ, 'B2_risk_total_ref2021_2040.csv'))
    SEAS = ['DJF', 'MAM', 'JJA', 'SON']; SSPS = ['ssp126', 'ssp585']
    ROWS = [('NW_Europe', 'NWE'), ('NW_Atlantic', 'NWA'), ('Med_Black', 'MBS'), ('Asia', 'ESA'), ('Indian_Ocean_N', 'NIO'), ('__ALL__', 'All\nregions')]
    f = lambda d: d[(d['slice'] == slice_) & (d.weight == weight) & (d.level == 'SEA')]
    VMAX = 8; cmap = plt.get_cmap('RdBu_r'); norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0, vmax=VMAX)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    for ax, (coef, pl) in zip(axes, [('RR_rq1_wave', 'a'), ('RR_rq1_wind', 'b')]):
        M = np.full((len(ROWS), 8), np.nan); DOT = np.zeros((len(ROWS), 8), bool)
        for i, (reg, _) in enumerate(ROWS):
            d = f(B2) if reg == '__ALL__' else f(B1)[f(B1).region == reg]; d = d[d.coef == coef]
            for j, (sea, ssp) in enumerate([(s, p) for s in SEAS for p in SSPS]):
                v = d[(d.season == sea) & (d.scenario == ssp)].set_index('model')['pct']
                if len(v) == 2: M[i, j] = v.mean(); DOT[i, j] = np.sign(v.iloc[0]) == np.sign(v.iloc[1])
        ax.imshow(M, cmap=cmap, norm=norm, aspect='auto')
        for i in range(len(ROWS)):
            for j in range(8):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:+.1f}', ha='center', va='center', fontsize=FS['annot'], color='white' if abs(M[i, j]) > 4.5 else INK)
                    if DOT[i, j]: ax.plot(j + .36, i - .34, 'o', ms=5, color=INK)
        for j in range(8):
            ax.add_patch(plt.Rectangle((j-.5, -.5), 1, len(ROWS), fill=False, ec=SSP_C[SSPS[j % 2]], lw=1.6, ls=(0, (3, 2)), zorder=5))
        ax.set_xticks(np.arange(0.5, 8, 2)); ax.set_xticklabels(SEAS); ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[1] for r in ROWS])
        ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
        letter(ax, pl, -0.18)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='horizontal', fraction=.05, pad=.12, aspect=40)
    cb.set_label(f'Projected change in accident occurrence rate, {slice_} vs 2021–2040 (%)'); cb.outline.set_visible(False)
    fig.legend(handles=[Line2D([], [], color=SSP_C['ssp126'], lw=1.6, ls=(0, (3, 2)), label='SSP1-2.6'), Line2D([], [], color=SSP_C['ssp585'], lw=1.6, ls=(0, (3, 2)), label='SSP5-8.5'),
                        Line2D([], [], marker='o', ls='', ms=5, color=INK, label='Models agree on sign')], loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=3)
    finish(fig, f"FigS{9 if slice_ == '2041-2060' else 10}_seasonal_heat_{slice_}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Supplementary Figs. 11–12 — Arctic: monthly navigable area fraction, OW vs PC7 (reads M5_monthfrac_*)
# ═══════════════════════════════════════════════════════════════════════════════
PASS_SI = [('NEP', 'Northeast Passage (R1)'), ('NWP', 'Northwest Passage (R2)'), ('TSR', 'Transpolar Sea Route (R3)'), ('Fram_Strait_approach', 'Fram Strait approach (R4)')]
MON = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
def fig_S11_S12_navigability(ssp='ssp585'):
    cmap = plt.get_cmap('YlGnBu'); norm = BoundaryNorm([0, 5, 10, 20, 40, 60, 80, 90, 95, 100], cmap.N)
    fig, axes = plt.subplots(4, 4, figsize=(20, 14), sharex=True, sharey=True)
    for i, (p, plab) in enumerate(PASS_SI):
        for j, (m, ic) in enumerate([(m, ic) for m in ['ACCESS', 'ECEARTH'] for ic in ['OW', 'PC7']]):
            ax = axes[i, j]; fn = os.path.join(P_M5, f'M5_monthfrac_{m}_{ssp}_{ic}_polaris_{p}.csv')
            if not os.path.exists(fn): ax.axis('off'); continue
            d = pd.read_csv(fn, index_col=0); periods = list(d.columns)
            im = ax.imshow(d.values.T, cmap=cmap, norm=norm, aspect='auto')
            ax.set_yticks(range(len(periods))); ax.set_yticklabels(periods); ax.set_xticks(range(12)); ax.set_xticklabels(MON)
            ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
            if i == 0: ax.text(.5, 1.03, f'{MODEL_LAB[m]}, {"open-water vessel" if ic == "OW" else "Polar Class 7"}', transform=ax.transAxes, ha='center', va='bottom', fontsize=FS['base'], color=INK)
            if j == 0: ax.set_ylabel(plab, fontsize=FS['label'])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='horizontal', fraction=.03, pad=.04, aspect=50)
    cb.set_label(f'Navigable area share (%), {SSP_LAB[ssp]}'); cb.outline.set_visible(False)
    finish(fig, f"FigS{11 if ssp == 'ssp126' else 12}_navigability_OW_PC7_{ssp}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Supplementary Fig. 13 — Arctic: share of navigable days with wave data (reads M5_coverage_*)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_S13_coverage(ssp='ssp585'):
    cmap = plt.get_cmap('PuBuGn'); norm = BoundaryNorm([0, .2, .4, .5, .6, .7, .8, .9, .95, 1.0], cmap.N)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=True)
    for i, m in enumerate(['ACCESS', 'ECEARTH']):
        for j, (p, plab) in enumerate(PASS_SI):
            ax = axes[i, j]; fn = os.path.join(P_M5, f'M5_coverage_{m}_{ssp}_{p}.csv')
            if not os.path.exists(fn): ax.axis('off'); continue
            d = pd.read_csv(fn, index_col=0); periods = list(d.columns)
            ax.imshow(d.values.T, cmap=cmap, norm=norm, aspect='auto')
            ax.set_yticks(range(len(periods))); ax.set_yticklabels(periods); ax.set_xticks(range(12)); ax.set_xticklabels(MON)
            ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
            if i == 0: ax.text(.5, 1.03, plab, transform=ax.transAxes, ha='center', va='bottom', fontsize=FS['base'], color=INK)
            if j == 0: ax.set_ylabel(MODEL_LAB[m], fontsize=FS['label'])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='horizontal', fraction=.04, pad=.06, aspect=50)
    cb.set_label(f'Share of navigable days with valid wave data, {SSP_LAB[ssp]}'); cb.outline.set_visible(False)
    finish(fig, f'FigS13_wave_coverage_{ssp}')


# ═══════════════════════════════════════════════════════════════════════════════
#  Supplementary Fig. 14 — Arctic: decomposition of exposure change in newly opened areas (window / sea-state / interaction; reads M5_decomp_*)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_S14_decomposition(kind='new: change 2041–2060 → 2081–2100'):
    D = pd.concat([pd.read_csv(os.path.join(P_M5, f'M5_decomp_{m}.csv')).assign(model=m) for m in ['ACCESS', 'ECEARTH']])
    D = D[D.kind == kind]
    haz = D.hazard.unique().tolist(); pas = [p for p, _ in [(a, b) for a, b in [(x.split(' ')[0], x) for x in D.passage.unique()]]]
    COMP = [('window', 'Window effect (more navigable days)', '#8ec4de'), ('roughness', 'Sea-state effect (rougher navigable days)', '#e08a5b'), ('interaction', 'Interaction', '#b9bfc4')]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=False)
    for ax, hz in zip(axes, haz):
        d = D[D.hazard == hz]; groups = [(p, s, m) for p in D.passage.unique() for s in ['ssp126', 'ssp585'] for m in ['ACCESS', 'ECEARTH']]
        x = np.arange(len(D.passage.unique())); w = 0.8 / 4
        for k, (s, m) in enumerate([(s, m) for s in ['ssp126', 'ssp585'] for m in ['ACCESS', 'ECEARTH']]):
            xx = x + (k - 1.5) * w; base_pos = np.zeros(len(x)); base_neg = np.zeros(len(x))
            for key, lab, col in COMP:
                v = np.array([float(d[(d.passage == p) & (d.ssp == s) & (d.model == m)][key].iloc[0]) if len(d[(d.passage == p) & (d.ssp == s) & (d.model == m)]) else np.nan for p in D.passage.unique()])
                v = np.nan_to_num(v); pos = v > 0
                ax.bar(xx, np.where(pos, v, 0), bottom=base_pos, width=w*.9, color=col, edgecolor=INK if m == 'ECEARTH' else 'white', lw=.7, hatch='' if m == 'ACCESS' else '//', zorder=3)
                ax.bar(xx, np.where(~pos, v, 0), bottom=base_neg, width=w*.9, color=col, edgecolor=INK if m == 'ECEARTH' else 'white', lw=.7, hatch='' if m == 'ACCESS' else '//', zorder=3)
                base_pos += np.where(pos, v, 0); base_neg += np.where(~pos, v, 0)
            tot = np.array([float(d[(d.passage == p) & (d.ssp == s) & (d.model == m)]['dE'].iloc[0]) if len(d[(d.passage == p) & (d.ssp == s) & (d.model == m)]) else np.nan for p in D.passage.unique()])
            ax.plot(xx, tot, marker='_', ms=14, mew=2.5, ls='', color=SSP_C[s], zorder=5)
        ax.axhline(0, color='#4a4a4a', lw=1); ax.set_xticks(x); ax.set_xticklabels([p.split(' (')[0] for p in D.passage.unique()]); grid(ax)
        ax.set_ylabel(f'Change in {hz} navigable days per year'); letter(ax, 'ab'[haz.index(hz)], -0.1)
    h = [Patch(facecolor=c, label=l) for _, l, c in COMP] + [Patch(facecolor='white', edgecolor=INK, label='ACCESS-CM2'), Patch(facecolor='white', edgecolor=INK, hatch='//', label='EC-Earth3'),
         Line2D([], [], marker='_', ms=14, mew=2.5, ls='', color=SSP_C['ssp126'], label='Total change, SSP1-2.6'), Line2D([], [], marker='_', ms=14, mew=2.5, ls='', color=SSP_C['ssp585'], label='Total change, SSP5-8.5')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, -.04), ncol=4)
    fig.tight_layout(rect=(0, .12, 1, 1)); finish(fig, 'FigS14_decomposition')


# ═══════════════════════════════════════════════════════════════════════════════
#  Not in final SI (X3) — Arctic: full-coefficient index (Supplementary Eq. 12 with all five bins, M2 coefficient 95% CI) vs reduced main-text version
#             reads M5_hazard_index_* (Pan-Arctic); main-text version needs LINES from M5_vis_v2; only full version drawn here
# ═══════════════════════════════════════════════════════════════════════════════
def fig_X3_hazard_index_full():
    H = pd.concat([pd.read_csv(os.path.join(P_M5, f'M5_hazard_index_{m}.csv')).assign(model=m) for m in ['ACCESS', 'ECEARTH']])
    H = H[H.passage == 'Pan-Arctic']; periods = ['2041–2060', '2061–2080', '2081–2100']; x = np.arange(3)
    GRP = {'historically navigable': ('Historically navigable', '-', 'o'), 'newly navigable (by 2081–2100)': ('Newly navigable by 2081–2100', (0, (4, 2)), 's')}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, m in zip(axes, ['ACCESS', 'ECEARTH']):
        for ssp in ['ssp126', 'ssp585']:
            for g, (glab, ls, mk) in GRP.items():
                d = H[(H.model == m) & (H.ssp == ssp) & (H.group == g)].set_index('period').reindex(periods)
                off = -0.08 if ssp == 'ssp126' else 0.08
                ax.errorbar(x + off, d.main, yerr=[d.main - d.lo, d.hi - d.main], fmt=mk, ls=ls, color=SSP_C[ssp], ms=8, lw=2, elinewidth=1.3, capsize=3.5, zorder=4)
        ax.axhline(1, color='#4a4a4a', lw=1, ls=':'); ax.set_xticks(x); ax.set_xticklabels(periods); grid(ax)
        ax.text(.5, 1.02, MODEL_LAB[m], transform=ax.transAxes, ha='center', va='bottom', fontsize=FS['base'], color=INK)
        letter(ax, 'ab'[['ACCESS', 'ECEARTH'].index(m)], -0.12)
    axes[0].set_ylabel('Wave-associated occurrence index\n(all five category coefficients)')
    h = [Line2D([], [], color=SSP_C[s], lw=2, label=SSP_LAB[s]) for s in ['ssp126', 'ssp585']] + \
        [Line2D([], [], color='#4a4a4a', lw=2, ls=ls, marker=mk, ms=8, label=glab) for _, (glab, ls, mk) in GRP.items()]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, -.03), ncol=4)
    fig.tight_layout(rect=(0, .08, 1, 1)); finish(fig, 'FigX3_hazard_index_full_coefficients')


# ═══════════════════════════════════════════════════════════════════════════════
#  Supplementary Fig. 1 — zoomed map of each of six study regions: region and sub-region boundaries, cargo accidents in panel (2011–2021)
#            reads panel_cache_M2\grid2\M2_accident_level.parquet; requires cartopy
# ═══════════════════════════════════════════════════════════════════════════════
REGIONS_SI = {
    'NW_Europe':      ('Northwest Europe (NWE)', {'Iberia_Atlantic': [(36, 45, -15, -6)], 'Biscay_Celtic': [(45, 63, -15, -6), (43, 63, -6, 0)], 'North_Sea': [(48, 63, 0, 13)]}),
    'NW_Atlantic':    ('Northwest Atlantic (NWA)', {'GoM_Caribbean': [(18, 31, -98, -60)], 'NE_Coast': [(31, 42, -82, -60), (42, 50, -67, -60)], 'Newfoundland': [(45, 55, -60, -46)]}),
    'Asia':           ('East and Southeast Asia (ESA)', {'SE_Asia': [(-10, 20, 95, 130)], 'E_Asia': [(20, 46, 117, 146)]}),
    'Med_Black':      ('Mediterranean–Black Sea (MBS)', {'Med_West': [(30, 43, -6, 0), (30, 46, 0, 19)], 'Med_East': [(30, 46, 19, 26.5), (30, 41, 26.5, 37)], 'Black': [(41, 47.5, 26.5, 42)]}),
    'Baltic':         ('Baltic Sea (BALT)', {'Baltic': [(53, 66, 13, 30)]}),
    'Indian_Ocean_N': ('Northern Indian Ocean (NIO)', {'Arabian': [(5, 30, 50, 78)], 'BoB': [(5, 26, 78, 95)]}),
}
SUB_LAB = {'Iberia_Atlantic': 'Iberian Atlantic', 'Biscay_Celtic': 'Biscay–Celtic Seas', 'North_Sea': 'North Sea',
           'GoM_Caribbean': 'Gulf of Mexico–Caribbean', 'NE_Coast': 'Northeast Coast', 'Newfoundland': 'Newfoundland',
           'SE_Asia': 'Southeast Asia', 'E_Asia': 'East Asia', 'Med_West': 'W. Mediterranean', 'Med_East': 'E. Mediterranean',
           'Black': 'Black Sea', 'Baltic': 'Baltic Sea', 'Arabian': 'Arabian Sea', 'BoB': 'Bay of Bengal'}
SUB_C = ['#1f6fb4', '#e07b39', '#2e9e78']

def fig_S1_region_maps(in_panel_only=True):
    import pyarrow                                   # must be imported before cartopy (GEOS/PROJ DLL conflict)
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    A = pd.read_parquet(rf'{P_M2}\grid2\M2_accident_level.parquet', columns=['lat', 'lon', 'region', 'subregion', 'in_panel'])
    if in_panel_only: A = A[A.in_panel]
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    for ax, (reg, (rlab, subs)), pl in zip(axes.ravel(), REGIONS_SI.items(), 'abcdef'):
        boxes = [b for bs in subs.values() for b in bs]
        S, N, W, E = min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes), max(b[3] for b in boxes)
        pad = 0.06 * max(N - S, E - W)
        ax.set_extent([W - pad, E + pad, S - pad, N + pad], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#e9e5dc', zorder=0); ax.add_feature(cfeature.OCEAN, facecolor='#f5f8fb', zorder=0)
        ax.coastlines(resolution='50m', lw=.6, color='#8a8f95', zorder=2)
        ax.add_feature(cfeature.BORDERS, lw=.3, color='#b5b9bd', zorder=2)
        handles = []
        for k, (sub, bs) in enumerate(subs.items()):
            col = SUB_C[k % 3]
            d = A[(A.region == reg) & (A.subregion == sub)]
            ax.scatter(d.lon, d.lat, s=9, color=col, alpha=.55, lw=0, transform=ccrs.PlateCarree(), zorder=3)
            for (s_, n_, w_, e_) in bs:
                ax.add_patch(plt.Rectangle((w_, s_), e_ - w_, n_ - s_, fill=False, ec=col, lw=1.8, ls=(0, (5, 3)), transform=ccrs.PlateCarree(), zorder=4))
            handles.append(Line2D([], [], marker='o', ls='', ms=9, color=col, label=f'{SUB_LAB[sub]} ({len(d):,})'))
        gl = ax.gridlines(draw_labels=True, lw=.4, color='#c9ced3', ls=':', zorder=1)
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {'size': FS['tick'], 'color': INK2}; gl.ylabel_style = {'size': FS['tick'], 'color': INK2}
        ax.text(0.02, 0.97, f'{rlab}\n{int((A.region == reg).sum()):,} accidents', transform=ax.transAxes, ha='left', va='top',
                fontsize=FS['base'], color=INK, bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=4), zorder=6)
        ax.legend(handles=handles, loc='lower right', fontsize=FS['legend'] - 1, markerscale=1.2, handletextpad=.4,
                  framealpha=.85, frameon=True, edgecolor='none')
        letter(ax, pl, -0.06, 1.01)
        for sp in ax.spines.values(): sp.set_edgecolor('#8a8f95')
    fig.tight_layout(w_pad=2.5, h_pad=2.5); finish(fig, 'FigS1_region_maps')


# ═══════════════════════════════════════════════════════════════════════════════
#  generate all SI figures readable from disk (fig_X1 / fig_X2 need in-memory objects from M2_update_v3)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    fig_S2_resolution()
    fig_S3_S4_effects_at_values('age'); fig_S3_S4_effects_at_values('size')
    fig_S5_joint_vs_separate()
    fig_S6_qdm_evaluation()
    fig_S7_S8_regional_threshold_changes()
    fig_S9_S10_seasonal_heat('2041-2060'); fig_S9_S10_seasonal_heat('2061-2080')
    fig_S11_S12_navigability('ssp585'); fig_S11_S12_navigability('ssp126')
    fig_S13_coverage('ssp585')
    fig_S14_decomposition()
    fig_X3_hazard_index_full()
    fig_S1_region_maps()
