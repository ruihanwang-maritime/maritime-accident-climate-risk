#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Season × threshold heat map of change in exceedance days (Nature style)

Example working folder:
    <working folder>/
        season_heat.py
        C3_seasonal_aggregate_cellpooled.csv

Run:
    python season_heat.py --pdf

Default output:
    <script folder>\remote_figures\
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# ====================================================================
# Paths
# ====================================================================
PROJECT_DIR = Path(__file__).resolve().parent
CSV_DEFAULT = Path(
    rf'{DATA_ROOT}/new_climate/_proj_main/tables/C3_seasonal_aggregate_cellpooled.csv'
)
OUTDIR_DEFAULT = PROJECT_DIR / "remote_figures"

# ====================================================================
# Tunable parameters
# ====================================================================
MODELS = ["ACCESS", "ECEARTH"]
SLICE = "2081-2100"
VALUE = "pct"                  # "pct" or "delta"
VMAX = 35
MIN_BASE = 0.75
HIGHLIGHT = [("wind", 17.2, "JJA")]
KEEP = {
    "wave": [1.25, 2.5, 4.0, 6.0],
    "wind": [8.0, 10.8, 13.9, 17.2],
}

# ====================================================================
# Style constants
# ====================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SEASONS = ["DJF", "MAM", "JJA", "SON"]
SLAB = {"ssp126": "SSP1–2.6", "ssp585": "SSP5–8.5"}
MLAB = {"ACCESS": "ACCESS-CM2", "ECEARTH": "EC-Earth3"}
UNIT = {"wave": "m", "wind": "m s$^{-1}$"}
BLOCK = {
    "wave": "Wave-height\nexceedance days",
    "wind": "Wind-speed\nexceedance days",
}

CMAP = LinearSegmentedColormap.from_list(
    "BuGyRd",
    [
        "#0b3d70", "#1b6ba8", "#4a97c9", "#9cc7e0", "#d9e8f2",
        "#f2f2f0",
        "#fbe3d5", "#f4b79a", "#e07c58", "#c0452c", "#8c1d13",
    ],
)

NA_COLOR = "#e3e3e3"
INK, INK2, INK3 = "#1a1a1a", "#5a6167", "#8a9198"


def _fmt(z, nd=1):
    """+5 / −17.5; signed, drop redundant .0, use typographic minus."""
    r = round(float(z), nd) + 0.0
    return "0" if r == 0 else ("%+g" % r).replace("-", "−")


def _matrix(df, variable, model, scenario, slice_, value, min_base):
    """Value matrix: rows = thresholds, columns = seasons; cells with too thin a seasonal baseline set to NaN."""
    idx = [f"N_{c:g}" for c in KEEP[variable]]
    d = df[
        (df["variable"] == variable)
        & (df["model"] == model)
        & (df["scenario"] == scenario)
        & (df["slice"] == slice_)
        & (df["metric"].isin(idx))
    ]

    def piv(col):
        return (
            d.pivot_table(index="metric", columns="season", values=col)
            .reindex(index=idx, columns=SEASONS)
            .values.astype(float)
        )

    M, B = piv(value), piv("base")
    return np.where(np.isfinite(B) & (B >= min_base), M, np.nan), B


def season_heat(
    SEA_AGG,
    model="ACCESS",
    slice_=SLICE,
    value=VALUE,
    vmax=VMAX,
    min_base=MIN_BASE,
    highlight=HIGHLIGHT,
    panel_letter="a",
    subtitle=None,
    show_base=False,
    savepath=None,
    dpi=300,
):
    """Heat map of 2 variable blocks (Hs / wind speed) × 2 scenario columns."""
    scen = ["ssp126", "ssp585"]
    hl = {(v, f"N_{c:g}", s) for v, c, s in highlight}

    fig = plt.figure(figsize=(9.6, 8.6), dpi=dpi)
    gs = fig.add_gridspec(
        2, 2,
        left=0.145, right=0.965, top=0.855, bottom=0.175,
        wspace=0.10, hspace=0.40,
    )

    any_na = False

    for i, kind in enumerate(["wave", "wind"]):
        for j, ssp in enumerate(scen):
            ax = fig.add_subplot(gs[i, j])
            M, B = _matrix(SEA_AGG, kind, model, ssp, slice_, value, min_base)
            any_na |= bool(np.isnan(M).any())

            ax.imshow(
                np.zeros_like(M),
                cmap=mpl.colors.ListedColormap([NA_COLOR]),
                aspect="auto", vmin=0, vmax=1,
            )
            ax.imshow(
                np.ma.masked_invalid(M),
                cmap=CMAP, vmin=-vmax, vmax=vmax,
                aspect="auto", interpolation="nearest",
            )

            for k in range(M.shape[1] + 1):
                ax.axvline(k - 0.5, color="white", lw=1.8, zorder=3)
            for k in range(M.shape[0] + 1):
                ax.axhline(k - 0.5, color="white", lw=1.8, zorder=3)

            for a in range(M.shape[0]):
                for b in range(M.shape[1]):
                    z = M[a, b]
                    if not np.isfinite(z):
                        ax.text(
                            b, a, "NA", ha="center", va="center",
                            fontsize=9, color=INK3, zorder=4,
                        )
                        continue

                    dark = abs(z) > 0.58 * vmax
                    ax.text(
                        b, a - (0.13 if show_base else 0), _fmt(z),
                        ha="center", va="center", fontsize=9.5,
                        color="white" if dark else "#16202a", zorder=4,
                    )

                    if show_base and np.isfinite(B[a, b]):
                        ax.text(
                            b, a + 0.21, f"({B[a, b]:.1f} d)",
                            ha="center", va="center", fontsize=6.8, zorder=4,
                            color="#dfe8ee" if dark else "#6b757c",
                        )

                    if (kind, f"N_{KEEP[kind][a]:g}", SEASONS[b]) in hl:
                        ax.add_patch(
                            Rectangle(
                                (b - 0.5, a - 0.5), 1, 1,
                                fill=False, ec="#111111", lw=1.8, zorder=6,
                            )
                        )

            ax.set_xticks(range(4))
            ax.set_xticklabels(SEASONS, fontsize=10.5, color=INK)
            ax.set_yticks(range(len(KEEP[kind])))

            labs = [f"$\\geq$ {c:g}" for c in KEEP[kind]]
            labs[-1] += f" {UNIT[kind]}"
            ax.set_yticklabels(labs if j == 0 else [], fontsize=10, color=INK)
            ax.tick_params(length=0, pad=4)

            for sp in ax.spines.values():
                sp.set_visible(False)

            ax.set_title(SLAB.get(ssp, ssp), fontsize=12.5, color=INK, pad=9)

            if j == 0:
                pos = ax.get_position()
                fig.text(
                    0.038, pos.y0 + pos.height / 2, BLOCK[kind],
                    rotation=90, va="center", ha="center",
                    fontsize=11.5, color=INK,
                )

    cax = fig.add_axes([0.30, 0.080, 0.40, 0.016])
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(-vmax, vmax),
            cmap=CMAP,
        ),
        cax=cax,
        orientation="horizontal",
    )
    cb.outline.set_visible(False)

    ticks = [-vmax, -vmax * 4 / 7, 0, vmax * 4 / 7, vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}".replace("-", "−") for t in ticks])
    cax.tick_params(length=3, width=0.8, colors=INK2, labelsize=9.5)
    cax.set_xlabel(
        {
            "pct": "Change in exceedance days (%)",
            "delta": "Change in exceedance days (d per season)",
        }[value],
        fontsize=10.5, color=INK, labelpad=6,
    )
    cax.xaxis.set_label_position("top")

    fig.text(0.028, 0.955, panel_letter, fontsize=15, fontweight="bold", color=INK)
    fig.text(
        0.068, 0.953,
        "Seasonal redistribution of threshold-exceedance days",
        fontsize=14, color=INK,
    )

    if subtitle is None:
        subtitle = (
            f"{slice_} vs ERA5 2002–2020 · {MLAB.get(model, model)} "
            "· five regions pooled (all 2° cells)"
        )
    fig.text(0.068, 0.922, subtitle, fontsize=10, color=INK2)

    note = []
    if show_base:
        note.append(
            "Parentheses = baseline exceedance days per season (ERA5 2002–2020)."
        )
    if any_na:
        note.append(
            f"Grey = undefined because baseline exceedance days < {min_base:g} d."
        )
    if note:
        fig.text(0.50, 0.028, "  ".join(note), fontsize=8.5, color=INK2, ha="center")

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


def main():
    p = argparse.ArgumentParser(description="seasonal threshold-exceedance heatmap")
    p.add_argument(
        "--csv",
        default=str(CSV_DEFAULT),
        help="C3_seasonal_aggregate_cellpooled.csv path",
    )
    p.add_argument(
        "--outdir",
        default=str(OUTDIR_DEFAULT),
        help="output folder for figures (created automatically)",
    )
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--slice", dest="slice_", default=SLICE)
    p.add_argument("--value", default=VALUE, choices=["pct", "delta"])
    p.add_argument("--vmax", type=float, default=VMAX)
    p.add_argument("--min-base", type=float, default=MIN_BASE)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show-base", action="store_true", help="show seasonal baseline days below each number")
    p.add_argument("--pdf", action="store_true", help="also write vector PDF")
    a = p.parse_args()

    csv_path = Path(a.csv)
    outdir = Path(a.outdir)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}\n"
            "Make sure the CSV is in the same folder as season_heat.py, or pass the full path with --csv."
        )

    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Project directory : {PROJECT_DIR}")
    print(f"Input CSV         : {csv_path}")
    print(f"Output directory  : {outdir}\n")

    SEA_AGG = pd.read_csv(csv_path)

    required_columns = {
        "model", "scenario", "slice", "variable", "metric",
        "season", "base", "pct", "delta",
    }
    missing = required_columns - set(SEA_AGG.columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    for m in a.models:
        stem = outdir / f"F_seasonheat_{m}_{a.slice_}_{a.value}"

        fig = season_heat(
            SEA_AGG,
            model=m,
            slice_=a.slice_,
            value=a.value,
            vmax=a.vmax,
            min_base=a.min_base,
            show_base=a.show_base,
            savepath=stem.with_suffix(".png"),
            dpi=a.dpi,
        )

        if a.pdf:
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

        plt.close(fig)
        print(f"saved → {stem.with_suffix('.png')}")
        if a.pdf:
            print(f"saved → {stem.with_suffix('.pdf')}")

    print(f"\nAll figures in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
