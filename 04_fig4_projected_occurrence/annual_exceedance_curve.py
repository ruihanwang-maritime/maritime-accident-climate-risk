#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annual threshold-exceedance curves (Nature-style 2 × 2 layout)

Layout
------
rows    : wave / wind
columns : SSP1–2.6 / SSP5–8.5
colour  : threshold severity
line    : climate model
          solid  = ACCESS-CM2
          dashed = EC-Earth3

Input
-----
Default annual-series table:
<DATA_ROOT>/new_climate/_proj_main/tables/C2_hazard_annual_series.csv

Expected columns:
year, cut, value, base, delta, pct, variable, model, scenario

Output
------
By default, figures are saved into a "remote_figures" folder next to this
script on the remote Windows server.

Typical command
---------------
python annual_exceedance_curve.py --pdf

Notes
-----
1. Zero represents the ERA5 2002–2020 baseline.
2. Projected series are not forced to equal zero in 2020.
3. Default smoothing is a centred 10-year running mean.
4. Near the projection endpoints, at least 5 annual values are used so that
   the curve remains visible. Use --full-window to retain only complete
   10-year windows.
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ====================================================================
# Paths
# ====================================================================

PROJECT_DIR = Path(__file__).resolve().parent

CSV_DEFAULT = Path(
    rf"{DATA_ROOT}/new_climate/_proj_main/tables/C2_hazard_annual_series.csv"
)

OUTDIR_DEFAULT = PROJECT_DIR / "remote_figures"


# ====================================================================
# Controls
# ====================================================================

ROLL = 10
VALUE = "pct"


START_YEAR = 2020
END_YEAR = 2100
YLIM_DEFAULT = (-30, 5)

LATE_START = 2081
LATE_END = 2100

KEEP = {
    "wave": [1.25, 2.5, 4.0, 6.0],
    "wind": [8.0, 10.8, 13.9, 17.2],
}

MODELS = ["ACCESS", "ECEARTH"]
SCENARIOS = ["ssp126", "ssp585"]

MODEL_LABEL = {
    "ACCESS": "ACCESS-CM2",
    "ECEARTH": "EC-Earth3",
}

SCENARIO_LABEL = {
    "ssp126": "SSP1–2.6",
    "ssp585": "SSP5–8.5",
}

MODEL_LINESTYLE = {
    "ACCESS": "-",
    "ECEARTH": (0, (6, 3)),
}

VARIABLE_LABEL = {
    "wave": "Significant wave height",
    "wind": "10-m wind speed",
}

# Light -> dark = more severe threshold
COLORS = {
    "wave": ["#bfd8ee", "#7fb6d6", "#3f8fc1", "#0b5aa6"],
    "wind": ["#fdc58f", "#f79249", "#e85a13", "#a93400"],
}


# ====================================================================
# Style
# ====================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
})

INK = "#1a1a1a"
INK2 = "#5c6166"
GRID = "#dedede"
ZERO = "#9b9b9b"
SHADE = "#f4f4f4"


# ====================================================================
# Helpers
# ====================================================================

def validate_input(df):
    required = {
        "year", "cut", "value", "base", "delta", "pct",
        "variable", "model", "scenario",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )


def rolling_series(s, window=ROLL, full_window=False):
    """
    Centred running mean.

    Default:
        min_periods = window // 2, so the line remains visible near 2020/2100.

    --full-window:
        min_periods = window; only complete windows are retained.
    """
    min_periods = window if full_window else max(1, window // 2)

    return s.rolling(
        window=window,
        center=True,
        min_periods=min_periods,
    ).mean()


def get_baseline(df, kind, cut):
    d = df[
        (df["variable"] == kind)
        & np.isclose(df["cut"].astype(float), float(cut))
    ]["base"].dropna()

    if d.empty:
        return np.nan

    return float(d.median())


def threshold_label(kind, cut, base):
    if kind == "wave":
        lhs = rf"$H_s \geq {cut:g}$ m"
    else:
        lhs = rf"$U_{{10}} \geq {cut:g}$ m s$^{{-1}}$"

    if np.isfinite(base):
        rhs = rf"({base:.0f} d yr$^{{-1}}$)"
    else:
        rhs = "(baseline unavailable)"

    return f"{lhs}    {rhs}"


def choose_ylim(df, value, fixed_ylim=None):
    if fixed_ylim is not None:
        return tuple(float(x) for x in fixed_ylim)

    vals = (
        df[value]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
        .values
    )

    if len(vals) == 0:
        return -30.0, 5.0

    lo = min(float(np.nanpercentile(vals, 1)), 0.0)
    hi = max(float(np.nanpercentile(vals, 99)), 0.0)

    pad = max(2.0, 0.08 * max(hi - lo, 10.0))
    lo -= pad
    hi += pad

    lo = 5 * np.floor(lo / 5)
    hi = 5 * np.ceil(hi / 5)
    hi = max(5.0, hi)

    return float(lo), float(hi)


# ====================================================================
# Main figure
# ====================================================================

def make_figure(
    df,
    value=VALUE,
    roll=ROLL,
    full_window=False,
    start_year=START_YEAR,
    end_year=END_YEAR,
    late_start=LATE_START,
    late_end=LATE_END,
    ylim=None,
    panel_letter="a",
):
    mask = (
        df["variable"].isin(["wave", "wind"])
        & df["model"].isin(MODELS)
        & df["scenario"].isin(SCENARIOS)
        & (df["year"] >= start_year)
        & (df["year"] <= end_year)
    )

    dplot = df.loc[mask].copy()

    if dplot.empty:
        raise ValueError("No data remain after the plotting filters.")

    ylo, yhi = choose_ylim(dplot, value, ylim)

    fig = plt.figure(figsize=(15.6, 8.9), dpi=180)

    gs = fig.add_gridspec(
        2,
        2,
        left=0.115,
        right=0.785,
        top=0.845,
        bottom=0.145,
        wspace=0.10,
        hspace=0.19,
    )

    axes = [[None, None], [None, None]]

    for i, kind in enumerate(["wave", "wind"]):
        for j, scenario in enumerate(SCENARIOS):
            ax = fig.add_subplot(gs[i, j])
            axes[i][j] = ax

            # Highlight late-century period used in seasonal analysis
            ax.axvspan(
                late_start,
                late_end,
                color=SHADE,
                lw=0,
                zorder=0,
            )

            # ERA5 reference
            ax.axhline(
                0,
                color=ZERO,
                lw=0.9,
                zorder=1,
            )

            ax.grid(
                axis="y",
                color=GRID,
                lw=0.65,
                alpha=0.75,
                zorder=0,
            )

            for k, cut in enumerate(KEEP[kind]):
                colour = COLORS[kind][k]

                for model in MODELS:
                    sub = dplot[
                        (dplot["variable"] == kind)
                        & np.isclose(dplot["cut"].astype(float), float(cut))
                        & (dplot["scenario"] == scenario)
                        & (dplot["model"] == model)
                    ].sort_values("year")

                    if sub.empty:
                        continue

                    s = (
                        sub.set_index("year")[value]
                        .astype(float)
                        .sort_index()
                    )

                    smooth = rolling_series(
                        s,
                        window=roll,
                        full_window=full_window,
                    )

                    ax.plot(
                        smooth.index,
                        smooth.values,
                        color=colour,
                        linestyle=MODEL_LINESTYLE[model],
                        lw=2.15,
                        solid_capstyle="round",
                        dash_capstyle="round",
                        zorder=4,
                    )

            ax.set_xlim(start_year, end_year)
            ax.set_ylim(ylo, yhi)
            ax.set_xticks([2020, 2040, 2060, 2080, 2100])

            yrange = yhi - ylo
            ystep = 10 if yrange >= 30 else 5
            yticks = np.arange(
                np.ceil(ylo / ystep) * ystep,
                np.floor(yhi / ystep) * ystep + 0.1,
                ystep,
            )
            ax.set_yticks(yticks)

            ax.tick_params(
                axis="both",
                labelsize=10,
                colors=INK,
                length=4,
                pad=4,
            )

            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)

            ax.spines["left"].set_color(INK)
            ax.spines["bottom"].set_color(INK)

            if i == 0:
                ax.set_title(
                    SCENARIO_LABEL[scenario],
                    fontsize=14,
                    fontweight="bold",
                    color=INK,
                    pad=12,
                )

            if i == 1:
                ax.set_xlabel(
                    "Year",
                    fontsize=11.5,
                    color=INK,
                    labelpad=6,
                )
            else:
                ax.tick_params(labelbottom=False)

            if j == 1:
                ax.tick_params(labelleft=False)

                ax.text(
                    (late_start + late_end) / 2,
                    yhi - 0.045 * (yhi - ylo),
                    f"{late_start}–{late_end}",
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    color="#777777",
                    zorder=5,
                )

        # Row label
        pos = axes[i][0].get_position()
        fig.text(
            0.053,
            pos.y0 + pos.height / 2,
            VARIABLE_LABEL[kind],
            rotation=90,
            va="center",
            ha="center",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
        )

    # Shared y label
    if value == "pct":
        ylab = "Change in annual threshold-exceedance days (%)"
    else:
        ylab = r"Change in annual threshold-exceedance days (d yr$^{-1}$)"

    fig.text(
        0.020,
        0.49,
        ylab,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        color=INK,
    )

    # ----------------------------------------------------------------
    # Title / subtitle
    # ----------------------------------------------------------------
    fig.text(
        0.060,
        0.952,
        panel_letter,
        fontsize=17,
        fontweight="bold",
        color=INK,
    )

    fig.text(
        0.115,
        0.952,
        "Projected change in annual threshold-exceedance days",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )

    fig.text(
        0.115,
        0.918,
        "Relative to ERA5 (2002–2020); five regions combined using "
        f"traffic weights · {roll}-year centred running mean",
        fontsize=10.5,
        color=INK2,
    )

    # ----------------------------------------------------------------
    # Right-side legend blocks
    # ----------------------------------------------------------------
    model_handles = [
        Line2D(
            [0], [0],
            color="#222222",
            lw=2.1,
            linestyle=MODEL_LINESTYLE[m],
            label=MODEL_LABEL[m],
        )
        for m in MODELS
    ]

    leg_model = fig.legend(
        handles=model_handles,
        title="Climate model",
        loc="upper left",
        bbox_to_anchor=(0.812, 0.905),
        frameon=False,
        fontsize=10.5,
        title_fontsize=10.5,
        handlelength=3.1,
        handletextpad=0.8,
        borderaxespad=0,
    )
    leg_model._legend_box.align = "left"

    wave_handles = []
    for k, cut in enumerate(KEEP["wave"]):
        base = get_baseline(df, "wave", cut)

        wave_handles.append(
            Line2D(
                [0], [0],
                color=COLORS["wave"][k],
                lw=2.7,
                label=threshold_label("wave", cut, base),
            )
        )

    leg_wave = fig.legend(
        handles=wave_handles,
        title="Wave threshold   (baseline)",
        loc="upper left",
        bbox_to_anchor=(0.812, 0.735),
        frameon=False,
        fontsize=9.9,
        title_fontsize=10.5,
        handlelength=2.9,
        handletextpad=0.9,
        labelspacing=0.85,
        borderaxespad=0,
    )
    leg_wave._legend_box.align = "left"

    wind_handles = []
    for k, cut in enumerate(KEEP["wind"]):
        base = get_baseline(df, "wind", cut)

        wind_handles.append(
            Line2D(
                [0], [0],
                color=COLORS["wind"][k],
                lw=2.7,
                label=threshold_label("wind", cut, base),
            )
        )

    leg_wind = fig.legend(
        handles=wind_handles,
        title="Wind threshold   (baseline)",
        loc="upper left",
        bbox_to_anchor=(0.812, 0.355),
        frameon=False,
        fontsize=9.9,
        title_fontsize=10.5,
        handlelength=2.9,
        handletextpad=0.9,
        labelspacing=0.85,
        borderaxespad=0,
    )
    leg_wind._legend_box.align = "left"

    # Keep all legends
    fig.add_artist(leg_model)
    fig.add_artist(leg_wave)

    # Small smoothing note
    if full_window:
        note = (
            f"Complete {roll}-year centred windows only; "
            "edge years without a full window are omitted."
        )
    else:
        note = (
            f"At projection edges, centred means use at least "
            f"{max(1, roll // 2)} available annual values."
        )

    fig.text(
        0.115,
        0.055,
        note,
        fontsize=8.5,
        color=INK2,
        ha="left",
    )

    return fig


# ====================================================================
# CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Annual threshold-exceedance trajectory figure."
    )

    parser.add_argument(
        "--csv",
        default=str(CSV_DEFAULT),
        help="Path to C2_hazard_annual_series.csv",
    )

    parser.add_argument(
        "--outdir",
        default=str(OUTDIR_DEFAULT),
        help="Output folder",
    )

    parser.add_argument(
        "--value",
        default=VALUE,
        choices=["pct", "delta"],
    )

    parser.add_argument(
        "--roll",
        type=int,
        default=ROLL,
    )

    parser.add_argument(
        "--full-window",
        action="store_true",
        help="Use only complete running-mean windows.",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
    )

    parser.add_argument(
        "--late-start",
        type=int,
        default=LATE_START,
    )

    parser.add_argument(
        "--late-end",
        type=int,
        default=LATE_END,
    )

    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=YLIM_DEFAULT,
        metavar=("YMIN", "YMAX"),
        help="Common y-axis limits; default = -30 to 5.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save PDF.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    if not csv_path.exists():
        raise FileNotFoundError(
            "Input annual-series CSV not found:\n"
            f"{csv_path}\n\n"
            "Expected default:\n"
            f"{DATA_ROOT}/new_climate/_proj_main/tables/"
            "C2_hazard_annual_series.csv"
        )

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    validate_input(df)

    for col in ["year", "cut", "value", "base", "delta", "pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("Input :", csv_path)
    print("Output:", outdir)
    print("Rows  :", len(df))
    print()

    print("ERA5 baselines:")
    for kind in ["wave", "wind"]:
        for cut in KEEP[kind]:
            base = get_baseline(df, kind, cut)
            print(f"  {kind:4s} >= {cut:5g}: {base:7.2f} d yr^-1")
    print()

    fig = make_figure(
        df=df,
        value=args.value,
        roll=args.roll,
        full_window=args.full_window,
        start_year=args.start_year,
        end_year=args.end_year,
        late_start=args.late_start,
        late_end=args.late_end,
        ylim=args.ylim,
    )

    stem = outdir / (
        f"F_annual_threshold_curves_{args.roll}yr_{args.value}"
    )

    png_path = stem.with_suffix(".png")
    fig.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    print("saved ->", png_path)

    if args.pdf:
        pdf_path = stem.with_suffix(".pdf")
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )
        print("saved ->", pdf_path)

    plt.close(fig)


if __name__ == "__main__":
    main()
