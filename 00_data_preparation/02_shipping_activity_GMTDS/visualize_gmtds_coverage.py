"""
visualize_gmtds_coverage.py
===========================
Map on a world map "which seas the downloaded GMTDS data actually cover".
Extents are read from each GeoTIFF's true georeference (ignoring folder names); for each sea area
the number of unique months present in 2011-2021 is counted and used for colouring.

Choose one data source (set MODE):
  MODE="raw"       : scan the raw extracted / east_asia_CMTDS (unorganized, actual current state)
  MODE="organized" : scan the ORGANIZED folder from organize_gmtds.py (each sea should be 132/132)

Requires: rasterio, matplotlib, cartopy
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README

import os, re, glob, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from rasterio.warp import transform_bounds

# ----------------------- Config -----------------------
MODE      = "raw"     # "raw" = raw state; "organized" = organized result
BASE      = rf"{DATA_ROOT}/climate_AIS_data/GMTDS"
RAW_DIRS  = [os.path.join(BASE, "extracted"), os.path.join(BASE, "east_asia_CMTDS")]
ORGANIZED = os.path.join(BASE, "ORGANIZED")
YEARS     = range(2011, 2022)     # 2011-2021, full = 132 months
CLUSTER_D = 1.5                   # tifs with centres closer than this (deg) are one sea
# ----------------------------------------------------

MON_RE = re.compile(r"_(\d{4})_(\d{2})\.tif$", re.I)


def latlon_bounds(ds):
    b = ds.bounds
    try:
        return transform_bounds(ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
    except Exception:
        return (b.left, b.bottom, b.right, b.top)


def scan_tifs(dirs):
    """Walk all tifs under the directory trees, return [(west,south,east,north, year, month), ...]."""
    recs = []
    for d in dirs:
        for t in glob.glob(os.path.join(d, "**", "*.tif"), recursive=True):
            m = MON_RE.search(os.path.basename(t))
            if not m:
                continue
            y, mo = int(m.group(1)), int(m.group(2))
            if y not in YEARS:
                continue
            with rasterio.open(t) as ds:
                w, s, e, n = latlon_bounds(ds)
            recs.append((round(w, 3), round(s, 3), round(e, 3), round(n, 3), y, mo))
    return recs


def cluster(recs):
    """Cluster tifs into sea areas by bbox centre; return (representative bbox, unique month count) per sea."""
    clusters = []  # each element: {"cx","cy","bboxes":[],"months":set()}
    for w, s, e, n, y, mo in recs:
        cx, cy = (w + e) / 2, (s + n) / 2
        hit = None
        for c in clusters:
            if abs(c["cx"] - cx) <= CLUSTER_D and abs(c["cy"] - cy) <= CLUSTER_D:
                hit = c
                break
        if hit is None:
            hit = {"cx": cx, "cy": cy, "bboxes": {}, "months": set()}
            clusters.append(hit)
        hit["bboxes"][(w, s, e, n)] = hit["bboxes"].get((w, s, e, n), 0) + 1
        hit["months"].add((y, mo))
    out = []
    for c in clusters:
        rep = max(c["bboxes"].items(), key=lambda kv: kv[1])[0]   # most frequent bbox
        out.append((rep, len(c["months"]), len(c["bboxes"])))
    return out


def main():
    dirs = RAW_DIRS if MODE == "raw" else [ORGANIZED]
    print(f"MODE={MODE}  scanning: {dirs}")
    recs = scan_tifs(dirs)
    print(f"Read {len(recs)} tifs ({min(YEARS)}-{max(YEARS)})")
    regions = cluster(recs)
    print(f"Merged into {len(regions)} true sea areas\n")

    fig = plt.figure(figsize=(20, 10.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0ece1")
    ax.add_feature(cfeature.OCEAN, facecolor="#dceaf2")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#c8c8c8")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.35)

    full = 132  # full month count for 2011-2021
    # draw large areas first so small boxes stay visible
    regions.sort(key=lambda r: -((r[0][2] - r[0][0]) * (r[0][3] - r[0][1])))
    for rep, nmonths, nbbox in regions:
        w, s, e, n = rep
        if nmonths >= full:
            color, ls, lw = "#2e7d32", "-", 1.6            # green: complete
        elif nmonths >= full * 0.5:
            color, ls, lw = "#ef8c00", (0, (5, 2)), 2.4    # orange: partial/extent shift
        else:
            color, ls, lw = "#d32f2f", "--", 2.4           # red: severely incomplete
        ax.add_patch(mpatches.Rectangle((w, s), e - w, n - s, transform=ccrs.PlateCarree(),
                     facecolor=color, edgecolor=color, alpha=0.20, linewidth=lw, linestyle=ls, zorder=5))
        ax.add_patch(mpatches.Rectangle((w, s), e - w, n - s, transform=ccrs.PlateCarree(),
                     facecolor="none", edgecolor=color, linewidth=lw, linestyle=ls, zorder=6))
        tag = f"{nmonths}/{full}" + ("" if nbbox == 1 else f"\n({nbbox} extents)")
        ax.text((w + e) / 2, (s + n) / 2, tag, transform=ccrs.PlateCarree(), fontsize=7.5,
                ha="center", va="center", weight="bold", color="black", zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))

    legend = [
        Line2D([0], [0], color="#2e7d32", lw=3, label="Complete (132/132 months)"),
        Line2D([0], [0], color="#ef8c00", lw=3, ls=(0, (5, 2)), label="Partial / extent shifted mid-series"),
        Line2D([0], [0], color="#d32f2f", lw=3, ls="--", label="Severely incomplete (<50%)"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_title(f"GMTDS downloaded coverage ({MODE}) — {len(regions)} sea regions by REAL TIFF georeferencing\n"
                 f"labels = unique months present in {min(YEARS)}-{max(YEARS)}", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
