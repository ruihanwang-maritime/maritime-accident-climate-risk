"""
reorganize_by_target.py
========================
Re-split the downloaded GMTDS monthly tiles by [the exact bounds of the 6 study regions in TARGET_BOXES]:
for each region and month, mosaic all source tiles falling in the box, then crop to the box shape,
writing 1 clean GeoTIFF. Box edges (e.g. 20N, 10E) actually cut through the original sea tiles.

Output: OUT/<region>/GMTDS_<region>_<YYYY>_<MM>.tif   (6 regions × 132 months)
Also:   OUT/_coverage_report.csv  (share of in-box pixels with data, per region and month)

Workflow:
  1) Run with DRY_RUN=True first: quick "tile assignment" report only (source seas per region, months covered), no files written.
  2) Once confirmed, set DRY_RUN=False: actual mosaic + crop + write.

Requires: rasterio, numpy, shapely
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README

import os, re, glob, csv, sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from shapely.geometry import box as sbox
from shapely.ops import unary_union

# ----------------------- Config -----------------------
# Format: (south_lat, north_lat, west_lon, east_lon)
TARGET_BOXES = {
    "NW_Europe":   [(43, 63, -15, 0), (48, 63, 0, 10)],
    "NW_Atlantic": [(18, 50, -98, -60)],
    "SE_Asia":     [(-10, 20, 95, 127)],
    "E_Asia":      [(20, 42, 117, 145)],
    "Med_Black":   [(30, 43, -5.5, 0), (30, 47.5, 0, 42)],
    "Baltic":      [(53, 66, 10, 30)],
}

BASE      = rf"{DATA_ROOT}/climate_AIS_data/GMTDS"
SRC_ROOT  = os.path.join(BASE, "extracted")   # extracted only (has all seas; east_asia_CMTDS is a duplicate)
OUT       = os.path.join(BASE, "REGION_ORGANIZED")
YEARS     = range(2011, 2022)                 # 2011-2021
INDEX_JSON = os.path.join(BASE, "_gmtds_tile_index.json")  # tile bounds cache, avoids rescanning
DRY_RUN   = False                             # True=assignment report only, no files; set False once confirmed
ALL_TOUCHED = True                            # count edge pixels as inside when cropping (True is looser)
# ----------------------------------------------------

MON_RE = re.compile(r"_(\d{4})_(\d{2})\.tif$", re.I)


def latlon_bounds(ds):
    b = ds.bounds
    try:
        return transform_bounds(ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
    except Exception:
        return (b.left, b.bottom, b.right, b.top)


def build_index(use_cache=True):
    """Scan all tifs under SRC_ROOT -> {path: [year, month, w, s, e, n]}, cached to json."""
    if use_cache and os.path.exists(INDEX_JSON):
        with open(INDEX_JSON) as f:
            idx = json.load(f)
        print(f"Loaded tile index cache: {len(idx)} tifs  ({INDEX_JSON})")
        return idx
    idx = {}
    tifs = glob.glob(os.path.join(SRC_ROOT, "**", "*.tif"), recursive=True)
    print(f"Scanning true coordinates of {len(tifs)} source tifs...")
    for i, t in enumerate(tifs, 1):
        m = MON_RE.search(os.path.basename(t))
        if not m:
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        if y not in YEARS:
            continue
        with rasterio.open(t) as ds:
            w, s, e, n = latlon_bounds(ds)
        idx[t] = [y, mo, round(w, 4), round(s, 4), round(e, 4), round(n, 4)]
        if i % 500 == 0:
            print(f"  {i}/{len(tifs)}")
    with open(INDEX_JSON, "w") as f:
        json.dump(idx, f)
    print(f"Index cached: {len(idx)} tifs")
    return idx


def region_geometry(boxes):
    """Return (union_bbox=(W,S,E,N), shapely polygon)."""
    W = min(b[2] for b in boxes); S = min(b[0] for b in boxes)
    E = max(b[3] for b in boxes); N = max(b[1] for b in boxes)
    poly = unary_union([sbox(w, s, e, n) for (s, n, w, e) in boxes])
    return (W, S, E, N), poly


def intersects(tb, box):
    w, s, e, n = tb
    W, S, E, N = box
    return not (e <= W or w >= E or n <= S or s >= N)


def main():
    idx = build_index()
    # group tiles by (year, month)
    by_month = {}
    for path, (y, mo, w, s, e, n) in idx.items():
        by_month.setdefault((y, mo), []).append((path, (w, s, e, n)))

    months = [(y, mo) for y in YEARS for mo in range(1, 13)]
    os.makedirs(OUT, exist_ok=True)
    report_rows = []

    print("\n== Tile assignment report per region ==")
    for region, boxes in TARGET_BOXES.items():
        ubox, poly = region_geometry(boxes)
        # count months contributed by each source sea (by parent folder name)
        contrib = {}
        months_with_data = 0
        for (y, mo) in months:
            hits = [(p, tb) for (p, tb) in by_month.get((y, mo), []) if intersects(tb, ubox)]
            if hits:
                months_with_data += 1
            for p, tb in hits:
                sea = os.path.basename(os.path.dirname(p))
                contrib[sea] = contrib.get(sea, 0) + 1
        seas = ", ".join(f"{k}×{v//12 if v%12==0 else v}" for k, v in sorted(contrib.items(), key=lambda x:-x[1]))
        print(f"\n[{region}]  union bbox(W,S,E,N)={tuple(round(v,1) for v in ubox)}")
        print(f"   Months with data: {months_with_data}/132")
        print(f"   Source seas: {seas}")

    if DRY_RUN:
        print("\n== DRY_RUN=True: above is the assignment plan, no files written. Set DRY_RUN to False and rerun once confirmed. ==")
        return

    # ---- Actual mosaic + crop + write ----
    print("\nMosaicking and cropping by region...")
    for region, boxes in TARGET_BOXES.items():
        ubox, poly = region_geometry(boxes)
        W, S, E, N = ubox
        outdir = os.path.join(OUT, region)
        os.makedirs(outdir, exist_ok=True)
        res = None  # use first tile's resolution so all months share one grid
        for (y, mo) in months:
            hits = [(p, tb) for (p, tb) in by_month.get((y, mo), []) if intersects(tb, ubox)]
            if not hits:
                report_rows.append([region, y, mo, 0, 0, 0.0]); continue
            dss = [rasterio.open(p) for p, _ in hits]
            try:
                if res is None:
                    res = dss[0].res  # (xres, yres)
                mosaic, transform = merge(dss, bounds=(W, S, E, N), res=res)
                prof = dss[0].profile.copy()
                nod = dss[0].nodata
                if nod is None:
                    nod = 0
            finally:
                for d in dss:
                    d.close()
            band = mosaic[0]
            outshape = band.shape
            # crop to box polygon: outside set to nodata
            outside = geometry_mask([poly.__geo_interface__], out_shape=outshape,
                                    transform=transform, invert=False, all_touched=ALL_TOUCHED)
            band = band.copy()
            band[outside] = nod
            # coverage: share of in-box pixels with data (≠nodata)
            inside = ~outside
            valid = inside & (band != nod)
            frac = float(valid.sum()) / float(inside.sum()) if inside.sum() else 0.0
            report_rows.append([region, y, mo, int(inside.sum()), int(valid.sum()), round(frac, 4)])
            # write
            prof.update(height=outshape[0], width=outshape[1], transform=transform,
                        count=1, dtype=band.dtype, nodata=nod, compress="lzw")
            dst_path = os.path.join(outdir, f"GMTDS_{region}_{y}_{mo:02d}.tif")
            with rasterio.open(dst_path, "w", **prof) as dst:
                dst.write(band, 1)
        print(f"  Done {region}: {sum(1 for r in report_rows if r[0]==region and r[4]>0)} months with data")

    rep = os.path.join(OUT, "_coverage_report.csv")
    with open(rep, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["region", "year", "month", "box_pixels", "valid_pixels", "coverage_frac"])
        w.writerows(report_rows)
    print(f"\n✅ Done. Output: {OUT}\nCoverage report: {rep}")


if __name__ == "__main__":
    main()
