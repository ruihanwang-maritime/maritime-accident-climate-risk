"""
bucket_to_6region.py
====================
Bucket the raw GMTDS monthly tiles in the old extracted folder into 6 study-region folders by [each tile's true coordinates].
No cropping, whole tiles copied. Tiles crossing a boundary (e.g. South China Sea across 20N) are copied into every region involved, for later handling.

Rule: tile true bounds overlap a region's union bbox -> copy into that region's folder.
Deduplicate: within a region, tiles with [same year-month + same true bbox] (from duplicate/mislabelled source folders) are kept once.

Output: <DATA_ROOT>/climate_AIS_data/GMTDS/6_region/<region>/<original filename>.tif
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README

import os, re, glob, shutil, csv, json, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import rasterio
from rasterio.warp import transform_bounds

# Format: (south, north, west, east)
TARGET_BOXES = {
    "NW_Europe":   [(43, 63, -15, 0), (48, 63, 0, 10)],
    "NW_Atlantic": [(18, 50, -98, -60)],
    "SE_Asia":     [(-10, 20, 95, 127)],
    "E_Asia":      [(20, 42, 117, 145)],
    "Med_Black":   [(30, 43, -5.5, 0), (30, 47.5, 0, 42)],
    "Baltic":      [(53, 66, 10, 30)],
}

BASE       = rf"{DATA_ROOT}/climate_AIS_data/GMTDS"
SRC_ROOT   = os.path.join(BASE, "extracted")
OUT        = os.path.join(BASE, "6_region")
INDEX_JSON = os.path.join(BASE, "_gmtds_tile_index.json")   # reuse previously cached tile coordinate index
YEARS      = range(2011, 2022)
DRY_RUN    = False    # True=assignment plan only, no copy; set False once confirmed
MON_RE     = re.compile(r"_(\d{4})_(\d{2})\.tif$", re.I)

# Extra prepared standalone sources merged into a region (copied whole, no coordinate test).
# Note: 0725_data/ready is 0.00904° all vessel types, a different grid from old data (0.0135°); filename prefix marks it.
EXTRA_SOURCES = {
    "NW_Atlantic": [(os.path.join(BASE, "0725_data", "ready"), "res0009_All")],
}


def latlon_bounds(ds):
    b = ds.bounds
    try:
        return transform_bounds(ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
    except Exception:
        return (b.left, b.bottom, b.right, b.top)


def build_index():
    if os.path.exists(INDEX_JSON):
        idx = json.load(open(INDEX_JSON))
        print(f"Reusing tile index cache: {len(idx)} tiles")
        return idx
    idx = {}
    tifs = glob.glob(os.path.join(SRC_ROOT, "**", "*.tif"), recursive=True)
    print(f"Scanning coordinates of {len(tifs)} source tiles...")
    for t in tifs:
        m = MON_RE.search(os.path.basename(t))
        if not m:
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        if y not in YEARS:
            continue
        with rasterio.open(t) as ds:
            w, s, e, n = latlon_bounds(ds)
        idx[t] = [y, mo, round(w, 4), round(s, 4), round(e, 4), round(n, 4)]
    json.dump(idx, open(INDEX_JSON, "w"))
    return idx


def region_ubox(boxes):
    W = min(b[2] for b in boxes); S = min(b[0] for b in boxes)
    E = max(b[3] for b in boxes); N = max(b[1] for b in boxes)
    return (W, S, E, N)


def overlaps(tb, box):
    w, s, e, n = tb
    W, S, E, N = box
    return (min(e, E) - max(w, W)) > 0 and (min(n, N) - max(s, S)) > 0


def main():
    idx = build_index()
    os.makedirs(OUT, exist_ok=True)
    summary = []
    manifest = []

    for region, boxes in TARGET_BOXES.items():
        ubox = region_ubox(boxes)
        seen = set()          # (year, month, bbox) deduplicate
        picked = []           # (path, sea, year, month)
        seas = {}
        for path, (y, mo, w, s, e, n) in idx.items():
            tb = (w, s, e, n)
            if not overlaps(tb, ubox):
                continue
            key = (y, mo, tb)
            if key in seen:
                continue
            seen.add(key)
            sea = os.path.basename(os.path.dirname(path))
            picked.append((path, sea, y, mo))
            seas[sea] = seas.get(sea, 0) + 1
        # extra standalone sources (whole tiles, no coordinate test), filename prefixed
        for folder, tag in EXTRA_SOURCES.get(region, []):
            for ep in sorted(glob.glob(os.path.join(folder, "*.tif"))):
                m = MON_RE.search(os.path.basename(ep))
                y, mo = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
                picked.append((ep, tag, y, mo))
                seas[tag] = seas.get(tag, 0) + 1

        months = sorted(set((y, mo) for _, _, y, mo in picked))
        summary.append((region, ubox, len(picked), len(months), seas))
        for path, sea, y, mo in picked:
            # prefix extra sources to avoid clashing with old data filenames
            fname = os.path.basename(path)
            if sea in {t for _, t in EXTRA_SOURCES.get(region, [])}:
                fname = f"{sea}__{fname}"
            manifest.append([region, y, mo, sea, fname, path])

    # Report
    print(f"\n{'region':13s} {'tiles':>6} {'months':>7}  source seas (after dedup)")
    for region, ubox, ntiles, nmonths, seas in summary:
        seatxt = ", ".join(f"{k}:{v}" for k, v in sorted(seas.items(), key=lambda x: -x[1]))
        print(f"{region:13s} {ntiles:>6} {nmonths:>7}  {seatxt}")

    if DRY_RUN:
        print("\n== DRY_RUN=True: above is the assignment plan, nothing copied. Set DRY_RUN to False and rerun once confirmed. ==")
        # still write a plan manifest for checking
        mp = os.path.join(BASE, "_6region_plan.csv")
        with open(mp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["region", "year", "month", "source_sea", "filename", "source_path"])
            w.writerows(manifest)
        print(f"Plan manifest: {mp} ({len(manifest)} rows)")
        return

    # Actual copy
    print("\nCopying...")
    n = 0
    for region, ubox, ntiles, nmonths, seas in summary:
        outdir = os.path.join(OUT, region)
        os.makedirs(outdir, exist_ok=True)
    for region, y, mo, sea, fname, src in manifest:
        dst = os.path.join(OUT, region, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            n += 1
    print(f"✅ Copied {n} tiles -> {OUT}")
    mp = os.path.join(OUT, "_manifest.csv")
    with open(mp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["region", "year", "month", "source_sea", "filename", "source_path"])
        w.writerows(manifest)
    print(f"Manifest: {mp}")


if __name__ == "__main__":
    main()
