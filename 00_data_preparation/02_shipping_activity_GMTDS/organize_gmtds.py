"""
organize_gmtds.py
=================
Reclassify the monthly GeoTIFFs in the extracted / east_asia_CMTDS folders under <DATA_ROOT>/climate_AIS_data/GMTDS,
which are scattered and whose folder names do not match their true coordinates, by [each tif's true georeference],
into a clean new folder ORGANIZED, one subfolder per true sea area.

Rules (defined by CATALOG below, based only on true TIFF bounds, not file names):
  - mode="copy" : single consistent extent, full 132 months 2011-2021 -> copy monthly tifs as is
  - mode="crop" : 5 seas whose extent shifts mid-series -> crop to common overlap bbox for a consistent grid
  - mode="skip" : incomplete time span (2011 only) -> not organized, only flagged in report, needs re-download

Deduplicate: if a sea appears in both extracted and east_asia_CMTDS, keep PREFER_SOURCE.

Run with DRY_RUN=True first to check report and plan; set False to execute once confirmed.
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README

import os, re, glob, shutil, csv, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # print non-ASCII text correctly on Windows, even when redirected
except Exception:
    pass
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

# ----------------------- Config -----------------------
BASE          = rf"{DATA_ROOT}/climate_AIS_data/GMTDS"
SRC_FOLDERS   = ["extracted", "east_asia_CMTDS"]     # scan these two parent folders
ORGANIZED     = rf"{DATA_ROOT}/climate_AIS_data/GMTDS/ORGANIZED"   # new clean folder
PREFER_SOURCE = "extracted"     # parent folder preferred on duplicates
YEARS         = range(2011, 2022)   # required time span 2011-2021
DRY_RUN       = True            # True=print plan only, no file changes; set False once confirmed
MATCH_TOL     = 1.0             # coordinate tolerance (deg) for assigning a tif to an extent
# ----------------------------------------------------

# catalog of 23 canonical sea areas clustered from true coordinates (bbox=W,S,E,N, EPSG:4326)
CATALOG = [
  ('NE_Pacific_Canada', (-153.1, 45.5, -121.8, 55.1), 'copy', [(-153.1, 45.5, -121.8, 55.1)], 'single consistent extent, 132/132'),
  ('US_West_Coast_Pacific', (-147.8, 36.8, -122.4, 52.9), 'copy', [(-147.8, 36.8, -122.4, 52.9)], 'single consistent extent, 132/132'),
  ('Gulf_of_Mexico', (-97.1, 14.0, -74.1, 29.5), 'copy', [(-97.1, 14.0, -74.1, 29.5)], 'single consistent extent, 132/132'),
  ('Caribbean_Sea', (-87.8, 8.2, -59.1, 21.9), 'copy', [(-87.8, 8.2, -59.1, 21.9)], 'single consistent extent, 132/132'),
  ('US_East_Coast', (-82.2, 24.2, -70.1, 42.0), 'copy', [(-82.2, 24.2, -70.1, 42.0)], 'single consistent extent, 132/132'),
  ('NW_Atlantic_GrandBanks', (-67.4, 36.7, -50.3, 51.8), 'copy', [(-67.4, 36.7, -50.3, 51.8)], 'single consistent extent, 132/132'),
  ('N_Atlantic_Central', (-56.1, 48.1, -10.7, 56.4), 'copy', [(-56.1, 48.1, -10.7, 56.4)], 'single consistent extent, 132/132'),
  ('Norwegian_Sea', (-30.3, 60.8, 9.8, 68.0), 'copy', [(-30.3, 60.8, 9.8, 68.0)], 'single consistent extent, 132/132'),
  ('NW_Europe_North_Sea', (-11.1, 46.3, 12.0, 62.8), 'copy', [(-11.1, 46.3, 12.0, 62.8)], 'single consistent extent, 132/132'),
  ('Mediterranean_E_Central', (11.8, 30.1, 36.4, 45.7), 'copy', [(11.8, 30.1, 36.4, 45.7)], 'single consistent extent, 132/132'),
  ('W_Africa_Gulf_of_Guinea', (-2.6, -7.5, 13.7, 6.7), 'copy', [(-2.6, -7.5, 13.7, 6.7)], 'single consistent extent, 132/132'),
  ('Black_Sea_Caspian_approx', (24.8, 36.7, 54.4, 47.9), 'copy', [(24.8, 36.7, 54.4, 47.9)], 'single consistent extent, 132/132'),
  ('Arabian_Sea_N_Indian_Ocean', (43.1, 4.9, 82.6, 13.8), 'copy', [(43.1, 4.9, 82.6, 13.8)], 'single consistent extent, 132/132'),
  ('Red_Sea_Persian_Gulf', (46.5, 16.5, 73.3, 30.5), 'copy', [(46.5, 16.5, 73.3, 30.5)], 'single consistent extent, 132/132'),
  ('Andaman_E_Indian_Ocean', (77.8, 5.2, 99.2, 22.6), 'copy', [(77.8, 5.2, 99.2, 22.6)], 'single consistent extent, 132/132'),
  ('Philippines_Sulawesi', (119.3, -6.1, 131.4, 18.5), 'copy', [(119.3, -6.1, 131.4, 18.5)], 'single consistent extent, 132/132'),
  ('Japan_Korea_Pacific', (126.0, 28.0, 146.0, 46.7), 'copy', [(126.0, 28.0, 146.0, 46.7)], 'single consistent extent, 132/132'),
  ('Iberian_W_Mediterranean', (-10.7, 35.0, 8.7, 47.2), 'crop', [(-10.7, 35.0, 21.4, 47.2), (-11.3, 34.2, 8.7, 47.5)], 'extent shifted mid-series; crop to overlap'),
  ('Baltic_Scandinavia', (12.0, 53.7, 28.0, 65.5), 'crop', [(10.0, 53.7, 28.2, 65.5), (12.0, 53.6, 28.0, 65.9)], 'extent shifted mid-series; crop to overlap'),
  ('Sunda_Java_Sea', (94.7, -8.8, 118.2, 6.1), 'crop', [(94.6, -8.8, 118.2, 6.1), (94.7, -9.4, 118.6, 6.3)], 'extent shifted mid-series; crop to overlap'),
  ('South_China_Sea', (104.1, 5.6, 122.3, 21.8), 'crop', [(104.1, 5.6, 127.1, 21.8), (102.5, 5.5, 122.3, 22.0)], 'extent shifted mid-series; crop to overlap'),
  ('East_China_Sea', (116.4, 21.9, 132.0, 41.2), 'crop', [(116.4, 20.1, 134.7, 41.5), (114.9, 21.9, 132.0, 41.2)], 'extent shifted mid-series; crop to overlap'),
  ('NW_Atlantic_ShelfBreak_2011ONLY', (-79.9, 29.4, -54.4, 45.0), 'skip', [(-79.9, 29.4, -54.4, 45.0)], 'ONLY 2011 -> incomplete, re-download 2012-2021'),
]

MON_RE = re.compile(r"_(\d{4})_(\d{2})\.tif$", re.I)


def latlon_bounds(ds):
    b = ds.bounds
    try:
        return transform_bounds(ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
    except Exception:
        return (b.left, b.bottom, b.right, b.top)


def match_region(bounds):
    """Assign a tif's true bounds to the closest sea area in CATALOG; return None if no match."""
    best, best_d = None, 1e9
    for name, target, mode, sources, note in CATALOG:
        for sb in sources:
            d = max(abs(bounds[i] - sb[i]) for i in range(4))
            if d < best_d:
                best_d, best = d, (name, target, mode, note)
    return best if best_d <= MATCH_TOL else None


def list_all_tifs():
    out = []
    for fol in SRC_FOLDERS:
        root = os.path.join(BASE, fol)
        if not os.path.isdir(root):
            continue
        for region in sorted(os.listdir(root)):
            rp = os.path.join(root, region)
            if os.path.isdir(rp):
                for t in sorted(glob.glob(os.path.join(rp, "*.tif"))):
                    out.append((fol, region, t))
    return out


def crop_to_bbox(src_path, dst_path, target):
    """Crop src to target bbox (W,S,E,N, lon/lat) and write a new GeoTIFF."""
    with rasterio.open(src_path) as ds:
        # target is lon/lat; if raster CRS is not 4326, transform to raster CRS
        if ds.crs and ds.crs.to_epsg() != 4326:
            w, s, e, n = transform_bounds("EPSG:4326", ds.crs, *target)
        else:
            w, s, e, n = target
        win = from_bounds(w, s, e, n, transform=ds.transform).round_offsets().round_lengths()
        # clip to raster extent
        win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
        data = ds.read(window=win)
        prof = ds.profile.copy()
        prof.update(height=int(win.height), width=int(win.width),
                    transform=ds.window_transform(win))
    with rasterio.open(dst_path, "w", **prof) as dst:
        dst.write(data)


def main():
    tifs = list_all_tifs()
    print(f"Found {len(tifs)} tifs, classifying by true coordinates...\n")

    # plan[region][ (year,month) ] = (chosen_src_path, mode, target, source_folder)
    plan = {}
    unmatched = []
    for fol, region, t in tifs:
        m = MON_RE.search(os.path.basename(t))
        if not m:
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        if y not in YEARS:
            continue
        with rasterio.open(t) as ds:
            b = latlon_bounds(ds)
        hit = match_region(b)
        if hit is None:
            unmatched.append(t)
            continue
        name, target, mode, note = hit
        d = plan.setdefault(name, {"mode": mode, "target": target, "note": note, "months": {}})
        key = (y, mo)
        # deduplicate: prefer PREFER_SOURCE
        if key not in d["months"] or (fol == PREFER_SOURCE and d["months"][key][1] != PREFER_SOURCE):
            d["months"][key] = (t, fol)

    # ---- Report ----
    print(f"{'canonical region':32s} {'mode':5s} {'months':>7}  status")
    manifest = []
    for name, _, mode, _, _ in CATALOG:
        d = plan.get(name)
        n = len(d["months"]) if d else 0
        status = "OK 132/132" if n == 132 else (f"!! {n}/132 (incomplete)" if mode != "skip" else f"SKIP ({n} mo, 2011 only)")
        print(f"{name:32s} {mode:5s} {n:>7}  {status}")
        if d:
            for (y, mo), (src, fol) in sorted(d["months"].items()):
                manifest.append((name, mode, y, mo, fol, src))
    if unmatched:
        print(f"\n⚠ {len(unmatched)} tifs could not be classified (coordinates beyond tolerance), e.g.: {unmatched[:3]}")

    # write manifest CSV (always, dry-run or not, for checking)
    os.makedirs(ORGANIZED, exist_ok=True)
    man_csv = os.path.join(ORGANIZED, "_manifest.csv")
    with open(man_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["region", "mode", "year", "month", "source_folder", "source_path"])
        w.writerows(manifest)
    print(f"\nManifest written: {man_csv}  ({len(manifest)} rows)")

    if DRY_RUN:
        print("\n== DRY_RUN=True: above is the plan, no files copied/cropped. Set DRY_RUN to False and rerun once confirmed. ==")
        return

    # ---- Execute ----
    print("\nOrganizing files...")
    for name, d in plan.items():
        if d["mode"] == "skip":
            continue
        outdir = os.path.join(ORGANIZED, name)
        os.makedirs(outdir, exist_ok=True)
        for (y, mo), (src, fol) in sorted(d["months"].items()):
            dst = os.path.join(outdir, f"GMTDS_{name}_{y}_{mo:02d}.tif")
            if d["mode"] == "copy":
                shutil.copy2(src, dst)
            else:  # crop
                crop_to_bbox(src, dst, d["target"])
        print(f"  Done {name}: {len(d['months'])} months")
    print(f"\n✅ Organizing done, output folder: {ORGANIZED}")


if __name__ == "__main__":
    main()
