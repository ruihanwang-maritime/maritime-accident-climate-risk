"""
add_072555_to_regions.py
========================
Merge the two new downloads in 072555_data (fine resolution 0.009°, CRS missing) into the matching 6_region folders.
- overlap each zip's true bbox with the 6 target regions; assign to the region with largest overlap
- add EPSG:4326 (raw CRS=None)
- copy whole, no cropping; filename prefix marks [vessel type + resolution + source] to tell apart from old data

Prefix example: 072555_Cargo_res0009__grid_float_Cargo Ships_2015_06_converted.tif
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
import os, re, glob, zipfile, shutil, stat, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import rasterio
from rasterio.crs import CRS

SRC   = rf"{DATA_ROOT}/climate_AIS_data/GMTDS/072555_data"
REG   = rf"{DATA_ROOT}/climate_AIS_data/GMTDS/6_region"
EPSG  = 4326
MON   = re.compile(r"_(\d{4})_(\d{2})_", re.I)

TARGET_BOXES = {  # (south, north, west, east)
    "NW_Europe":   [(43, 63, -15, 0), (48, 63, 0, 10)],
    "NW_Atlantic": [(18, 50, -98, -60)],
    "SE_Asia":     [(-10, 20, 95, 127)],
    "E_Asia":      [(20, 42, 117, 145)],
    "Med_Black":   [(30, 43, -5.5, 0), (30, 47.5, 0, 42)],
    "Baltic":      [(53, 66, 10, 30)],
}

def ov_area(bx, boxes):  # bx=(W,S,E,N); boxes list of (S,N,W,E)
    W, S, E, N = bx; tot = 0.0
    for s, n, w, e in boxes:
        iw, ie = max(W, w), min(E, e); is_, in_ = max(S, s), min(N, n)
        if ie > iw and in_ > is_: tot += (ie - iw) * (in_ - is_)
    return tot

def best_region(bx):
    best, ba = None, 0.0
    for r, boxes in TARGET_BOXES.items():
        a = ov_area(bx, boxes)
        if a > ba: ba, best = a, r
    return best

def main():
    zips = sorted(glob.glob(os.path.join(SRC, "*.zip")))
    print(f"{len(zips)} zips")
    total = 0
    for z in zips:
        base = os.path.basename(z)
        kind = "Cargo" if "Cargo" in base else "All"
        with zipfile.ZipFile(z) as zf:
            tifs = sorted(n for n in zf.namelist() if n.lower().endswith(".tif"))
            # use the first tif to pick region
            with zf.open(tifs[0]) as f, open(os.path.join(SRC, "_probe.tif"), "wb") as o:
                shutil.copyfileobj(f, o)
        pt = os.path.join(SRC, "_probe.tif")
        with rasterio.open(pt) as ds:
            b = ds.bounds; bx = (b.left, b.bottom, b.right, b.top)
        os.remove(pt)
        region = best_region(bx)
        outdir = os.path.join(REG, region)
        os.makedirs(outdir, exist_ok=True)
        prefix = f"072555_{kind}_res0009__"
        with zipfile.ZipFile(z) as zf:
            for n in tifs:
                dst = os.path.join(outdir, prefix + os.path.basename(n))
                with zf.open(n) as f, open(dst, "wb") as o:
                    shutil.copyfileobj(f, o)
                os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)
                with rasterio.open(dst, "r+") as ds:
                    if ds.crs is None:
                        ds.crs = CRS.from_epsg(EPSG)
                total += 1
        print(f"  {base:46s} [{kind:5s}] bbox={tuple(round(v,1) for v in bx)} -> {region} ({len(tifs)} files)")
    print(f"\n✅ Merged {total} files")
    for r in TARGET_BOXES:
        c = len(glob.glob(os.path.join(REG, r, "072555_*.tif")))
        if c: print(f"   {r}: {c} new 072555 files")

if __name__ == "__main__":
    main()
