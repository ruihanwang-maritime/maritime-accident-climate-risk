"""
prepare_0725_data.py
====================
Unzip the 11 annual zips in 0725_data, add the missing CRS (EPSG:4326), rename consistently,
and output monthly GeoTIFFs ready for GIS/analysis.

Raw data check findings:
  - Region: fixed bbox (W -73.221, S 34.957, E -51.145, N 45.339)  NW Atlantic, Grand Banks area
  - Time: 2011-2021, all 132 months, same extent every month, no shift
  - Resolution: 0.00904° (~1km), shape 1148x2441
  - Issue: raw tif CRS is missing (None); this script sets EPSG:4326 (metadata only, pixels not rewritten)

Output: 0725_data/ready/GMTDS_All_<YYYY>_<MM>.tif
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

SRC   = rf"{DATA_ROOT}/climate_AIS_data/GMTDS/0725_data"
READY = os.path.join(SRC, "ready")
EPSG  = 4326
MON_RE = re.compile(r"_(\d{4})_(\d{2})_", re.I)


def main():
    os.makedirs(READY, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(SRC, "*.zip")))
    print(f"Found {len(zips)} zips")
    done = 0
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            tifs = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            for n in tifs:
                m = MON_RE.search(n)
                if not m:
                    print(f"  Skipped (cannot parse year-month): {n}"); continue
                y, mo = m.group(1), m.group(2)
                dst = os.path.join(READY, f"GMTDS_All_{y}_{mo}.tif")
                # extract to target name
                with zf.open(n) as fsrc, open(dst, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)   # clear read-only
                # add CRS
                with rasterio.open(dst, "r+") as ds:
                    if ds.crs is None:
                        ds.crs = CRS.from_epsg(EPSG)
                done += 1
        print(f"  Done {os.path.basename(z)} ({len(tifs)} months)")
    print(f"\n✅ Processed {done} files -> {READY}")

    # Check
    outs = sorted(glob.glob(os.path.join(READY, "*.tif")))
    with rasterio.open(outs[0]) as ds:
        print(f"Check sample {os.path.basename(outs[0])}: CRS={ds.crs} bounds={tuple(round(v,3) for v in ds.bounds)} shape={(ds.height,ds.width)}")
    print(f"ready folder holds {len(outs)} monthly tifs")


if __name__ == "__main__":
    main()
