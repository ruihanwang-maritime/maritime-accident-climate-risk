"""
Process the Cargo Ships data in 0719_daata:
- 5 regions (told apart by suffix, 11 years = 132 TIFs each)
- unzip and rename -> matching subfolder of extracted/
- pack into per-region zips -> 0719_compressed/
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
import os, re, zipfile

SRC_DIR      = rf"{DATA_ROOT}/climate_AIS_data/GMTDS/0719_daata"
EXTRACT_DIR  = rf"{DATA_ROOT}/climate_AIS_data/GMTDS/extracted"
COMPRESS_DIR = rf"{DATA_ROOT}/climate_AIS_data/GMTDS/0719_compressed"
os.makedirs(COMPRESS_DIR, exist_ok=True)

# suffix -> region name mapping (no suffix = "")
REGION_MAP = {
    "":   "US_W_Coast_Pacific",
    "(1)": "Baltic_Scandinavia",
    "(2)": "Norwegian_Sea_NE_Atlantic",
    "(3)": "Indian_Ocean_E_Bay_Bengal",
    "(4)": "N_Atlantic_Central",
}

def get_suffix(fname):
    """Extract the Windows download suffix from the filename, e.g. (1) / (2) / '' """
    m = re.search(r'Cargo Ships(\s*\(\d+\))?\.zip$', fname)
    if not m: return None
    suf = m.group(1)
    return suf.strip() if suf else ""

def parse_year(fname):
    m = re.search(r'_(\d{4})_01', fname)
    return int(m.group(1)) if m else None

def parse_month_from_tif(tif_name):
    m = re.search(r'_(\d{4})_(\d{2})_', tif_name)
    if m: return int(m.group(1)), int(m.group(2))
    m2 = re.search(r'(\d{4})(\d{2})\.tif$', tif_name, re.I)
    if m2: return int(m2.group(1)), int(m2.group(2))
    return None, None

# ── Group by region ────────────────────────────────────────────────────
groups = {suf: [] for suf in REGION_MAP}

for fname in sorted(os.listdir(SRC_DIR)):
    if not fname.endswith(".zip"): continue
    if "Cargo Ships" not in fname: continue
    suf = get_suffix(fname)
    if suf is None: continue
    yr = parse_year(fname)
    if yr is None: continue
    groups[suf].append((yr, fname))

print("Files grouped by region:")
for suf, files in groups.items():
    yrs = sorted(y for y,_ in files)
    print(f"  {REGION_MAP[suf]:<35}  {len(files)} files  years={yrs[0]}-{yrs[-1] if yrs else '?'}")
print()

# ── Unzip + pack per region ─────────────────────────────────────────────
for suf, files in groups.items():
    region = REGION_MAP[suf]
    region_dir = os.path.join(EXTRACT_DIR, region)
    os.makedirs(region_dir, exist_ok=True)
    out_zip = os.path.join(COMPRESS_DIR, f"{region}.zip")

    tif_count = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for yr, fname in sorted(files):
            zip_path = os.path.join(SRC_DIR, fname)
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    tifs = sorted([n for n in zf.namelist() if n.lower().endswith(".tif")])
                    for tif_name in tifs:
                        fyr, mo = parse_month_from_tif(tif_name)
                        if fyr is None:
                            fyr, mo = yr, tifs.index(tif_name) + 1
                        new_name = f"GMTDS_{region}_{fyr:04d}_{mo:02d}.tif"
                        dest = os.path.join(region_dir, new_name)
                        data = zf.read(tif_name)
                        with open(dest, "wb") as f:
                            f.write(data)
                        zout.write(dest, new_name)
                        tif_count += 1
            except Exception as e:
                print(f"  ERROR {fname}: {e}")

    print(f"  {region:<35}  {tif_count:3d} TIF  -> {os.path.basename(out_zip)}")

print(f"\nAll done! Zips saved in: {COMPRESS_DIR}")
print(f"TIF files extracted to: {EXTRACT_DIR}/<region>/")
