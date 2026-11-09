"""
Regroup, deduplicate and unzip the Cargo Ships data in cargo_ship/1-10 into 6 region folders by true bbox.
- each tif goes to every target region its true coordinates overlap (may be several); others go to _other
- deduplicate: one copy per (bbox, year-month)
- add EPSG:4326; resumable (skip if dst exists)
Output: cargo_ship/organized/<region>/Cargo_b{idx}_{YYYY}_{MM}.tif  + _manifest.csv
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
import os, re, glob, zipfile, json, csv, stat, sys, time
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box as sbox

BASE=rf"{DATA_ROOT}/climate_AIS_data/GMTDS/cargo_ship"
OUT=os.path.join(BASE,"organized")
FOLDERS=[str(i) for i in range(1,11)]
MON=re.compile(r"_(\d{4})_(\d{2})_",re.I)
TB={"NW_Europe":[(43,63,-15,0),(48,63,0,10)],"NW_Atlantic":[(18,50,-98,-60)],
 "SE_Asia":[(-10,20,95,127)],"E_Asia":[(20,42,117,145)],
 "Med_Black":[(30,43,-5.5,0),(30,47.5,0,42)],"Baltic":[(53,66,10,30)]}
def regions_of(bx):
    b=sbox(*bx); out=[]
    for r,bs in TB.items():
        from shapely.ops import unary_union
        p=unary_union([sbox(w,s,e,n) for(s,n,w,e) in bs])
        if b.intersection(p).area>0: out.append(r)
    return out or ["_other"]

def flush(m): print(m,flush=True)

# ---- Pass 1: index + deduplicate (bbox,ym)->(zip,inner) ----
flush("Building index...")
t0=time.time(); seen={}; boxes={}
for fold in FOLDERS:
    for z in sorted(glob.glob(os.path.join(BASE,fold,"*.zip"))):
        try:
            with zipfile.ZipFile(z) as zf:
                tifs=[x for x in zf.namelist() if x.lower().endswith(".tif")]
        except: continue
        for t in tifs:
            m=MON.search(t)
            if not m: continue
            ym=f"{int(m.group(1))}_{int(m.group(2)):02d}"
            try:
                with rasterio.open(f"/vsizip/{z}/{t}") as ds:
                    b=ds.bounds; bx=(round(b.left,2),round(b.bottom,2),round(b.right,2),round(b.top,2))
            except: continue
            key=(bx,ym)
            if key in seen: continue
            seen[key]=(z,t); boxes.setdefault(bx,set()).add(ym)
flush(f"Index done {time.time()-t0:.0f}s | unique (bbox,month)={len(seen)} | unique bbox={len(boxes)}")

# assign each bbox an id + regions
bxlist=sorted(boxes.keys())
bxid={bx:i+1 for i,bx in enumerate(bxlist)}
bxreg={bx:regions_of(bx) for bx in bxlist}
os.makedirs(OUT,exist_ok=True)
man=[]
for bx in bxlist:
    man.append([bxid[bx],*bx,len(boxes[bx]),";".join(bxreg[bx])])
with open(os.path.join(OUT,"_manifest.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["box_id","W","S","E","N","n_months","regions"]); w.writerows(man)
flush("Wrote _manifest.csv")

# ---- Pass 2: unzip + add CRS + distribute to regions ----
for r in list(TB)+["_other"]: os.makedirs(os.path.join(OUT,r),exist_ok=True)
n=0; total=len(seen); t1=time.time()
for (bx,ym),(z,inner) in seen.items():
    regs=bxreg[bx]; bid=bxid[bx]; fname=f"Cargo_b{bid:02d}_{ym}.tif"
    dsts=[os.path.join(OUT,r,fname) for r in regs]
    if all(os.path.exists(dp) for dp in dsts):
        n+=1; continue
    # extract to the first target
    import shutil
    first=dsts[0]
    if not os.path.exists(first):
        with zipfile.ZipFile(z) as zf, zf.open(inner) as fs, open(first,"wb") as fo:
            shutil.copyfileobj(fs,fo)
        os.chmod(first,stat.S_IWRITE|stat.S_IREAD)
        with rasterio.open(first,"r+") as ds:
            if ds.crs is None: ds.crs=CRS.from_epsg(4326)
    for dp in dsts[1:]:
        if not os.path.exists(dp): shutil.copy2(first,dp)
    n+=1
    if n%300==0: flush(f"[{n}/{total}] {time.time()-t1:.0f}s")
flush(f"\n✅ Done, extracted {total} unique (bbox,month); files per region:")
for r in list(TB)+["_other"]:
    c=len(glob.glob(os.path.join(OUT,r,"*.tif"))); flush(f"   {r:12s} {c}")
