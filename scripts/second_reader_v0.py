import glob, os, numpy as np
from scipy import ndimage as ndi

st3 = np.ones((3, 3, 3), int); st2 = np.ones((1, 3, 3), bool)
FRACS = [0.5, 0.6, 0.7, 0.8, 1.0]      # 1.0 = no darkness test (trivial baseline)
MIN_MM = 10
tot = {f: dict(rec=0, fp=0, cand=0) for f in FRACS}
n_total = n_found = n_missed = cases = 0

for fpath in sorted(glob.glob("demo_data/UMD_*.npz")):
    d = np.load(fpath); t2 = d["t2"].astype(float)
    gt, pr, sp = d["gt"].astype(int), d["pr"].astype(int), d["sp"]
    vox = float(np.prod(sp)); cases += 1
    glab, gn = ndi.label(gt == 3, structure=st3)
    big = []
    for i in range(1, gn + 1):
        m = glab == i
        if (6 * m.sum() * vox / np.pi) ** (1 / 3) >= MIN_MM:
            big.append(m)
    n_total += len(big)
    missed = [m for m in big if not (m & (pr == 3)).any()]
    n_found += len(big) - len(missed); n_missed += len(missed)
    wall = pr == 1
    if not wall.any():
        continue
    med = np.median(t2[wall])
    for f in FRACS:
        dark = wall & (t2 < f * med) & (pr != 3)
        dark = ndi.binary_opening(dark, structure=st2)
        lab, n = ndi.label(dark, structure=st3)
        cands = []
        for j in range(1, n + 1):
            m = lab == j
            if (6 * m.sum() * vox / np.pi) ** (1 / 3) >= MIN_MM:
                cands.append(m)
        tot[f]["cand"] += len(cands)
        tot[f]["fp"] += sum(1 for m in cands if not (m & (gt == 3)).any())
        tot[f]["rec"] += sum(1 for g in missed if any((g & m).any() for m in cands))

print(f"cases {cases} | expert lesions >= {MIN_MM} mm: {n_total} | found by AI: {n_found} | missed by AI: {n_missed}\n")
print("dark-threshold | missed lesions recovered | flags/patient | false flags/patient | combined recall")
for f in FRACS:
    t = tot[f]
    print(f"   {f:>4}        |   {t['rec']:>2}/{n_missed}                 |    {t['cand']/cases:5.2f}      |       {t['fp']/cases:5.2f}         |   {100*(n_found+t['rec'])/n_total:.1f}%")