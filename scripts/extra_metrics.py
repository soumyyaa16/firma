import glob, os, numpy as np, pandas as pd
from scipy import ndimage as ndi

st3 = np.ones((3, 3, 3), int)
dice = {1: [], 2: [], 3: []}
rows, cases = [], 0
for f in sorted(glob.glob("demo_data/UMD_*.npz")):
    d = np.load(f); gt, pr, sp = d["gt"].astype(int), d["pr"].astype(int), d["sp"]
    cid = os.path.basename(f)[:-4]; vox = float(np.prod(sp)); cases += 1
    for c in dice:
        a, b = gt == c, pr == c
        if a.sum() + b.sum() > 0:
            dice[c].append(2 * (a & b).sum() / (a.sum() + b.sum()))
    plab, pn = ndi.label(pr == 3, structure=st3)
    for j in range(1, pn + 1):
        m = plab == j; v = m.sum() * vox
        rows.append({"case": cid, "diam_mm": (6 * v / np.pi) ** (1 / 3), "tp": bool((m & (gt == 3)).any())})

print("cases:", cases)
names = {1: "wall", 2: "cavity", 3: "fibroid"}
print("\nPer-patient Dice (mean +/- SD, min):")
for c, v in dice.items():
    print(f"  {names[c]}: {np.mean(v):.3f} +/- {np.std(v):.3f}  (min {np.min(v):.3f}, n={len(v)})")

P = pd.DataFrame(rows)
print("\nLesion-level precision / false positives (predicted fibroid components):")
for thr in (0, 5, 10):
    s = P[P.diam_mm >= thr]
    print(f"  predicted >= {thr:>2} mm: {len(s)} components, precision {s.tp.mean():.3f}, false positives per patient {(~s.tp).sum() / cases:.2f}")

L = pd.read_csv("results/lesions.csv")
big = L[L.diam_mm >= 10]
g = big.groupby("case").det_any.agg(["sum", "count"]).to_numpy()
rng = np.random.default_rng(0)
boots = [g[idx, 0].sum() / g[idx, 1].sum() for idx in (rng.integers(0, len(g), len(g)) for _ in range(5000))]
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"\nRecall for lesions >=10 mm: {big.det_any.mean() * 100:.1f}% (95% CI {lo * 100:.1f}-{hi * 100:.1f}%, resampling patients, n={len(g)} patients)")