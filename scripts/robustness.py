import glob, numpy as np, pandas as pd
from scipy import ndimage as ndi

st3 = np.ones((3, 3, 3), int); rows = []
for f in sorted(glob.glob("demo_data/UMD_*.npz")):
    d = np.load(f); t2 = d["t2"].astype(float)
    gt, pr, sp = d["gt"].astype(int), d["pr"].astype(int), d["sp"]
    vox = float(np.prod(sp)); wall_med = np.median(t2[gt == 1]) if (gt == 1).any() else np.nan
    glab, gn = ndi.label(gt == 3, structure=st3); plab, pn = ndi.label(pr == 3, structure=st3)
    for i in range(1, gn + 1):
        m = glab == i; diam = (6 * m.sum() * vox / np.pi) ** (1 / 3); best = 0.0
        for j in np.unique(plab[m & (plab > 0)]):
            pm = plab == j; best = max(best, (m & pm).sum() / (m | pm).sum())
        rows.append({"diam": diam, "best_iou": best, "any": bool((m & (pr == 3)).any()),
                     "ratio": np.median(t2[m]) / wall_med})
D = pd.DataFrame(rows)

print("Recall by size cutoff and overlap rule")
for cut in [5, 10, 15, 20]:
    s = D[D.diam >= cut]
    print(f">= {cut:>2} mm [{len(s):>3}]: any {100*s['any'].mean():5.1f}% | IoU>=0.1 {100*(s.best_iou>=0.1).mean():5.1f}% | IoU>=0.3 {100*(s.best_iou>=0.3).mean():5.1f}% | IoU>=0.5 {100*(s.best_iou>=0.5).mean():5.1f}%")

B = D[D.diam >= 10].dropna(subset=["ratio"]).copy()
B["size"] = np.where(B.diam >= 15, "15mm+", "10-15mm")
B["contrast"] = np.where(B.ratio < 0.7, "dark", "bright")
T = B.groupby(["size", "contrast"]).agg(n=("any", "count"), missed=("any", lambda s: int((~s).sum())))
T["miss_%"] = (100 * T.missed / T.n).round(0)
print("\nMisses by size and contrast (lesions >= 10 mm)"); print(T.to_string())