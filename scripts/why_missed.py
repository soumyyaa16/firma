import glob, numpy as np, pandas as pd
from scipy import ndimage as ndi
st3 = np.ones((3, 3, 3), int); rows = []
for f in sorted(glob.glob("demo_data/UMD_*.npz")):
    d = np.load(f); t2 = d["t2"].astype(float)
    gt, pr, sp = d["gt"].astype(int), d["pr"].astype(int), d["sp"]
    vox = float(np.prod(sp)); wall_med = np.median(t2[gt == 1]) if (gt == 1).any() else np.nan
    lab, n = ndi.label(gt == 3, structure=st3)
    for i in range(1, n + 1):
        m = lab == i; diam = (6 * m.sum() * vox / np.pi) ** (1 / 3)
        if diam < 10: continue
        p = pr[m]
        rows.append({"found": bool((p == 3).any()), "diam_mm": diam,
                     "t2_ratio": np.median(t2[m]) / wall_med,
                     "pred_wall": (p == 1).mean(), "pred_bg": (p == 0).mean(), "pred_cavity": (p == 2).mean()})
D = pd.DataFrame(rows)
print(D.groupby("found").agg(n=("diam_mm", "count"), size_mm=("diam_mm", "median"), t2_ratio=("t2_ratio", "median"),
      pred_as_wall=("pred_wall", "mean"), pred_as_background=("pred_bg", "mean"), pred_as_cavity=("pred_cavity", "mean")).round(2).to_string())