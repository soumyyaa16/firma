# firma
# FIRMA: Fibroid Imaging, Reliability & Mapping Analysis

Research prototype for reliability-aware AI fibroid mapping on pelvic MRI.
AI segmentation can look accurate while silently missing small or cavity-adjacent
fibroids. FIRMA measures how complete the AI-generated fibroid map is, per lesion,
and works toward flagging likely omissions for human review.

**Research use only. Not a medical device. Not validated for clinical decisions.**

## Preliminary results (UMD, nnU-Net 2D, 50 epochs, fold 0, 60 held-out patients)
| Metric | Value |
|---|---|
| Myoma Dice | 0.55 |
| Lesion recall, ≥10 mm | 81% (86/106) |
| Lesion recall, <10 mm | 20% (18/89) |
| Cases with ≥1 missed ≥10 mm lesion | 18 / 52 |

Preliminary: single fold, short training, placeholder thresholds.

## Pipeline
MRI → segmentation → lesion extraction → per-lesion table → review flags (in progress)

## Roadmap
- [x] UMD audit (300 cases) and nnU-Net conversion
- [x] Baseline training and lesion-level evaluation
- [ ] Viewer and per-lesion table
- [ ] Second-reader candidate detector
- [ ] Cross-validation on all 5 folds

## Data
UMD (Pan et al., Scientific Data 2024), obtained from Figshare. Not redistributed here.

## Limitations
Public sagittal T2 data only. No outcome data. Lesions defined as connected components.
