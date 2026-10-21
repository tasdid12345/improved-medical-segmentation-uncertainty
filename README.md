# Class-balanced LightGBM vs k-means for synthetic organ-phantom segmentation

**Motivation.** Segmenting organs in medical slices is often done with simple intensity clustering, which struggles when tissue intensities overlap and images have smooth intensity inhomogeneity. The motivating paper, a 2020 work from the University of Salamanca Department of Anatomy available at https://repositorio.grial.eu/bitstreams/e38b76e5-23b8-4f7f-af28-fe73b9db868b/download, is cited here as the source of this line of inquiry; its exact title could not be confirmed from the extracted text.

**Question.** Can a class-balanced LightGBM pixel classifier with multi-scale neighbourhood features segment synthetic medical-style slices more accurately than a standard per-pixel k-means baseline, while also flagging low-confidence regions?

**Dataset.** Seeded synthetic 128x128 grayscale "organ phantom" slices generated in numpy, with smooth intensity inhomogeneity and Gaussian noise. No download is used. This is not the paper's own imaging data. Slices are split by image into train and held-out test sets with fixed seeds.

**Baseline vs improved.** The baseline is per-pixel k-means on raw intensity, with clusters mapped to labels by majority vote on training slices. The improved model is a class-balanced LightGBM on multi-scale local features (intensity, local means and standard deviations at several window sizes, gradients). Pixels with low predicted probability are flagged as uncertain. The main metric is mean Dice over organ labels on test slices.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts in `figures/`).

## Results (latest run)

On synthetic organ-phantom slices (20 held-out test slices), mean Dice over organ labels was 0.451 for a standard per-pixel k-means intensity segmentation and 0.853 for a class-balanced LightGBM with multi-scale features, a large gain on synthetic data only.

| Metric | Value |
|---|---|
| baseline_score | 0.451 |
| improved_score | 0.8532 |
| improvement | 0.4022 |
| baseline_score_std | 0.01885 |
| improved_score_std | 0.0004623 |
| baseline_accuracy | 0.9183 |
| improved_accuracy | 0.9783 |
| baseline_boundary_error | 0.2709 |
| improved_boundary_error | 0.06665 |
| baseline_f1 | 0.589 |
| improved_f1 | 0.9129 |
| improved_flagged_fraction | 0.008682 |
| improved_accuracy_confident | 0.9823 |
| selected_num_leaves | 63 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Synthetic data only, not the paper's imaging data; the motivating paper's exact title could not be confirmed from the extracted text. Dice is averaged over slices and organ labels. Boundary error = error rate within 2 px of ground-truth edges. Baseline has no confidence output. The 0.6 confidence threshold was fixed in advance, not tuned. Only num_leaves was chosen by training-set CV. Std is over 3 model seeds with a fixed data split; it does not capture data-sampling variability.

## Reference

This project is motivated by the following paper:

> SG Izard, RS Torres, ÓA Plaza (2020). *[TXT] … Anatomy, University of Salamanca, 37008 Salamanca, Spain; ujhik@ usal. es (RST); u157938@ usal. es (Ó. AP); jajm@ usal. es (JAJM); fgarcia@ usal. es  …*. Sensors. https://repositorio.grial.eu/bitstreams/e38b76e5-23b8-4f7f-af28-fe73b9db868b/download

Paper author: Prof. Francisco José García-Peñalvo, University of Salamanca (fgarcia@usal.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
