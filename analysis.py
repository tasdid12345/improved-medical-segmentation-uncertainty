"""Synthetic organ-phantom slices: per-pixel k-means baseline vs class-balanced LightGBM with multi-scale features."""
import json
import os
import sys
import time

import lightgbm as lgb
import matplotlib
import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
SEEDS = [42, 43, 44]          # both models are randomized -> run each with these seeds
N_SLICES, TEST_FRAC = 60, 0.33  # split by image, seeded permutation
SIZE, N_CLASSES = 128, 4      # background + 3 organ labels
MAX_TRAIN_PIXELS = 200_000    # subsample cap for LightGBM (time budget, shared logic)
KMEANS_PIXELS = 50_000        # subsample cap for k-means fitting
CONF_THRESH = 0.6             # fixed a priori; pixels with max-probability below it are flagged
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def make_slice(rng):
    """One synthetic phantom: 3 random ellipses on background, smooth multiplicative
    inhomogeneity, Gaussian noise. Intensity levels overlap, so raw-intensity clustering is hard."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    lab = np.zeros((SIZE, SIZE), dtype=np.int64)
    for k in range(1, N_CLASSES):
        cx, cy = rng.uniform(0.3, 0.7, 2)
        ax, ay = rng.uniform(0.10, 0.22, 2)
        lab[((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 < 1] = k
    means = np.array([0.15, 0.40, 0.55, 0.70])
    fx, fy = rng.uniform(0.5, 1.5, 2)
    px, py = rng.uniform(0, 2 * np.pi, 2)
    field = 1 + 0.2 * np.sin(2 * np.pi * fx * xx + px) * np.cos(2 * np.pi * fy * yy + py)
    img = means[lab] * field + rng.normal(0, 0.07, lab.shape)
    return img, lab


def features(img):
    """Per-pixel features. Column 0 is raw intensity (the only input of the baseline)."""
    f = [img]
    for w in (3, 7, 15):  # local mean and std at several window sizes
        m = ndi.uniform_filter(img, w)
        m2 = ndi.uniform_filter(img * img, w)
        f += [m, np.sqrt(np.maximum(m2 - m * m, 0))]
    g = ndi.gaussian_filter(img, 1.0)
    f.append(np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1)))  # gradient magnitude
    f.append(ndi.gaussian_gradient_magnitude(img, 2.0))  # multi-scale gradient
    return np.stack([x.ravel() for x in f], axis=1)


def evaluate(pred, gt):
    """pred, gt: (n, H, W) label maps. Returns mean organ Dice (main) and secondary metrics."""
    dices = []
    for p, t in zip(pred, gt):
        for k in range(1, N_CLASSES):  # organ labels only
            pm, tm = p == k, t == k
            dices.append((2 * np.logical_and(pm, tm).sum() + 1e-9) / (pm.sum() + tm.sum() + 1e-9))
    # boundary band: pixels within 2 px of a ground-truth label edge
    band = np.zeros(gt.shape, dtype=bool)
    for i, t in enumerate(gt):
        edge = np.zeros(t.shape, dtype=bool)
        edge[:, 1:] |= t[:, 1:] != t[:, :-1]
        edge[1:, :] |= t[1:, :] != t[:-1, :]
        band[i] = ndi.binary_dilation(edge, iterations=2)
    return {"dice": float(np.mean(dices)),
            "accuracy": float((pred == gt).mean()),
            "boundary_error": float((pred != gt)[band].mean()),
            "f1": float(f1_score(gt.ravel(), pred.ravel(), average="macro"))}


def subsample(idx, seed, n):
    return np.random.default_rng(seed).choice(idx, min(n, len(idx)), replace=False)


def make_lgb(leaves, seed):
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=leaves,
                              class_weight="balanced", random_state=seed, n_jobs=2, verbose=-1)


def ms(v):
    """mean and sample std over seeds."""
    return float(np.mean(v)), float(np.std(v, ddof=1))


def main():
    rng = np.random.default_rng(SEED)
    slices = [make_slice(rng) for _ in range(N_SLICES)]
    imgs = np.stack([s[0] for s in slices])
    labs = np.stack([s[1] for s in slices])
    perm = np.random.default_rng(SEED).permutation(N_SLICES)
    n_test = int(round(TEST_FRAC * N_SLICES))
    te, tr = perm[:n_test], perm[n_test:]
    log(f"generated {N_SLICES} slices ({SIZE}x{SIZE}); split by image: {len(tr)} train / {len(te)} test")

    F = np.stack([features(im) for im in imgs])
    D, P = F.shape[-1], SIZE * SIZE
    X_tr, y_tr, g_tr = F[tr].reshape(-1, D), labs[tr].ravel(), np.repeat(tr, P)
    X_te, gt_te = F[te].reshape(-1, D), labs[te]

    # ---- small CV on TRAINING slices only, to pick LightGBM num_leaves ----
    cv = {}
    for leaves in (15, 63):
        sc = []
        for a, b in GroupKFold(3).split(X_tr, y_tr, g_tr):
            a = subsample(a, SEED, MAX_TRAIN_PIXELS // 2)
            m = make_lgb(leaves, SEED).fit(X_tr[a], y_tr[a])
            pred = m.predict(X_tr[b]).reshape(-1, SIZE, SIZE)
            sc.append(evaluate(pred, y_tr[b].reshape(-1, SIZE, SIZE))["dice"])
        cv[leaves] = float(np.mean(sc))
        log(f"CV num_leaves={leaves}: mean Dice {cv[leaves]:.4f}")
    best_leaves = max(cv, key=cv.get)
    log(f"selected num_leaves={best_leaves} (by training CV only)")

    base_runs, imp_runs = [], []
    for s in SEEDS:
        # ---- BASELINE: per-pixel k-means on raw intensity, clusters -> labels by majority vote ----
        idx = subsample(np.arange(len(y_tr)), s, KMEANS_PIXELS)
        km = KMeans(N_CLASSES, n_init=3, random_state=s).fit(X_tr[idx, :1])
        cl_tr = km.predict(X_tr[:, :1])
        lookup = np.array([np.bincount(y_tr[cl_tr == c], minlength=N_CLASSES).argmax()
                           for c in range(N_CLASSES)])
        base_pred = lookup[km.predict(X_te[:, :1])].reshape(-1, SIZE, SIZE)
        base_runs.append(evaluate(base_pred, gt_te))
        log(f"seed {s} baseline k-means: Dice {base_runs[-1]['dice']:.4f}")

        # ---- IMPROVED: class-balanced LightGBM on multi-scale features + confidence flags ----
        a = subsample(np.arange(len(y_tr)), s, MAX_TRAIN_PIXELS)
        clf = make_lgb(best_leaves, s).fit(X_tr[a], y_tr[a])
        prob = clf.predict_proba(X_te)
        imp_pred = clf.classes_[prob.argmax(1)].reshape(-1, SIZE, SIZE)
        sure = (prob.max(1) >= CONF_THRESH).reshape(-1, SIZE, SIZE)
        r = evaluate(imp_pred, gt_te)
        r["flagged_fraction"] = float(1 - sure.mean())
        r["accuracy_confident"] = float((imp_pred == gt_te)[sure].mean())
        imp_runs.append(r)
        log(f"seed {s} improved LightGBM: Dice {r['dice']:.4f}, flagged {r['flagged_fraction']:.3f}")

    b_mean, b_std = ms([r["dice"] for r in base_runs])
    i_mean, i_std = ms([r["dice"] for r in imp_runs])
    if not all(np.isfinite([b_mean, i_mean])):
        log("non-finite metrics; failing")
        sys.exit(1)

    def mean_of(runs, key):
        return float(np.mean([r[key] for r in runs]))

    metrics = {
        "baseline_score": b_mean, "improved_score": i_mean,
        "improvement": i_mean - b_mean,
        "baseline_score_std": b_std, "improved_score_std": i_std,
        "baseline_accuracy": mean_of(base_runs, "accuracy"),
        "improved_accuracy": mean_of(imp_runs, "accuracy"),
        "baseline_boundary_error": mean_of(base_runs, "boundary_error"),
        "improved_boundary_error": mean_of(imp_runs, "boundary_error"),
        "baseline_f1": mean_of(base_runs, "f1"),
        "improved_f1": mean_of(imp_runs, "f1"),
        "improved_flagged_fraction": mean_of(imp_runs, "flagged_fraction"),
        "improved_accuracy_confident": mean_of(imp_runs, "accuracy_confident"),
        "selected_num_leaves": best_leaves,
    }

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["k-means", "LightGBM"], [b_mean, i_mean], yerr=[b_std, i_std], capsize=5,
           color=["grey", "steelblue"])
    ax.set_ylabel("mean Dice (test slices)")
    ax.set_title("Baseline vs improved (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig("figures/dice_comparison.png", dpi=120)
    plt.close(fig)

    results = {
        "question": ("Can a class-balanced LightGBM pixel classifier with multi-scale neighbourhood "
                     "features segment synthetic medical-style slices more accurately than a standard "
                     "per-pixel k-means baseline, while also flagging low-confidence regions?"),
        "method": ("Seeded synthetic organ-phantom slices split by image; k-means baseline on raw "
                   "intensity vs class-balanced LightGBM on multi-scale features with confidence flags; "
                   "small training-only CV for num_leaves; test set used once; seeds 42-44."),
        "dataset": {
            "name": "Synthetic organ phantom slices (seeded numpy generator, 128x128)",
            "source_url": "none (generated inside analysis.py)",
            "n_rows": N_SLICES, "n_train": int(len(tr)), "n_test": int(len(te)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "mean Dice over organ labels on test slices",
        "higher_is_better": True,
        "baseline": "Per-pixel k-means (k=4) on raw intensity; clusters mapped to labels by majority vote on training slices.",
        "improved": ("Class-balanced LightGBM per-pixel classifier on multi-scale local features "
                     "(intensity, local means/stds at 3 window sizes, gradient magnitudes); "
                     "pixels with max-probability below 0.6 are flagged as uncertain."),
        "metrics": metrics,
        "notes": ("Synthetic data only, not the paper's imaging data; the motivating paper's exact title "
                  "could not be confirmed from the extracted text. Dice is averaged over slices and organ labels. "
                  "Boundary error = error rate within 2 px of ground-truth edges. Baseline has no confidence output. "
                  "The 0.6 confidence threshold was fixed in advance, not tuned. Only num_leaves was chosen by "
                  "training-set CV. Std is over 3 model seeds with a fixed data split; it does not capture "
                  "data-sampling variability."),
    }
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    log(f"done: baseline Dice {b_mean:.4f} vs improved {i_mean:.4f}; wrote results.json")


if __name__ == "__main__":
    main()
