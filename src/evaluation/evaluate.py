"""Evaluate one or more trained models on the test split and compare them.

  python -m src.evaluation.evaluate --models models/unet.keras models/mobilenet.keras
Needs outputs/split.json (written by train.py) so the SAME test images are used.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data.dataset import make_dataset
from src.evaluation.metrics import CLASS_NAMES, confusion_matrix, scores_from_cm

PALETTE = np.array([[160, 110, 60], [150, 150, 150], [240, 220, 100], [220, 40, 40],
                    [0, 0, 0]], dtype=np.uint8)   # soil, bedrock, sand, big rock, unlabeled


def colorize(mask):
    m = np.where(mask >= 4, 4, mask)
    return PALETTE[m]


def evaluate_model(path, test_ds, n_classes=4):
    model = tf.keras.models.load_model(path, compile=False)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for x, y in test_ds:
        pred = np.argmax(model.predict(x, verbose=0), axis=-1)
        cm += confusion_matrix(y.numpy().squeeze(-1), pred, n_classes)
    return model, cm


def save_cm_plot(cm, name, out):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(5, 4.5))
    plt.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(4), CLASS_NAMES, rotation=30); plt.yticks(range(4), CLASS_NAMES)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center")
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"Confusion matrix - {name}")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def save_prediction_grid(model, test_ds, name, out, n=6):
    x, y = next(iter(test_ds.unbatch().batch(n)))
    pred = np.argmax(model.predict(x, verbose=0), axis=-1)
    n = x.shape[0]   # test set may have fewer than n images
    fig, ax = plt.subplots(3, n, figsize=(2.6 * n, 8), squeeze=False)
    for i in range(n):
        ax[0, i].imshow(x[i].numpy().astype("uint8")); ax[1, i].imshow(colorize(y[i].numpy().squeeze(-1)))
        ax[2, i].imshow(colorize(pred[i]))
        for r in range(3):
            ax[r, i].axis("off")
    for r, t in enumerate(["image", "ground truth", f"prediction ({name})"]):
        ax[r, 0].set_title(t, loc="left", fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="outputs/split.json")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--img-size", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.outputs_dir, exist_ok=True)

    test_pairs = json.load(open(a.split))["test"]
    test_ds = make_dataset(test_pairs, a.img_size, batch_size=8)
    rows, results = [], {}
    for path in a.models:
        name = os.path.splitext(os.path.basename(path))[0]
        model, cm = evaluate_model(path, test_ds)
        s = scores_from_cm(cm)
        save_cm_plot(cm, name, os.path.join(a.outputs_dir, f"cm_{name}.png"))
        save_prediction_grid(model, test_ds, name, os.path.join(a.outputs_dir, f"preds_{name}.png"))
        results[name] = {"mean_iou": s["mean_iou"], "mean_dice": s["mean_dice"],
                         "pixel_acc": s["pixel_acc"],
                         "iou": dict(zip(CLASS_NAMES, map(float, s["iou"]))),
                         "dice": dict(zip(CLASS_NAMES, map(float, s["dice"])))}
        rows.append((name, s))

    json.dump(results, open(os.path.join(a.outputs_dir, "results.json"), "w"), indent=2)
    print(f"\n{'model':<12}{'mIoU':>8}{'mDice':>8}{'pixAcc':>8}" + "".join(f"{c:>10}" for c in CLASS_NAMES))
    for name, s in rows:
        print(f"{name:<12}{s['mean_iou']:>8.3f}{s['mean_dice']:>8.3f}{s['pixel_acc']:>8.3f}"
              + "".join(f"{v:>10.3f}" for v in s["iou"]))
    print("(per-class columns are IoU)")


if __name__ == "__main__":
    main()
