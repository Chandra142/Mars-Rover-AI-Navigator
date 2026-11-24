"""Train a segmentation model.

Example:
  python -m src.training.train --model unet --images-dir IMG --labels-dir LAB
  python -m src.training.train --model mobilenet --images-dir IMG --labels-dir LAB --epochs 20
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

from src.data.dataset import compute_class_weights, find_pairs, make_dataset, split_pairs
from src.models.unet import get_model
from src.training.losses import make_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["unet", "mobilenet"], required=True)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--images-dir")
    ap.add_argument("--labels-dir")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--models-dir")
    ap.add_argument("--outputs-dir")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    d, t, p = cfg["data"], cfg["train"], cfg["paths"]
    images_dir = a.images_dir or d["images_dir"]
    labels_dir = a.labels_dir or d["labels_dir"]
    epochs = a.epochs or t["epochs"]
    max_images = a.max_images or d["max_images"]
    models_dir = a.models_dir or p["models_dir"]
    outputs_dir = a.outputs_dir or p["outputs_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(d["seed"])

    pairs = find_pairs(images_dir, labels_dir)
    if max_images:
        pairs = pairs[:max_images]
    train_p, val_p, test_p = split_pairs(pairs, d["split"], d["seed"])
    print(f"pairs: {len(pairs)} -> train {len(train_p)}, val {len(val_p)}, test {len(test_p)}")
    json.dump({"train": train_p, "val": val_p, "test": test_p},
              open(os.path.join(outputs_dir, "split.json"), "w"))   # evaluate.py reuses this

    size = d["img_size"]
    train_ds = make_dataset(train_p, size, t["batch_size"], training=True, seed=d["seed"])
    val_ds = make_dataset(val_p, size, t["batch_size"])

    cw = compute_class_weights(train_p, d["num_classes"], d["ignore_label"]) \
        if t["use_class_weights"] else [1.0] * d["num_classes"]
    print("class weights:", [round(float(x), 2) for x in cw])

    model = get_model(a.model, (size, size, 3), d["num_classes"], t["imagenet_weights"])
    model.compile(optimizer=tf.keras.optimizers.Adam(t["lr"]),
                  loss=make_loss(cw, d["num_classes"], d["ignore_label"], t["dice_weight"]))

    out_path = os.path.join(models_dir, f"{a.model}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(out_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    json.dump(hist.history, open(os.path.join(outputs_dir, f"history_{a.model}.json"), "w"))
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title(a.model)
    plt.savefig(os.path.join(outputs_dir, f"loss_{a.model}.png"), dpi=120)
    print("saved best model to", out_path)


if __name__ == "__main__":
    main()
