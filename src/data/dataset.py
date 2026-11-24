"""Data loading: pair images with label masks, split, and build tf.data pipelines.

Classes in the label PNGs: 0 soil, 1 bedrock, 2 sand, 3 big rock, 255 = no label.
"""
import glob
import os
import random

import cv2
import numpy as np
import tensorflow as tf

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
LABEL_SUFFIXES = ("_merged", "_mask", "_label", "_labels")


def _stem(path):
    """File name without extension/suffix, lowercase, used to match image <-> label."""
    s = os.path.splitext(os.path.basename(path))[0]
    for suf in LABEL_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.lower()


def find_pairs(images_dir, labels_dir):
    """Return a sorted list of (image_path, label_path) matched by file stem."""
    imgs = {_stem(p): p for p in glob.glob(os.path.join(images_dir, "**", "*"), recursive=True)
            if p.lower().endswith(IMG_EXT)}
    labs = {_stem(p): p for p in glob.glob(os.path.join(labels_dir, "**", "*.png"), recursive=True)}
    keys = sorted(set(imgs) & set(labs))
    if not keys:
        raise FileNotFoundError(
            f"No image/label pairs found.\n images_dir={images_dir} ({len(imgs)} images)\n"
            f" labels_dir={labels_dir} ({len(labs)} pngs)\n"
            "Run scripts/inspect_dataset.py to see the real folder layout.")
    return [(imgs[k], labs[k]) for k in keys]


def split_pairs(pairs, fractions=(0.8, 0.1, 0.1), seed=42):
    """Deterministic shuffle + split into train / val / test."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]


def compute_class_weights(pairs, num_classes=4, ignore=255, sample=300, clip=10.0):
    """Inverse-frequency class weights from a sample of label masks (handles imbalance)."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, lab in pairs[:sample]:
        m = cv2.imread(lab, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = m[m != ignore]
        counts += np.bincount(m[m < num_classes], minlength=num_classes)
    counts = np.maximum(counts, 1)
    w = counts.sum() / (num_classes * counts)
    return np.clip(w, 0.2, clip).astype(np.float32)


def _load(img_path, lab_path, size):
    img = tf.io.decode_image(tf.io.read_file(img_path), channels=3, expand_animations=False)
    img = tf.image.resize(tf.cast(img, tf.float32), [size, size])          # 0..255
    lab = tf.io.decode_png(tf.io.read_file(lab_path), channels=1)
    lab = tf.image.resize(lab, [size, size], method="nearest")             # keep class ids
    return img, tf.cast(lab, tf.uint8)


def _augment(img, lab):
    """Random flip (applied to image AND mask) + brightness/contrast (image only)."""
    if tf.random.uniform(()) > 0.5:
        img, lab = tf.image.flip_left_right(img), tf.image.flip_left_right(lab)
    img = tf.image.random_brightness(img, 30.0)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return tf.clip_by_value(img, 0.0, 255.0), lab


def make_dataset(pairs, size=256, batch_size=16, training=False, seed=42):
    """Build a batched tf.data.Dataset of (image, mask)."""
    imgs = tf.constant([p[0] for p in pairs])
    labs = tf.constant([p[1] for p in pairs])
    ds = tf.data.Dataset.from_tensor_slices((imgs, labs))
    if training:
        ds = ds.shuffle(len(pairs), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda i, l: _load(i, l, size), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
