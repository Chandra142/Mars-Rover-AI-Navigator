"""Print the real folder layout of the downloaded dataset so you can set the paths.

  python scripts/inspect_dataset.py /kaggle/input
  python scripts/inspect_dataset.py path/to/ai4mars
"""
import os
import sys
from collections import Counter

import cv2
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "."
MAX_DEPTH = 5
for cur, dirs, files in os.walk(root):
    depth = cur[len(root):].count(os.sep)
    if depth > MAX_DEPTH:
        dirs[:] = []
        continue
    ext = Counter(os.path.splitext(f)[1].lower() for f in files)
    if files:
        print(f"{'  ' * depth}{cur}  files={len(files)} {dict(ext)}  e.g. {files[0]}")
    else:
        print(f"{'  ' * depth}{cur}/")

# peek at label values in the first few PNGs found
shown = 0
for cur, _, files in os.walk(root):
    for f in files:
        if f.lower().endswith(".png") and shown < 3:
            m = cv2.imread(os.path.join(cur, f), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                print(f"\nlabel check {os.path.join(cur, f)}: shape={m.shape} values={np.unique(m)[:10]}")
                shown += 1
print("\nExpected label values: 0 soil, 1 bedrock, 2 sand, 3 big rock, 255 unlabeled.")
print("Pick images_dir and labels_dir so file names match (same stem), then pass them to train.py.")
