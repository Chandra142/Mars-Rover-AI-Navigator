"""Hazard analysis: segmentation mask -> class grid -> traversal-cost grid.

Class ids: 0 soil, 1 bedrock, 2 sand, 3 big_rock, 4 unknown (unlabeled / uncertain).
Blocked cells have cost = np.inf.
"""
import cv2
import numpy as np
import yaml

CLASSES = ["soil", "bedrock", "sand", "big_rock", "unknown"]
UNKNOWN = 4
SAFE = (0, 1)      # bedrock, soil
HAZARD = (2, 3)    # sand, big rock


def load_costs(path="configs/costs.yaml"):
    cfg = yaml.safe_load(open(path))
    return cfg["costs"], cfg.get("grid_size", 64), cfg.get("inflate_blocked", 1)


def mask_to_grid(mask, grid_size=64):
    """Downsample an (H,W) class mask to (grid,grid) by majority vote per cell."""
    m = np.where(mask > 3, UNKNOWN, mask).astype(np.int64)
    onehot = np.stack([(m == c).astype(np.float32) for c in range(5)], axis=-1)
    small = cv2.resize(onehot, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    return np.argmax(small, axis=-1)


def class_grid_to_cost(class_grid, costs, inflate=1):
    """Look up each cell's cost; None -> blocked (inf). Optionally grow blocked areas."""
    lut = np.array([np.inf if costs[c] is None else float(costs[c]) for c in CLASSES])
    cost = lut[class_grid]
    if inflate > 0:
        blocked = np.isinf(cost).astype(np.uint8)
        k = np.ones((2 * inflate + 1, 2 * inflate + 1), np.uint8)
        cost[cv2.dilate(blocked, k) > 0] = np.inf
    return cost
