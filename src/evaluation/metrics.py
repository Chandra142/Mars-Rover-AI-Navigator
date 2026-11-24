"""Segmentation metrics from a confusion matrix (ignores unlabeled pixels)."""
import numpy as np

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]


def confusion_matrix(y_true, y_pred, n_classes=4, ignore=255):
    """Rows = true class, columns = predicted class."""
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    keep = (y_true != ignore) & (y_true < n_classes)
    idx = y_true[keep].astype(np.int64) * n_classes + y_pred[keep].astype(np.int64)
    return np.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def scores_from_cm(cm):
    """Per-class IoU/Dice, mean IoU (over classes present) and pixel accuracy."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = np.where(tp + fp + fn > 0, tp / np.maximum(tp + fp + fn, 1), np.nan)
    dice = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    return {
        "iou": iou, "dice": dice,
        "mean_iou": float(np.nanmean(iou)),
        "mean_dice": float(np.nanmean(dice)),
        "pixel_acc": float(tp.sum() / max(cm.sum(), 1)),
    }
