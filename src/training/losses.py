"""Loss = class-weighted cross-entropy + Dice, both ignoring unlabeled (255) pixels."""
import tensorflow as tf


def make_loss(class_weights, num_classes=4, ignore=255, dice_weight=1.0):
    cw = tf.constant(class_weights, tf.float32)

    def loss(y_true, y_pred):
        y = tf.squeeze(tf.cast(y_true, tf.int32), -1)                  # (B,H,W)
        valid = tf.cast(tf.not_equal(y, ignore), tf.float32)           # 1 where labeled
        y = tf.where(y >= num_classes, tf.zeros_like(y), y)            # safe index for ignored px

        # weighted cross-entropy over labeled pixels only
        ce = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        w = tf.gather(cw, y) * valid
        ce = tf.reduce_sum(ce * w) / (tf.reduce_sum(w) + 1e-6)

        # soft Dice over labeled pixels only
        onehot = tf.one_hot(y, num_classes) * valid[..., None]
        probs = y_pred * valid[..., None]
        inter = tf.reduce_sum(onehot * probs, axis=(0, 1, 2))
        denom = tf.reduce_sum(onehot, axis=(0, 1, 2)) + tf.reduce_sum(probs, axis=(0, 1, 2))
        dice = 1.0 - tf.reduce_mean((2.0 * inter + 1.0) / (denom + 1.0))
        return ce + dice_weight * dice

    return loss
