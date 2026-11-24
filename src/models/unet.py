"""Model A: plain U-Net. Model B: U-Net with a MobileNetV2 encoder (transfer learning).

Both take a 0..255 RGB image and output per-pixel softmax over `n_classes`.
Normalization is done inside the model, so inference code stays simple.
"""
import tensorflow as tf
from tensorflow.keras import Model, layers


def _conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    return x


def build_unet(input_shape=(256, 256, 3), n_classes=4, base=32):
    """Model A: classic U-Net (4 down/up levels)."""
    inp = layers.Input(input_shape)
    x = layers.Rescaling(1.0 / 255)(inp)
    skips, filters = [], [base, base * 2, base * 4, base * 8]
    for f in filters:                                   # encoder
        x = _conv_block(x, f)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
    x = _conv_block(x, base * 16)                       # bottleneck
    for f, skip in zip(reversed(filters), reversed(skips)):   # decoder
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, f)
    out = layers.Conv2D(n_classes, 1, activation="softmax")(x)
    return Model(inp, out, name="unet_plain")


def build_unet_mobilenet(input_shape=(256, 256, 3), n_classes=4, imagenet=True):
    """Model B: MobileNetV2 encoder + U-Net decoder (needs input size divisible by 32)."""
    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False,
            weights="imagenet" if imagenet else None)
    except Exception as e:  # no internet -> fall back to random init
        print(f"[warn] could not load ImageNet weights ({e}); using random init")
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights=None)

    skip_names = ["block_1_expand_relu",    # 1/2
                  "block_3_expand_relu",    # 1/4
                  "block_6_expand_relu",    # 1/8
                  "block_13_expand_relu"]   # 1/16
    encoder = Model(base.input, [base.get_layer(n).output for n in skip_names]
                    + [base.get_layer("block_16_project").output], name="mobilenet_encoder")

    inp = layers.Input(input_shape)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inp)   # MobileNetV2 expects [-1, 1]
    s1, s2, s3, s4, x = encoder(x)
    for f, skip in zip([256, 128, 64, 32], [s4, s3, s2, s1]):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, f)
    x = layers.Conv2DTranspose(16, 2, strides=2, padding="same")(x)   # back to full size
    x = _conv_block(x, 16)
    out = layers.Conv2D(n_classes, 1, activation="softmax")(x)
    return Model(inp, out, name="unet_mobilenetv2")


def get_model(name, input_shape=(256, 256, 3), n_classes=4, imagenet=True):
    if name == "unet":
        return build_unet(input_shape, n_classes)
    if name == "mobilenet":
        return build_unet_mobilenet(input_shape, n_classes, imagenet)
    raise ValueError("model must be 'unet' or 'mobilenet'")
