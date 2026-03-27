"""Grad-CAM explainability utilities for the Streamlit app."""

from __future__ import annotations

import numpy as np
from PIL import Image


def _output_rank(layer) -> int | None:
    """Return output tensor rank for a layer if available."""
    output = getattr(layer, "output", None)
    if output is not None and hasattr(output, "shape"):
        shape = output.shape
        if hasattr(shape, "rank") and shape.rank is not None:
            return int(shape.rank)
        try:
            return len(shape)
        except TypeError:
            pass

    output_shape = getattr(layer, "output_shape", None)
    if output_shape is not None:
        try:
            return len(output_shape)
        except TypeError:
            pass
    return None


def _is_conv_feature_layer(layer) -> bool:
    """Heuristic for Grad-CAM candidate layers.

    Grad-CAM needs a spatial feature map (rank-4 tensor) from a conv-like layer.
    """
    rank = _output_rank(layer)
    if rank != 4:
        return False

    class_name = layer.__class__.__name__.lower()
    conv_like = ("conv" in class_name) or ("depthwise" in class_name) or ("separable" in class_name)
    if conv_like:
        return True

    # Fallback for unusual wrappers: still accept 4D feature-producing layers.
    return True


def _find_last_conv_layer(model):
    """Find the deepest conv-like layer with a 4D output.

    This works well for many transfer-learning models, including EfficientNet.
    """
    stack = list(reversed(getattr(model, "layers", [])))
    while stack:
        layer = stack.pop(0)
        nested_layers = getattr(layer, "layers", None)
        if nested_layers:
            stack = list(reversed(nested_layers)) + stack
        if _is_conv_feature_layer(layer):
            return layer

    raise ValueError("Could not find a 4D convolutional feature layer for Grad-CAM.")


def _heatmap_to_image(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    import matplotlib.cm as cm

    heatmap = np.uint8(255 * heatmap)
    colormap = cm.get_cmap("jet")
    colored = colormap(heatmap)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored).resize(size)


def generate_gradcam_overlay(model, image: Image.Image, target_class: str | None = None, alpha: float = 0.35):
    """Generate a Grad-CAM overlay as a PIL image.

    The ``target_class`` argument is optional because the function can default to
    the model's top predicted class.
    """
    import tensorflow as tf
    from inference import preprocess_image

    last_conv_layer = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    batch = preprocess_image(image)
    input_tensor = tf.convert_to_tensor(batch)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        if target_class is None:
            class_idx = tf.argmax(predictions[0])
        else:
            # Try to resolve the target class by matching model output names if needed.
            class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()

    base_image = image.convert("RGB")
    color_map = _heatmap_to_image(heatmap, base_image.size)
    overlay = Image.blend(base_image, color_map, alpha=alpha)
    return overlay
