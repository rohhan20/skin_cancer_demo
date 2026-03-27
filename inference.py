"""Inference helpers for the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)
TOP_K = 3

DEFAULT_HAM10000_CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# A deliberately conservative UI bucket. This is *not* a clinical diagnosis.
REVIEW_BUCKETS = {
    "mel": "Higher-priority specialist review suggested (non-diagnostic triage label).",
    "bcc": "Higher-priority specialist review suggested (non-diagnostic triage label).",
    "akiec": "Higher-priority specialist review suggested (non-diagnostic triage label).",
    "bkl": "Routine review bucket shown by model (still requires clinician assessment).",
    "nv": "Routine review bucket shown by model (still requires clinician assessment).",
    "vasc": "Routine review bucket shown by model (still requires clinician assessment).",
    "df": "Routine review bucket shown by model (still requires clinician assessment).",
}


def load_model_and_classes(model_path: Path, class_names_path: Path | None = None):
    """Load trained Keras model and class-name mapping from disk.

    If class names are not provided, fall back to:
    1) models/class_names.json (same folder as model), then
    2) default HAM10000 ordering for 7-class outputs, else generic class_i names.
    """
    import tensorflow as tf

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    resolved_classes_path = class_names_path
    if resolved_classes_path is None:
        candidate = model_path.parent / "class_names.json"
        if candidate.exists():
            resolved_classes_path = candidate

    class_names = None
    if resolved_classes_path is not None and resolved_classes_path.exists():
        with open(resolved_classes_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, list) or not all(isinstance(x, str) for x in loaded):
            raise ValueError("class_names.json must contain a JSON list of class labels.")
        class_names = loaded

    if class_names is None:
        output_dim = int(model.output_shape[-1])
        if output_dim == len(DEFAULT_HAM10000_CLASS_NAMES):
            class_names = DEFAULT_HAM10000_CLASS_NAMES
        else:
            class_names = [f"class_{i}" for i in range(output_dim)]

    return model, class_names


def preprocess_image(image: Image.Image, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize a PIL image and return a batched float32 array."""
    image = image.convert("RGB").resize(image_size)
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    return array


def _make_top_k(probabilities: np.ndarray, class_names: list[str], top_k: int = TOP_K) -> list[tuple[str, float]]:
    order = np.argsort(probabilities)[::-1][:top_k]
    return [(class_names[i], float(probabilities[i])) for i in order]


def predict_image(model: Any, class_names: list[str], image: Image.Image) -> dict:
    """Run model inference and return a UI-friendly prediction payload."""
    batch = preprocess_image(image)
    predictions = model.predict(batch, verbose=0)[0]
    top_k = _make_top_k(predictions, class_names, top_k=min(TOP_K, len(class_names)))
    top_class, top_score = top_k[0]
    review_bucket = REVIEW_BUCKETS.get(
        top_class,
        "Model output available. Clinical assessment remains necessary.",
    )
    return {
        "top_class": top_class,
        "top_score": float(top_score),
        "top_k": top_k,
        "all_scores": {class_names[i]: float(predictions[i]) for i in range(len(class_names))},
        "review_bucket": review_bucket,
    }
