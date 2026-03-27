"""Training pipeline for transfer learning on HAM10000.

This script follows the lowest-risk path:
- manifest-driven loading
- EfficientNetB0 transfer learning
- frozen-head training first
- optional short fine-tuning phase
- majority-class baseline for the comparison panel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFESTS_DIR = PROCESSED_DIR / "manifests"
IMAGE_SIZE = (224, 224)
AUTOTUNE = None  # set after TensorFlow import


# ---------- Manifest loading ----------
def load_manifest(name: str) -> pd.DataFrame:
    manifest_path = MANIFESTS_DIR / f"{name}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}. Run `python prepare_data.py` first."
        )
    df = pd.read_csv(manifest_path)
    required = {"image_path", "dx"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest {manifest_path} must include columns {sorted(required)}")
    return df


def build_label_mapping(train_df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    class_names = sorted(train_df["dx"].unique().tolist())
    class_to_index = {label: i for i, label in enumerate(class_names)}
    return class_names, class_to_index


# ---------- Dataset helpers ----------
def decode_image(path, label, image_size):
    import tensorflow as tf

    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    return image, label


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_to_index: dict[str, int],
    batch_size: int = 32,
    image_size: tuple[int, int] = IMAGE_SIZE,
):
    import tensorflow as tf

    global AUTOTUNE
    AUTOTUNE = tf.data.AUTOTUNE

    def to_dataset(df: pd.DataFrame, training: bool):
        labels = df["dx"].map(class_to_index).astype(np.int32).to_numpy()
        paths = df["image_path"].astype(str).to_numpy()
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(len(df), seed=42, reshuffle_each_iteration=True)
        ds = ds.map(lambda p, y: decode_image(p, y, image_size), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    return to_dataset(train_df, True), to_dataset(val_df, False), to_dataset(test_df, False)


# ---------- Model ----------
def build_model(num_classes: int, image_size: tuple[int, int] = IMAGE_SIZE, dropout_rate: float = 0.3):
    import tensorflow as tf
    from tensorflow import keras

    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.03),
            keras.layers.RandomZoom(0.08),
            keras.layers.RandomContrast(0.05),
        ],
        name="augmentation",
    )

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(image_size[0], image_size[1], 3), name="image")
    x = data_augmentation(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = keras.Model(inputs, outputs, name="ham10000_efficientnetb0")
    return model, base_model


# ---------- Training + evaluation ----------
def compute_class_weights(train_df: pd.DataFrame, class_to_index: dict[str, int]) -> dict[int, float]:
    labels = train_df["dx"].map(class_to_index).to_numpy()
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def compile_model(model, learning_rate: float):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def fine_tune_backbone(base_model, freeze_fraction: float = 0.8):
    import tensorflow as tf

    base_model.trainable = True
    freeze_until = int(len(base_model.layers) * freeze_fraction)
    for idx, layer in enumerate(base_model.layers):
        if idx < freeze_until or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def collect_predictions(model, dataset):
    y_true_batches = []
    y_pred_batches = []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true_batches.append(labels.numpy())
        y_pred_batches.append(np.argmax(preds, axis=1))
    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)
    return y_true, y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def compute_majority_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame, class_to_index: dict[str, int], class_names: list[str]) -> dict:
    majority_label = train_df["dx"].value_counts().idxmax()
    y_true = eval_df["dx"].map(class_to_index).to_numpy()
    y_pred = np.full_like(y_true, fill_value=class_to_index[majority_label])
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["majority_label"] = majority_label
    return metrics


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def train_model(args: argparse.Namespace):
    train_df = load_manifest("train")
    val_df = load_manifest("val")
    test_df = load_manifest("test")
    class_names, class_to_index = build_label_mapping(train_df)

    train_ds, val_ds, test_ds = build_datasets(
        train_df, val_df, test_df, class_to_index, batch_size=args.batch_size, image_size=(args.image_size, args.image_size)
    )

    model, base_model = build_model(num_classes=len(class_names), image_size=(args.image_size, args.image_size), dropout_rate=args.dropout)
    class_weights = compute_class_weights(train_df, class_to_index)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "best_model.keras"

    import tensorflow as tf

    compile_model(model, learning_rate=args.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    history_finetune = None
    if args.fine_tune_epochs > 0:
        fine_tune_backbone(base_model, freeze_fraction=args.freeze_fraction)
        compile_model(model, learning_rate=args.fine_tune_learning_rate)
        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

    model = tf.keras.models.load_model(best_model_path)

    y_val_true, y_val_pred = collect_predictions(model, val_ds)
    y_test_true, y_test_pred = collect_predictions(model, test_ds)
    val_metrics = compute_metrics(y_val_true, y_val_pred, class_names)
    test_metrics = compute_metrics(y_test_true, y_test_pred, class_names)

    baseline_val = compute_majority_baseline(train_df, val_df, class_to_index, class_names)
    baseline_test = compute_majority_baseline(train_df, test_df, class_to_index, class_names)

    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    save_json(
        {
            "class_names": class_names,
            "class_weights": class_weights,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "head_epochs_requested": args.epochs,
            "fine_tune_epochs_requested": args.fine_tune_epochs,
        },
        MODELS_DIR / "training_summary.json",
    )
    save_json(val_metrics, MODELS_DIR / "evaluation_val.json")
    save_json(test_metrics, MODELS_DIR / "evaluation_test.json")
    save_json(baseline_val, MODELS_DIR / "baseline_val.json")
    save_json(baseline_test, MODELS_DIR / "baseline_test.json")

    save_confusion_matrix(
        np.array(test_metrics["confusion_matrix"]), class_names, MODELS_DIR / "confusion_matrix_test.png", "Model confusion matrix (test)"
    )
    save_confusion_matrix(
        np.array(baseline_test["confusion_matrix"]), class_names, MODELS_DIR / "confusion_matrix_baseline_test.png", "Majority baseline confusion matrix (test)"
    )

    history_payload = {
        "head": history_head.history,
        "fine_tune": history_finetune.history if history_finetune is not None else None,
    }
    save_json(history_payload, MODELS_DIR / "history.json")

    print("Training complete.")
    print(f"Saved best model to: {best_model_path}")
    print(f"Validation macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 on HAM10000 manifests.")
    parser.add_argument("--epochs", type=int, default=8, help="Head-training epochs.")
    parser.add_argument("--fine-tune-epochs", type=int, default=4, help="Optional fine-tuning epochs after head training.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--freeze-fraction", type=float, default=0.8, help="Fraction of EfficientNet layers to keep frozen during fine-tuning.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
