"""Data preparation for the HAM10000 demo.

What this script does
---------------------
1. Verifies metadata and image files exist.
2. Attaches absolute file paths to each metadata row.
3. Creates *group-aware* splits so images from the same lesion stay together.
4. Writes CSV manifests under ``data/processed/manifests``.
5. Copies a small sample-image pool for the Streamlit UI.

The goal is to reduce the most common failure modes before training starts.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFESTS_DIR = PROCESSED_DIR / "manifests"
SAMPLE_DIR = RAW_DIR / "sample_images"

IMAGE_SEARCH_DIRS = [
    RAW_DIR,
    RAW_DIR / "HAM10000_images_part_1",
    RAW_DIR / "HAM10000_images_part_2",
    RAW_DIR / "images",
]

REQUIRED_METADATA_COLUMNS = {"image_id", "dx"}


def load_metadata() -> pd.DataFrame:
    metadata_path = RAW_DIR / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_path}. Put HAM10000_metadata.csv under data/raw/."
        )

    metadata = pd.read_csv(metadata_path)
    missing_columns = REQUIRED_METADATA_COLUMNS.difference(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing_columns)}. "
            f"Columns found: {list(metadata.columns)}"
        )

    metadata = metadata.copy()
    metadata["image_id"] = metadata["image_id"].astype(str)
    metadata["dx"] = metadata["dx"].astype(str)
    if "lesion_id" not in metadata.columns:
        # Lowest-risk fallback: each image is its own group.
        metadata["lesion_id"] = metadata["image_id"]
    else:
        metadata["lesion_id"] = metadata["lesion_id"].astype(str)
    return metadata


def _candidate_paths(image_id: str) -> Iterable[Path]:
    for directory in IMAGE_SEARCH_DIRS:
        for ext in (".jpg", ".jpeg", ".png"):
            yield directory / f"{image_id}{ext}"


def _find_image_path(image_id: str) -> Path | None:
    for candidate in _candidate_paths(image_id):
        if candidate.exists():
            return candidate.resolve()
    return None


def attach_image_paths(metadata: pd.DataFrame) -> pd.DataFrame:
    """Match metadata rows to actual image files and attach ``image_path``."""
    metadata = metadata.copy()
    metadata["image_path"] = metadata["image_id"].map(_find_image_path)

    missing = metadata[metadata["image_path"].isna()]["image_id"].tolist()
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Could not find image files for {len(missing)} metadata rows. "
            f"Examples: {preview}. Expected folders include: {IMAGE_SEARCH_DIRS}"
        )

    metadata["image_path"] = metadata["image_path"].astype(str)
    return metadata


def _select_groups_for_fold(group_df: pd.DataFrame, n_splits: int, seed: int):
    """Return (train_groups, held_out_groups) using a single SGKF fold.

    We use the first fold only, because the purpose here is a deterministic
    project split rather than cross-validation.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_x = group_df[["group_id"]]
    y = group_df["label"]
    groups = group_df["group_id"]
    train_idx, test_idx = next(splitter.split(dummy_x, y, groups=groups))
    return set(group_df.iloc[train_idx]["group_id"]), set(group_df.iloc[test_idx]["group_id"])


def create_splits(metadata: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic, group-aware train/val/test splits.

    Approximate proportions:
    - test: ~20%
    - val:  ~16% of full data
    - train: remainder

    Keeping lesion groups together is more important than hitting an exact ratio.
    """
    data = metadata.copy()

    group_df = (
        data.groupby("lesion_id", as_index=False)
        .agg(label=("dx", "first"))
        .rename(columns={"lesion_id": "group_id"})
    )

    train_val_groups, test_groups = _select_groups_for_fold(group_df, n_splits=5, seed=seed)
    train_val_df = data[data["lesion_id"].isin(train_val_groups)].copy()
    test_df = data[data["lesion_id"].isin(test_groups)].copy()

    inner_group_df = (
        train_val_df.groupby("lesion_id", as_index=False)
        .agg(label=("dx", "first"))
        .rename(columns={"lesion_id": "group_id"})
    )
    train_groups, val_groups = _select_groups_for_fold(inner_group_df, n_splits=5, seed=seed + 1)

    train_df = train_val_df[train_val_df["lesion_id"].isin(train_groups)].copy()
    val_df = train_val_df[train_val_df["lesion_id"].isin(val_groups)].copy()

    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        split_df["split"] = split_name

    return train_df, val_df, test_df


def save_manifests(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        output_path = MANIFESTS_DIR / f"{name}_manifest.csv"
        df.to_csv(output_path, index=False)


def export_sample_pool(source_df: pd.DataFrame, samples_per_class: int = 8, overwrite: bool = True) -> pd.DataFrame:
    """Create a small sample-image pool for the app.

    Uses the held-out split by default so the app can demonstrate real inference on
    unseen images.
    """
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old_file in SAMPLE_DIR.glob("*"):
            if old_file.is_file():
                old_file.unlink()

    sampled = (
        source_df.sort_values(["dx", "image_id"])
        .groupby("dx", group_keys=False)
        .head(samples_per_class)
        .copy()
    )

    records: list[dict] = []
    for row in sampled.itertuples(index=False):
        src = Path(row.image_path)
        dst = SAMPLE_DIR / f"{row.image_id}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        records.append(
            {
                "image_id": row.image_id,
                "dx": row.dx,
                "image_path": str(dst.resolve()),
                "source_split": getattr(row, "split", "unknown"),
            }
        )

    sample_df = pd.DataFrame(records)
    sample_df.to_csv(MANIFESTS_DIR / "sample_manifest.csv", index=False)
    return sample_df


def save_summary(metadata: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    summary = {
        "total_rows": int(len(metadata)),
        "class_counts": metadata["dx"].value_counts().sort_index().to_dict(),
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "split_class_counts": {
            "train": train_df["dx"].value_counts().sort_index().to_dict(),
            "val": val_df["dx"].value_counts().sort_index().to_dict(),
            "test": test_df["dx"].value_counts().sort_index().to_dict(),
        },
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HAM10000 metadata and manifests.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=8,
        help="How many sample images per class to copy for the Streamlit UI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata()
    metadata = attach_image_paths(metadata)
    train_df, val_df, test_df = create_splits(metadata, seed=args.seed)
    save_manifests(train_df, val_df, test_df)
    sample_df = export_sample_pool(test_df, samples_per_class=args.samples_per_class)
    save_summary(metadata, train_df, val_df, test_df)

    print(f"Loaded metadata rows: {len(metadata):,}")
    print(
        "Saved manifests: "
        f"train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}."
    )
    print(f"Exported {len(sample_df):,} sample images to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
