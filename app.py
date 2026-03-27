"""Professional Streamlit demo for HAM10000 transfer learning inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image

from explainability import generate_gradcam_overlay
from inference import load_model_and_classes, predict_image

APP_TITLE = "Dermoscopic Skin Lesion Classification Demo"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MANIFESTS_DIR = DATA_DIR / "processed" / "manifests"
SAMPLES_DIR = RAW_DIR / "sample_images"

MODEL_CANDIDATES = [
    "best_model.keras",
    "model.keras",
    "best_model.h5",
    "model.h5",
]

LABEL_DESCRIPTIONS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


# ---------- Cached data ----------
@st.cache_data
def load_metadata() -> pd.DataFrame:
    metadata_path = RAW_DIR / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        return pd.DataFrame()
    metadata = pd.read_csv(metadata_path)
    if "image_id" in metadata.columns:
        metadata["image_id"] = metadata["image_id"].astype(str)
    return metadata


@st.cache_data
def load_sample_manifest() -> pd.DataFrame:
    sample_path = MANIFESTS_DIR / "sample_manifest.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    return pd.DataFrame(columns=["image_id", "dx", "image_path", "source_split"])


@st.cache_data
def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_model_file() -> Path | None:
    for name in MODEL_CANDIDATES:
        candidate = MODELS_DIR / name
        if candidate.exists():
            return candidate
    return None


@st.cache_resource
def load_demo_artifacts():
    model_path = _find_model_file()
    classes_path = MODELS_DIR / "class_names.json"
    if model_path is None:
        return None, None
    try:
        return load_model_and_classes(model_path, classes_path if classes_path.exists() else None)
    except Exception as exc:  # pragma: no cover - useful for interactive debugging
        st.warning(f"Could not load saved model artifacts: {exc}")
        return None, None


# ---------- UI helpers ----------
def set_page_style() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")
    st.markdown(
        """
        <style>
        .hero {
            padding: 1.2rem 1.25rem;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(17, 132, 176, 0.08), rgba(25, 118, 210, 0.02));
            margin-bottom: 1rem;
        }
        .small-note { color: #6b7280; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(metadata: pd.DataFrame) -> None:
    dataset_ready = "Yes" if not metadata.empty else "No"
    model_ready = "Yes" if _find_model_file() is not None else "No"
    st.markdown(
        f"""
        <div class="hero">
            <h1 style="margin-bottom:0.2rem;">{APP_TITLE}</h1>
            <div class="small-note">
                Educational clinical decision-support demo using transfer learning on dermatoscopic images.
                This application is not a medical device and does not replace clinician judgment or pathology.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Approach", "EfficientNetB0")
    col2.metric("Task", "7-class lesion classification")
    col3.metric("Dataset ready", dataset_ready)
    col4.metric("Model ready", model_ready)


def _resolve_sample_image_path(image_id: str, sample_manifest: pd.DataFrame) -> Optional[Path]:
    if not sample_manifest.empty:
        match = sample_manifest[sample_manifest["image_id"].astype(str) == str(image_id)]
        if not match.empty:
            candidate = Path(match.iloc[0]["image_path"])
            if candidate.exists():
                return candidate
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = SAMPLES_DIR / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def render_sidebar(metadata: pd.DataFrame, sample_manifest: pd.DataFrame):
    st.sidebar.header("Input image")
    source = st.sidebar.radio(
        "Choose image source",
        ["Upload dermatoscopic image", "Choose sample image from dataset"],
    )

    image = None
    image_label = None
    selected_metadata = None

    if source == "Upload dermatoscopic image":
        upload = st.sidebar.file_uploader(
            "Upload a dermatoscopic image",
            type=["jpg", "jpeg", "png"],
            help="The trained model is intended for dermatoscopic images, not ordinary smartphone photos.",
        )
        if upload is not None:
            image = Image.open(upload).convert("RGB")
            image_label = upload.name
    else:
        if not sample_manifest.empty:
            options_df = sample_manifest.copy()
        elif not metadata.empty and {"image_id", "dx"}.issubset(metadata.columns):
            options_df = metadata[["image_id", "dx"]].copy().head(200)
        else:
            options_df = pd.DataFrame(columns=["image_id", "dx"])

        if options_df.empty:
            st.sidebar.info("Sample images will appear here after running prepare_data.py.")
        else:
            options_df["display"] = options_df.apply(
                lambda row: f"{row['image_id']}  •  {row.get('dx', 'unknown')}", axis=1
            )
            choice = st.sidebar.selectbox("Choose a sample", options_df["display"].tolist())
            chosen_row = options_df[options_df["display"] == choice].iloc[0]
            selected_metadata = chosen_row.to_dict()
            image_path = _resolve_sample_image_path(str(chosen_row["image_id"]), sample_manifest)
            if image_path is not None and image_path.exists():
                image = Image.open(image_path).convert("RGB")
                image_label = str(image_path.name)

    show_gradcam = st.sidebar.checkbox("Show Grad-CAM overlay", value=True)
    return image, image_label, selected_metadata, show_gradcam


def render_prediction_tab(image: Image.Image | None, image_label: str | None, selected_metadata: dict | None, show_gradcam: bool) -> None:
    model, class_names = load_demo_artifacts()
    left, right = st.columns([1.05, 1.25])

    with left:
        st.subheader("Selected image")
        if image is None:
            st.info("Upload a dermatoscopic image or choose a prepared dataset sample.")
        else:
            st.image(image, caption=image_label or "Selected image", use_container_width=True)
            if selected_metadata and selected_metadata.get("dx"):
                dx = str(selected_metadata["dx"])
                st.caption(f"Sample metadata label: {dx} — {LABEL_DESCRIPTIONS.get(dx, dx)}")

    with right:
        st.subheader("Model output")
        if image is None:
            st.warning("No image selected yet.")
            return
        if model is None or class_names is None:
            st.error("Model artifact not found. Add a trained model file under models/ and reload the app.")
            st.code("models/best_model.keras (or model.keras / .h5)\nmodels/class_names.json (optional)", language="bash")
            return

        if st.button("Run prediction", type="primary"):
            with st.spinner("Running EfficientNet inference..."):
                result = predict_image(model, class_names, image)
                gradcam_overlay = None
                if show_gradcam:
                    try:
                        gradcam_overlay = generate_gradcam_overlay(model, image, target_class=result["top_class"])
                    except Exception as exc:  # pragma: no cover
                        st.info(f"Grad-CAM unavailable for this run: {exc}")

            score_cols = st.columns(2)
            score_cols[0].metric("Top predicted class", result["top_class"])
            score_cols[1].metric("Top score", f"{result['top_score']:.1%}")
            st.info(result["review_bucket"])
            st.caption(
                "Scores shown here are model outputs on the trained label set. They are not clinical certainty estimates."
            )

            top_df = pd.DataFrame(result["top_k"], columns=["Class", "Score"])
            top_df["Description"] = top_df["Class"].map(lambda x: LABEL_DESCRIPTIONS.get(x, x))
            st.dataframe(top_df, use_container_width=True, hide_index=True)

            if gradcam_overlay is not None:
                st.image(gradcam_overlay, caption="Grad-CAM overlay", use_container_width=True)

            with st.expander("Detailed probability table", expanded=False):
                prob_df = pd.DataFrame(
                    [{"Class": k, "Score": v, "Description": LABEL_DESCRIPTIONS.get(k, k)} for k, v in result["all_scores"].items()]
                ).sort_values("Score", ascending=False)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)


def _comparison_table(model_metrics: dict, baseline_metrics: dict) -> pd.DataFrame:
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    rows = []
    for key in keys:
        rows.append(
            {
                "Metric": key,
                "Model": model_metrics.get(key),
                "Majority baseline": baseline_metrics.get(key),
            }
        )
    return pd.DataFrame(rows)


def _format_metric_value(value: float | int | None, style: str = "percent") -> str:
    if value is None:
        return "N/A"
    if style == "float3":
        return f"{value:.3f}"
    return f"{value:.1%}"


def render_comparison_tab() -> None:
    st.subheader("Comparison panel")
    st.caption(
        "This panel compares the trained image model with a non-clinical benchmark. It does not claim to reproduce physician diagnosis."
    )

    test_metrics = load_json_if_exists(MODELS_DIR / "evaluation_test.json")
    baseline_metrics = load_json_if_exists(MODELS_DIR / "baseline_test.json")
    training_summary = load_json_if_exists(MODELS_DIR / "training_summary.json")

    if test_metrics is None or baseline_metrics is None:
        st.info("Comparison metrics will appear here after model training completes.")
        st.code("python prepare_data.py\npython train.py", language="bash")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Model accuracy", _format_metric_value(test_metrics.get("accuracy"), style="percent"))
    metric_cols[1].metric("Model macro F1", _format_metric_value(test_metrics.get("macro_f1"), style="float3"))
    metric_cols[2].metric(
        "Baseline accuracy", _format_metric_value(baseline_metrics.get("accuracy"), style="percent")
    )
    metric_cols[3].metric(
        "Baseline macro F1", _format_metric_value(baseline_metrics.get("macro_f1"), style="float3")
    )

    st.dataframe(_comparison_table(test_metrics, baseline_metrics), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    model_cm = MODELS_DIR / "confusion_matrix_test.png"
    baseline_cm = MODELS_DIR / "confusion_matrix_baseline_test.png"
    with left:
        st.markdown("**Model confusion matrix (test)**")
        if model_cm.exists():
            st.image(str(model_cm), use_container_width=True)
        else:
            st.info("No model confusion matrix saved yet.")
    with right:
        st.markdown("**Majority-class baseline confusion matrix (test)**")
        if baseline_cm.exists():
            st.image(str(baseline_cm), use_container_width=True)
        else:
            st.info("No baseline confusion matrix saved yet.")

    with st.expander("What this comparison means", expanded=False):
        st.markdown(
            """
            - The baseline here is a **majority-class benchmark**, not a clinician benchmark.
            - This keeps the demo honest and avoids unsupported claims about physician reasoning.
            - If a literature-supported clinician workflow is added later, it should be described as that workflow, not as a generic doctor prediction.
            """
        )
        if training_summary:
            st.json(training_summary)


def render_dataset_tab(metadata: pd.DataFrame, sample_manifest: pd.DataFrame) -> None:
    st.subheader("Dataset view")
    if metadata.empty:
        st.warning("Dataset metadata not found. Add `HAM10000_metadata.csv` under data/raw/.")
        return

    summary = load_json_if_exists(DATA_DIR / "processed" / "dataset_summary.json")

    top_row = st.columns(3)
    top_row[0].metric("Metadata rows", f"{len(metadata):,}")
    top_row[1].metric("Classes in metadata", f"{metadata['dx'].nunique() if 'dx' in metadata.columns else 0}")
    top_row[2].metric("Prepared sample images", f"{len(sample_manifest):,}")

    if "dx" in metadata.columns:
        class_counts = metadata["dx"].value_counts().rename_axis("Class").reset_index(name="Count")
        class_counts["Description"] = class_counts["Class"].map(lambda x: LABEL_DESCRIPTIONS.get(x, x))
        st.dataframe(class_counts, use_container_width=True, hide_index=True)

    if summary is not None:
        with st.expander("Prepared split summary", expanded=False):
            st.json(summary)

    with st.expander("Metadata preview", expanded=False):
        st.dataframe(metadata.head(25), use_container_width=True)


def render_build_notes_tab() -> None:
    st.subheader("How to run this demo")
    st.markdown(
        """
        **Minimal (external training workflow)**

        1. Train your model externally (e.g., Colab).
        2. Copy model artifact into `models/`:
           - `best_model.keras` (recommended), or `model.keras`, `.h5`
        3. (Optional) Add `models/class_names.json`.
        4. Launch:
           ```bash
           streamlit run app.py
           ```

        **Optional local data/training workflow**
        - Use `prepare_data.py` and `train.py` only if you want to rebuild locally.
        """
    )

    with st.expander("Implementation notes", expanded=False):
        st.markdown(
            """
            - Transfer learning uses EfficientNetB0.
            - The comparison panel intentionally uses a non-clinical baseline.
            - Grad-CAM is for qualitative inspection only.
            - Future extensions could add calibration plots, alternate backbones, or a held-out sample gallery.
            """
        )


def main() -> None:
    set_page_style()
    metadata = load_metadata()
    sample_manifest = load_sample_manifest()
    render_header(metadata)
    image, image_label, selected_metadata, show_gradcam = render_sidebar(metadata, sample_manifest)

    tabs = st.tabs(["Prediction", "Comparison", "Dataset", "Build notes"])
    with tabs[0]:
        render_prediction_tab(image, image_label, selected_metadata, show_gradcam)
    with tabs[1]:
        render_comparison_tab()
    with tabs[2]:
        render_dataset_tab(metadata, sample_manifest)
    with tabs[3]:
        render_build_notes_tab()


if __name__ == "__main__":
    main()
