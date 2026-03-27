# HAM10000 Skin Lesion Demo (Transfer Learning + Streamlit)

This project is a **low-risk, honest demo scaffold** for dermoscopic lesion classification using the HAM10000 dataset.

## What is implemented

- Dataset preparation and verification (`prepare_data.py`)
- Group-aware train/validation/test manifest creation
- Sample image export for the Streamlit UI
- EfficientNetB0 transfer-learning training pipeline (`train.py`)
- Saved model + class-name artifacts for inference
- Streamlit app with two input modes:
  - upload a dermatoscopic image
  - choose a sample image from the dataset
- Grad-CAM overlay support
- Comparison panel using a **majority-class baseline**, not a fake clinician baseline

## Project structure

```text
skin_cancer_demo_scaffold/
  app.py
  prepare_data.py
  train.py
  inference.py
  explainability.py
  requirements.txt
  README.md
  data/
    raw/
      HAM10000_metadata.csv
      HAM10000_images_part_1/
      HAM10000_images_part_2/
      sample_images/
    processed/
      manifests/
        train_manifest.csv
        val_manifest.csv
        test_manifest.csv
        sample_manifest.csv
      dataset_summary.json
  models/
    best_model.keras
    class_names.json
    evaluation_test.json
    baseline_test.json
    confusion_matrix_test.png
    confusion_matrix_baseline_test.png
```

## Setup

```bash
pip install -r requirements.txt
```

## Prepare the data

Place these under `data/raw/`:
- `HAM10000_metadata.csv`
- `HAM10000_images_part_1/`
- `HAM10000_images_part_2/`

Then run:

```bash
python prepare_data.py
```

## Train the model

```bash
python train.py
```

Optional training flags:

```bash
python train.py --epochs 8 --fine-tune-epochs 4 --batch-size 32
```

## Run the app

```bash
streamlit run app.py
```

## Notes on the comparison panel

This project deliberately avoids presenting a fabricated “doctor prediction.”

The default comparison is:
- **trained image model** vs
- **majority-class benchmark**

That keeps the demo conservative and easy to defend.

## Known limitations

- The app expects dermatoscopic images, not arbitrary clinical photos.
- Model scores are not calibrated clinical certainty estimates.
- Grad-CAM is qualitative and should not be overinterpreted.
- This is an educational demo, not a diagnostic system.

## Future enhancements

- Add calibration plots
- Add an alternate pretrained backbone for benchmark comparison
- Add sample-case cards in the UI
- Add model cards / experiment tracking
