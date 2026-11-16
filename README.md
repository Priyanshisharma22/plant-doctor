# Plant Disease Detection with Mobile Deployment

Detects plant diseases from leaf photos using a CNN (MobileNetV2 transfer learning, plus a from-scratch CNN baseline),
converts it to TensorFlow Lite, and ships it in an Android app that shows the disease and basic treatment advice for farmers.

**Stack:** Python, TensorFlow/Keras, CNN, OpenCV, TensorFlow Lite, FastAPI, Android Studio (Kotlin)

## Architecture
```
PlantVillage images -> ml/train.py (augment + MobileNetV2) -> plant_disease.keras
                    -> ml/convert_tflite.py (quantised)    -> plant_disease.tflite + labels.txt
                         |-> android/app/src/main/assets  (offline, on-device inference)
                         |-> backend/ (FastAPI + OpenCV leaf check, optional server inference)
treatments.json (38 classes) provides disease name -> treatment / prevention text
```

## 1. Train
```bash
pip install -r requirements.txt
# Download PlantVillage (color images, 38 classes): https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
python ml/train.py --data_dir data/plantvillage/color --model mobilenet --epochs 8
python ml/train.py --data_dir data/plantvillage/color --model cnn --epochs 15 --out artifacts_cnn   # baseline for comparison
```
Outputs in `artifacts/`: model, `labels.txt`, training curves, confusion matrix, classification report, `metrics.json`.
Tip: no GPU? Run on Google Colab (free T4) and download `artifacts/`.

## 2. Convert to TFLite
```bash
python ml/convert_tflite.py     # also copies model + labels into android assets and backend/
```

## 3. Backend (optional, FastAPI)
```bash
cd backend && python make_treatments.py   # only needed if you edit the advice text
uvicorn app:app --host 0.0.0.0 --port 8000
# test: curl -F "file=@leaf.jpg" http://localhost:8000/predict
```
OpenCV is used to decode/resize images and to estimate leaf-pixel ratio (HSV mask) so non-leaf photos get a warning.

## 4. Android app
1. Open the `android/` folder in Android Studio (Koala or newer) and let Gradle sync.
2. Make sure `app/src/main/assets/` contains `plant_disease.tflite`, `labels.txt`, `treatments.json` (step 2 does this).
3. Run on a phone or emulator. Take a photo or pick from gallery -> disease, confidence, treatment, prevention.
The app runs fully offline (important for farmers with poor connectivity).

## Notes / limitations (good to mention in your report)
- PlantVillage images are lab-style (single leaf, plain background); accuracy drops on real field photos. Future work: add field-image datasets (e.g. PlantDoc), more augmentation.
- Treatment text is general guidance only - confirm with a local agriculture officer before spraying.
- Quantisation shrinks the model ~4x with negligible accuracy loss (verified in `convert_tflite.py`).

## Report outline
Introduction -> Literature survey -> Dataset (38 classes, 70/15/15 split) -> Methodology (preprocessing, augmentation, CNN vs MobileNetV2, fine-tuning) -> TFLite conversion -> Backend & app design -> Results (accuracy, confusion matrix, per-class F1, CNN vs MobileNetV2 table, model size) -> Limitations & future work.
