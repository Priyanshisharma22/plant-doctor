"""FastAPI inference server.  Run:  uvicorn app:app --host 0.0.0.0 --port 8000"""
import json, pathlib, sys
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "ml"))
from preprocess import load_image, leaf_ratio, to_model_input  # noqa: E402

HERE = pathlib.Path(__file__).parent
LABELS = (HERE / "labels.txt").read_text().splitlines()
TREATMENTS = json.loads((HERE / "treatments.json").read_text(encoding="utf-8"))

interp = tf.lite.Interpreter(model_path=str(HERE / "plant_disease.tflite"))
interp.allocate_tensors()
IN, OUT = interp.get_input_details()[0], interp.get_output_details()[0]

app = FastAPI(title="Plant Disease Detection API")


def pretty(label):
    plant, _, cond = label.partition("___")
    return plant.replace("_", " ").replace(",", ""), cond.replace("_", " ")


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(LABELS)}


@app.get("/classes")
def classes():
    return LABELS


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        rgb = load_image(await file.read())
    except ValueError:
        raise HTTPException(400, "Invalid image file")

    ratio = leaf_ratio(rgb)
    interp.set_tensor(IN["index"], to_model_input(rgb))
    interp.invoke()
    probs = interp.get_tensor(OUT["index"])[0]
    top = np.argsort(probs)[::-1][:3]

    best = LABELS[top[0]]
    plant, cond = pretty(best)
    info = TREATMENTS.get(best, {})
    conf = float(probs[top[0]])
    return {
        "plant": plant,
        "condition": cond,
        "healthy": "healthy" in best.lower(),
        "confidence": round(conf, 4),
        "treatment": info.get("treatment", "No treatment data available."),
        "prevention": info.get("prevention", ""),
        "top3": [{"label": LABELS[i], "confidence": round(float(probs[i]), 4)} for i in top],
        "warning": ("Image does not look like a leaf - retake the photo." if ratio < 0.15
                    else "Low confidence - retake in good light, close to one leaf." if conf < 0.6 else None),
    }
