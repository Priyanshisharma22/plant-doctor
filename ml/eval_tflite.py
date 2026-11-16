"""Measure the accuracy of the converted TFLite model on the held-out test split.
Usage (from project root):
    python ml/eval_tflite.py --data_dir "../plantvillage dataset/color" --n 2000
Uses the same stratified split (random_state=42) as ml/train.py.
"""
import argparse, pathlib, sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.append(str(pathlib.Path(__file__).parent))
from preprocess import load_image, to_model_input

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--model", default="artifacts/plant_disease.tflite")
ap.add_argument("--n", type=int, default=2000, help="max test images to evaluate (0 = all)")
a = ap.parse_args()

root = pathlib.Path(a.data_dir)
names = sorted(d.name for d in root.iterdir() if d.is_dir())
paths, labels = [], []
for i, n in enumerate(names):
    for p in (root / n).iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(str(p)); labels.append(i)
paths, labels = np.array(paths), np.array(labels)
_, p_rest, _, y_rest = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=42)
_, p_te, _, y_te = train_test_split(p_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42)

if a.n and a.n < len(p_te):
    idx = np.random.RandomState(0).permutation(len(p_te))[: a.n]
    p_te, y_te = p_te[idx], y_te[idx]

it = tf.lite.Interpreter(model_path=a.model); it.allocate_tensors()
inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
correct = 0
for k, (p, y) in enumerate(zip(p_te, y_te), 1):
    it.set_tensor(inp, to_model_input(load_image(p)))
    it.invoke()
    correct += int(np.argmax(it.get_tensor(out)[0]) == y)
    if k % 500 == 0:
        print(f"{k}/{len(p_te)}  running accuracy {correct / k:.4f}")
print(f"TFLite accuracy on {len(p_te)} test images: {correct / len(p_te):.4f}")