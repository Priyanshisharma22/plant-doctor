"""Convert the trained Keras model to TensorFlow Lite and copy it into the Android app + backend."""
import argparse, shutil, numpy as np, tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="artifacts/plant_disease.keras")
ap.add_argument("--labels", default="artifacts/labels.txt")
ap.add_argument("--out", default="artifacts/plant_disease.tflite")
a = ap.parse_args()

model = tf.keras.models.load_model(a.model)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]   # dynamic-range quantization (~4x smaller)
open(a.out, "wb").write(conv.convert())

# sanity check: TFLite output should match Keras output closely
x = np.random.randint(0, 255, (1, 224, 224, 3)).astype(np.float32)
it = tf.lite.Interpreter(model_path=a.out); it.allocate_tensors()
it.set_tensor(it.get_input_details()[0]["index"], x); it.invoke()
tfl = it.get_tensor(it.get_output_details()[0]["index"])
print("max |keras - tflite| =", float(np.abs(model.predict(x, verbose=0) - tfl).max()))
print("model size (MB):", round(len(open(a.out, "rb").read()) / 1e6, 2))

for dest in ("android/app/src/main/assets", "backend"):
    shutil.copy(a.out, f"{dest}/plant_disease.tflite")
    shutil.copy(a.labels, f"{dest}/labels.txt")
print("Copied model + labels to Android assets and backend/")
