"""Train a leaf-disease classifier on PlantVillage.

Usage:
    python ml/train.py --data_dir "../plantvillage dataset/color" --model mobilenet --epochs 5 --finetune_epochs 2
Dataset layout: data_dir/<Class_Name>/*.jpg  (e.g. Tomato___Late_blight)
"""
import argparse, json, os, pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

IMG = 224
AUTOTUNE = tf.data.AUTOTUNE


def build_datasets(data_dir, batch):
    """Stratified 70/15/15 split by file path, so train/val/test never overlap
    and every class appears in every split."""
    root = pathlib.Path(data_dir)
    names = sorted(d.name for d in root.iterdir() if d.is_dir())
    paths, labels = [], []
    for i, n in enumerate(names):
        for p in (root / n).iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(p))
                labels.append(i)
    paths, labels = np.array(paths), np.array(labels)
    print(f"Found {len(paths)} images in {len(names)} classes")

    p_tr, p_rest, y_tr, y_rest = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=42)
    p_va, p_te, y_va, y_te = train_test_split(
        p_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG, IMG))
        img.set_shape((IMG, IMG, 3))
        return img, label

    def make(p, y, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((p, y))
        if shuffle:
            ds = ds.shuffle(len(p), seed=42)
        return ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)

    return make(p_tr, y_tr, True), make(p_va, y_va, False), make(p_te, y_te, False), names


def augment():
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")


def build_mobilenet(n):
    base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inp = keras.Input((IMG, IMG, 3))                # raw 0-255 pixels
    x = augment()(inp)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)   # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n, activation="softmax")(x)
    return keras.Model(inp, out), base


def build_cnn(n):
    """Plain custom CNN trained from scratch - a baseline to compare in your report."""
    inp = keras.Input((IMG, IMG, 3))
    x = augment()(inp)
    x = layers.Rescaling(1 / 255.0)(x)
    for f in (32, 64, 128, 256):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n, activation="softmax")(x)
    return keras.Model(inp, out), None


def plot_history(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["accuracy"], label="train"); ax[0].plot(hist["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(hist["loss"], label="train"); ax[1].plot(hist["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", choices=["mobilenet", "cnn"], default="mobilenet")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--finetune_epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train, val, test, names = build_datasets(a.data_dir, a.batch)
    print(f"{len(names)} classes")
    with open(f"{a.out}/labels.txt", "w") as f:
        f.write("\n".join(names))

    model, base = (build_mobilenet if a.model == "mobilenet" else build_cnn)(len(names))
    model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.3)]
    h = model.fit(train, validation_data=val, epochs=a.epochs, callbacks=cbs).history

    if base is not None and a.finetune_epochs > 0:   # fine-tune the top of the backbone
        base.trainable = True
        for l in base.layers[:-30]:
            l.trainable = False
        model.compile(keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])
        h2 = model.fit(train, validation_data=val, epochs=a.finetune_epochs, callbacks=cbs).history
        for k in h:
            h[k] += h2[k]

    plot_history(h, f"{a.out}/training_curves.png")
    model.save(f"{a.out}/plant_disease.keras")
    print("Model saved. Evaluating on the test set...")

    y_true = np.concatenate([y.numpy() for _, y in test])
    y_pred = np.argmax(model.predict(test), axis=1)
    all_ids = list(range(len(names)))
    rep = classification_report(y_true, y_pred, labels=all_ids, target_names=names, digits=4, zero_division=0)
    print(rep)
    open(f"{a.out}/classification_report.txt", "w").write(rep)
    cm = confusion_matrix(y_true, y_pred, labels=all_ids)
    plt.figure(figsize=(14, 12)); plt.imshow(cm, cmap="Blues"); plt.title("Confusion matrix (test set)")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar(); plt.tight_layout()
    plt.savefig(f"{a.out}/confusion_matrix.png", dpi=130); plt.close()
    loss, acc = model.evaluate(test)
    json.dump({"test_accuracy": float(acc), "test_loss": float(loss)}, open(f"{a.out}/metrics.json", "w"))
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()