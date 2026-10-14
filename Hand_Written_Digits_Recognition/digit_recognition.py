import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

SEED, EPOCHS = 42, 8
keras.utils.set_random_seed(SEED)

# 1. LOAD
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Train:", x_train.shape, "Test:", x_test.shape)

fig, axes = plt.subplots(2, 10, figsize=(14, 3.4))
for d in range(10):
    idx = np.where(y_train == d)[0]
    for r in range(2):
        axes[r, d].imshow(x_train[idx[r]], cmap="gray")
        axes[r, d].set_title(str(d)); axes[r, d].axis("off")
plt.suptitle("MNIST samples"); plt.tight_layout()
plt.savefig("sample_digits.png", dpi=130); plt.close()

# 2. PREPROCESS: normalise to 0-1, add channel axis
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test = (x_test.astype("float32") / 255.0)[..., None]

# 3. BUILD MODELS
def dense_nn():
    return keras.Sequential([
        layers.Input((28, 28, 1)), layers.Flatten(),
        layers.Dense(128, activation="relu"), layers.Dropout(0.2),
        layers.Dense(64, activation="relu"), layers.Dense(10, activation="softmax"),
    ], name="dense_nn")

def cnn():
    return keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dropout(0.3),
        layers.Dense(128, activation="relu"), layers.Dense(10, activation="softmax"),
    ], name="cnn")

# 4. TRAIN
results, histories, models = [], {}, {}
for build in (dense_nn, cnn):
    m = build()
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(f"\n=== Training {m.name} ===")
    h = m.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, validation_split=0.1, verbose=2)
    loss, acc = m.evaluate(x_test, y_test, verbose=0)
    results.append({"Model": m.name, "Parameters": m.count_params(),
                    "Test accuracy": round(acc, 4), "Test loss": round(loss, 4)})
    histories[m.name], models[m.name] = h.history, m

res = pd.DataFrame(results)
print("\n", res.to_string(index=False))
res.to_csv("model_comparison.csv", index=False)

# 5. EVALUATE
best_name = res.sort_values("Test accuracy", ascending=False).iloc[0]["Model"]
best = models[best_name]
probs = best.predict(x_test, verbose=0)
pred = probs.argmax(1)
print(f"\nBest model: {best_name}")
print(classification_report(y_test, pred, digits=4))
best.save("digit_model.keras")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, h in histories.items():
    ax[0].plot(h["val_accuracy"], label=name)
    ax[1].plot(h["val_loss"], label=name)
ax[0].set_title("Validation accuracy"); ax[1].set_title("Validation loss")
for a in ax: a.set_xlabel("epoch"); a.legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion matrix - {best_name}")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=130); plt.close()

# 6. VISUALIZE PREDICTIONS
rng = np.random.default_rng(SEED)
pick = rng.choice(len(x_test), 15, replace=False)
fig, axes = plt.subplots(3, 5, figsize=(10, 6.5))
for ax, i in zip(axes.ravel(), pick):
    ax.imshow(x_test[i, ..., 0], cmap="gray"); ax.axis("off")
    ok = pred[i] == y_test[i]
    ax.set_title(f"pred {pred[i]} ({probs[i].max():.0%})  true {y_test[i]}",
                 color="green" if ok else "red", fontsize=9)
plt.suptitle(f"Sample predictions - {best_name}"); plt.tight_layout()
plt.savefig("predictions.png", dpi=130); plt.close()

wrong = np.where(pred != y_test)[0]
print(f"Misclassified: {len(wrong)} of {len(y_test)}")
fig, axes = plt.subplots(3, 5, figsize=(10, 6.5))
for ax in axes.ravel(): ax.axis("off")
for ax, i in zip(axes.ravel(), wrong[:15]):
    ax.imshow(x_test[i, ..., 0], cmap="gray")
    ax.set_title(f"pred {pred[i]}  true {y_test[i]}", color="red", fontsize=9)
plt.suptitle("Misclassified digits"); plt.tight_layout()
plt.savefig("misclassified.png", dpi=130); plt.close()