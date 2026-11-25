from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv(ROOT / "data" / "customer_churn.csv")
X, y = df.drop(columns=["churn"]), df["churn"].astype("int32")
num = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
cat = ["contract_type", "internet_service", "paperless_billing"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y
)
X_train = preprocessor.fit_transform(X_train).astype("float32")
X_test = preprocessor.transform(X_test).astype("float32")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=.2, epochs=60,
                    batch_size=32, callbacks=[stop], verbose=1)

proba = model.predict(X_test, verbose=0).ravel()
pred = (proba >= .5).astype(int)
metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, proba)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "test_rows": int(len(y_test)),
    "dataset_note": "Synthetic educational dataset, not real customer data."
}
model.save(MODEL_DIR / "customer_churn_ann.keras")
joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("\nTEST RESULTS")
print(classification_report(y_test, pred, zero_division=0))
print(json.dumps(metrics, indent=2))
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("ANN Training and Validation Loss")
plt.legend(); plt.tight_layout()
plt.savefig(ROOT / "training_loss.png", dpi=150)
plt.close()

# Training and Validation Accuracy
plt.figure(figsize=(8, 4))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ANN Training and Validation Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig(ROOT / "training_accuracy.png", dpi=150)
plt.close()

print("\nTraining graphs saved successfully!")
print("Model and preprocessing pipeline saved in models/")

# Confusion Matrix Visualization

cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("ANN Confusion Matrix")
plt.tight_layout()

plt.savefig(ROOT / "confusion_matrix.png", dpi=150)
plt.close()

print("Confusion matrix saved successfully!")

# ROC Curve Visualization

fpr, tpr, thresholds = roc_curve(y_test, proba)

plt.figure(figsize=(8, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ANN (AUC = {metrics['roc_auc']:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ANN ROC Curve")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(ROOT / "roc_curve.png", dpi=150)
plt.close()

print("ROC curve saved successfully!")

# Save Training History

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(ROOT / "training_history.json", "w") as file:
    json.dump(history_data, file, indent=4)

print("Training history saved successfully!")