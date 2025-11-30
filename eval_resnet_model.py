# eval_resnet_model.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.applications.resnet import preprocess_input

# --- CONFIG (ubah jika perlu) ---
MODEL_PATH = "best_resnet_model.h5"   # ubah kalau model kamu di tempat lain
VAL_DIR    = r"clean_train_temp\val"  # folder val yang kamu pakai (relatif ke cwd)
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
OUT_DIR    = "resnet_eval"
# -------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Model loaded.")

# We will use flow_from_directory to get filenames + true labels, but let ImageDataGenerator apply preprocess_input
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # apply preprocessing inside generator

gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Map index -> class name
idx_to_class = {v:k for k,v in gen.class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

n = gen.samples
print(f"Found {n} validation images, {len(class_names)} classes.")

y_true_idxs = gen.classes  # integer labels in same order as filenames
filepaths = [os.path.join(gen.directory, p) for p in gen.filenames]

# Predict in one call (generator yields preprocessed batches)
print("Running predictions...")
steps = int(np.ceil(n / BATCH_SIZE))
preds = model.predict(gen, steps=steps, verbose=1)
probs = preds if preds.ndim == 2 else preds.reshape(len(preds), -1)
y_pred_idxs = np.argmax(probs, axis=1)
y_pred_probs = np.max(probs, axis=1)

y_pred_idxs = np.array(y_pred_idxs)
y_pred_probs = np.array(y_pred_probs)
y_true_idxs = np.array(y_true_idxs)

acc = accuracy_score(y_true_idxs, y_pred_idxs)
print(f"Accuracy on VAL set: {acc:.4f}")

# classification report with class names
report = classification_report(y_true_idxs, y_pred_idxs, target_names=class_names, digits=4)
print(report)
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(report)
print("Saved classification_report.txt")

# confusion matrix
cm = confusion_matrix(y_true_idxs, y_pred_idxs)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (ResNet)")
plt.tight_layout()
out_cm = os.path.join(OUT_DIR, "confusion_matrix_resnet.png")
plt.savefig(out_cm, dpi=200)
plt.close()
print("Saved", out_cm)

# save per-sample predictions
df = pd.DataFrame({
    "filepath": filepaths,
    "true_idx": y_true_idxs,
    "true_label": [idx_to_class[i] for i in y_true_idxs],
    "pred_idx": y_pred_idxs,
    "pred_label": [idx_to_class[i] for i in y_pred_idxs],
    "confidence": y_pred_probs
})
csv_path = os.path.join(OUT_DIR, "predictions.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")
print("Saved", csv_path)

print("DONE. Outputs in folder:", OUT_DIR)
