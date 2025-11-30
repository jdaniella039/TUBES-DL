# embedding_knn_eval.py
import os, numpy as np, shutil, sys
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------- CONFIG --------
PROJECT_ROOT = r"C:\DATA JOY\Tugas\SEM 7\DL\TUBES DL"
# gunakan folder yang memang berisi gambar (clean_train_temp)
TRAIN_DIR = os.path.join(PROJECT_ROOT, "clean_train_temp", "train")
VAL_DIR   = os.path.join(PROJECT_ROOT, "clean_train_temp", "val")
OUT_DIR   = os.path.join(PROJECT_ROOT, "embedding_eval")
IMG_SIZE  = (224,224)
BATCH     = 16


os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"misclassified"), exist_ok=True)

# -------- UTIL --------
def iter_images(folder):
    # yields (filepath, class_name)
    if not os.path.isdir(folder):
        return
    for cls in sorted(os.listdir(folder)):
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in sorted(os.listdir(cls_path)):
            fp = os.path.join(cls_path, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in ('.jpg','.jpeg','.png','.bmp','.webp','.tiff'):
                yield fp, cls

def load_and_preprocess(fp):
    img = load_img(fp, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    return arr

def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    cnt = 0
    for _ in iter_images(folder):
        cnt += 1
    return cnt

# -------- VALIDATE DATA --------
train_count = count_images(TRAIN_DIR)
val_count = count_images(VAL_DIR)
if train_count == 0:
    print("ERROR: No training images found.")
    print(f"Checked TRAIN_DIR: {TRAIN_DIR}")
    print("Ensure images are organized as data_split/train/<class_name>/*.jpg")
    sys.exit(1)
if val_count == 0:
    print("ERROR: No validation images found.")
    print(f"Checked VAL_DIR: {VAL_DIR}")
    print("Ensure images are organized as data_split/val/<class_name>/*.jpg")
    sys.exit(1)

# -------- MODEL (backbone) --------
print("Loading backbone ResNet50 (imagenet, pooling='avg') ...")
backbone = ResNet50(weights="imagenet", include_top=False, pooling='avg',
                    input_shape=(IMG_SIZE[0],IMG_SIZE[1],3))
print("Backbone ready.")

# -------- EXTRACT EMBEDDINGS (train) --------
train_files = []
train_labels = []
train_embs = []

print("Extracting train embeddings...")
for fp, cls in iter_images(TRAIN_DIR):
    train_files.append(fp)
    train_labels.append(cls)
    arr = load_and_preprocess(fp)
    emb = backbone.predict(arr)
    if emb is None or emb.shape[0] == 0:
        continue
    train_embs.append(emb[0])

if len(train_embs) == 0:
    print("ERROR: No train embeddings were extracted (train_embs is empty).")
    sys.exit(1)

train_embs = np.vstack(train_embs)
train_labels = np.array(train_labels)
print("Train:", train_embs.shape, "samples")

# -------- COMPUTE CENTROIDS PER CLASS --------
classes = np.unique(train_labels)
centroids = {}
for c in classes:
    idx = np.where(train_labels == c)[0]
    if len(idx) == 0:
        continue
    centroids[c] = train_embs[idx].mean(axis=0)

if len(centroids) == 0:
    print("ERROR: No centroids computed (no classes).")
    sys.exit(1)

print("Computed centroids for", len(centroids), "classes.")

# L2-normalize centroids for cosine similarity
def l2norm(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

centroid_names = list(centroids.keys())
centroid_matrix = np.vstack([centroids[c] for c in centroid_names])
centroid_matrix = l2norm(centroid_matrix)

# -------- EXTRACT EMBEDDINGS (val) & PREDICT --------
y_true = []
y_pred = []
val_files = []

print("Extracting val embeddings and predicting...")
for fp, cls in iter_images(VAL_DIR):
    val_files.append(fp)
    y_true.append(cls)
    arr = load_and_preprocess(fp)
    emb = backbone.predict(arr)[0]
    emb = emb.reshape(1,-1)
    emb = l2norm(emb)[0]  # normalize
    sims = centroid_matrix.dot(emb)
    best = np.argmax(sims)
    pred_cls = centroid_names[best]
    y_pred.append(pred_cls)

if len(y_true) == 0:
    print("ERROR: No validation samples processed.")
    sys.exit(1)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -------- METRICS & REPORT --------
acc = accuracy_score(y_true, y_pred)
print("Accuracy (nearest-centroid cosine):", acc)

report = classification_report(y_true, y_pred, digits=4)
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(report)
print("Saved classification report to", os.path.join(OUT_DIR, "classification_report.txt"))
print(report)

# confusion matrix (labels order = classes sorted)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(np.concatenate([y_true, y_pred]))
labels_order = le.classes_

cm = confusion_matrix(y_true, y_pred, labels=labels_order)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (centroid cosine)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()
print("Saved confusion matrix to", os.path.join(OUT_DIR, "confusion_matrix.png"))

# -------- SAVE some misclassified examples to folder for inspection --------
mis_dir = os.path.join(OUT_DIR, "misclassified")
shutil.rmtree(mis_dir, ignore_errors=True)
os.makedirs(mis_dir, exist_ok=True)
count=0
for fp, t, p in zip(val_files, y_true, y_pred):
    if t != p:
        fname = os.path.basename(fp)
        dst = os.path.join(mis_dir, f"true__{t}__pred__{p}__{fname}")
        shutil.copy(fp, dst)
        count += 1
print(f"Saved {count} misclassified images to {mis_dir}")

print("DONE. Check folder:", OUT_DIR)
