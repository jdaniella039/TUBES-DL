import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# konstanta
IMG_SIZE = (224, 224)
BATCH = 16
SEED = 42
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}

# candidate lokasi dataset (prioritas)
CANDIDATES = [
    ("data_split/train", "data_split/val"),
    ("clean_train_temp/train", "clean_train_temp/val"),
    ("Train", "Train")  # last resort (but requires subfolders per class)
]

def count_images_in_dir(d):
    if not os.path.isdir(d):
        return 0
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                total += 1
    return total

# pilih pasangan train/val yang valid
chosen = None
for tdir, vdir in CANDIDATES:
    tcount = count_images_in_dir(tdir)
    vcount = count_images_in_dir(vdir)
    # prefer pairs where both have images (val may be zero if user didn't split yet)
    if tcount > 0 and vcount > 0:
        chosen = (tdir, vdir)
        break
    # allow case where train has images and val missing -> instruct user to split
    if tcount > 0 and vcount == 0 and tdir != vdir:
        chosen = (tdir, vdir)  # still choose but will warn later
        break

if chosen is None:
    print("ERROR: Tidak menemukan data untuk training/validation.")
    print("Pastikan Anda sudah menjalankan cleaning + split. Contoh:")
    print("  python .\\clean_images.py")
    print("  python .\\split_dataset.py")
    print("Atau letakkan data ter-split di data_split/train dan data_split/val")
    sys.exit(1)

TRAIN_DIR, VAL_DIR = chosen
train_total = count_images_in_dir(TRAIN_DIR)
val_total = count_images_in_dir(VAL_DIR)

if train_total == 0:
    print(f"ERROR: Tidak ada gambar di folder TRAIN: {TRAIN_DIR}")
    sys.exit(1)

if VAL_DIR == TRAIN_DIR and val_total == 0:
    print(f"WARNING: Validation folder same as train ({TRAIN_DIR}) and no explicit val found.")
    print("Please run split_dataset.py to create data_split/train and data_split/val,")
    print("or provide a validation folder. Proceeding but training will fail if validation required.")
    # continue but may fail at fit if validation_generator empty

print("Using TRAIN_DIR:", TRAIN_DIR, " (images:", train_total, ")")
print("Using VAL_DIR  :", VAL_DIR, " (images:", val_total, ")")

# DEBUG: periksa isi folder sebelum flow_from_directory
print("DEBUG: TRAIN_DIR contents:", TRAIN_DIR, os.path.isdir(TRAIN_DIR))
if os.path.isdir(TRAIN_DIR):
    print("Subfolders:", sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]))
    for c in sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]):
        p = os.path.join(TRAIN_DIR, c)
        print(f"  {c}: {len([f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in VALID_EXTS])} images")
print("DEBUG: VAL_DIR contents:", VAL_DIR, os.path.isdir(VAL_DIR))
if os.path.isdir(VAL_DIR):
    print("Subfolders:", sorted([d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))]))
    for c in sorted([d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))]):
        p = os.path.join(VAL_DIR, c)
        print(f"  {c}: {len([f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in VALID_EXTS])} images")

# ========================
# IMAGE GENERATOR
# ========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=(0.8,1.2),
    zoom_range=0.05,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False,
    seed=SEED
)

num_classes = train_generator.num_classes
print("Classes:", num_classes)
print("Train samples:", train_generator.samples)
print("Val samples:", validation_generator.samples)

# ========================
# RESNET50 MODEL
# ========================
base = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3),
    pooling='avg'
)
base.trainable = False  # warm-up

inputs = layers.Input(shape=(224,224,3))
x = inputs
x = base(x, training=False)
x = layers.Dense(512)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========================
# CALLBACKS
# ========================
checkpoint = callbacks.ModelCheckpoint(
    "best_resnet_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

earlystop = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
)

# ========================
# TRAIN WARM-UP
# ========================
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint, earlystop]
)

# ========================
# UNFREEZE FOR FINE-TUNE
# ========================
for layer in base.layers[-40:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[checkpoint, earlystop]
)

# ========================
# EVALUATION
# ========================
print("\nEvaluating model...")

validation_generator.reset()
y_true = validation_generator.classes
y_prob = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

class_names = [k for k,_ in sorted(validation_generator.class_indices.items(), key=lambda kv: kv[1])]
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

