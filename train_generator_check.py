from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
import shutil
import random

# optional: PIL digunakan untuk memverifikasi gambar
try:
    from PIL import Image
except Exception:
    print("PIL tidak ditemukan. Install dengan: pip install pillow")
    sys.exit(1)

DATA_DIR = "Train"
IMG_SIZE = (224, 224)
BATCH = 32
VAL_SPLIT = 0.2
SEED = 42
MIN_IMAGES_PER_CLASS = 2  # kalau kelas punya <2 gambar, pindahkan ke invalid

# resolve path relatif ke file skrip (berjalan dengan VSCode/PowerShell)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = DATA_DIR if os.path.isabs(DATA_DIR) else os.path.join(BASE_DIR, DATA_DIR)
random.seed(SEED)

def resolve_nested_dataset_root(path):
    subs = [d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and not d.startswith('.') and d != '__MACOSX']
    # kalau hanya 1 subfolder dan subfolder itu punya subfolder, turun satu level
    if len(subs) == 1:
        candidate = os.path.join(path, subs[0])
        inner_subs = [d for d in os.listdir(candidate)
                      if os.path.isdir(os.path.join(candidate, d)) and not d.startswith('.') and d != '__MACOSX']
        if inner_subs:
            return candidate
    return path

if not os.path.isdir(DATA_DIR):
    print(f"ERROR: Folder dataset tidak ditemukan: {DATA_DIR}")
    print("Pastikan folder ada dan berisi subfolder per kelas (mis. Train/class1, Train/class2).")
    sys.exit(1)

# jika ada extra level (mis. Train/Train/...), turun
DATA_DIR = resolve_nested_dataset_root(DATA_DIR)

# folder untuk file tidak valid / terlalu sedikit
INVALID_DIR = os.path.join(BASE_DIR, "invalid_images")
os.makedirs(INVALID_DIR, exist_ok=True)

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp'}

def is_image_valid(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def clean_and_filter_classes(src_root):
    kept_classes = []
    moved = 0
    for name in os.listdir(src_root):
        class_path = os.path.join(src_root, name)
        if not os.path.isdir(class_path) or name.startswith('.') or name == '__MACOSX':
            continue
        # periksa file gambar valid di folder kelas ini
        valid_files = []
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            if os.path.isdir(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTS:
                # pindahkan non-image
                dest = os.path.join(INVALID_DIR, name)
                os.makedirs(dest, exist_ok=True)
                shutil.move(fpath, os.path.join(dest, fname))
                moved += 1
                continue
            if is_image_valid(fpath):
                valid_files.append(fname)
            else:
                dest = os.path.join(INVALID_DIR, name)
                os.makedirs(dest, exist_ok=True)
                shutil.move(fpath, os.path.join(dest, fname))
                moved += 1
        if len(valid_files) >= MIN_IMAGES_PER_CLASS:
            kept_classes.append((name, valid_files))
        else:
            # pindahkan seluruh sisa file kelas dengan sedikit gambar
            if valid_files:
                dest = os.path.join(INVALID_DIR, name)
                os.makedirs(dest, exist_ok=True)
                for fname in valid_files:
                    shutil.move(os.path.join(class_path, fname), os.path.join(dest, fname))
                    moved += 1
    return kept_classes, moved

kept_classes, moved_count = clean_and_filter_classes(DATA_DIR)
if moved_count:
    print(f"Pindah {moved_count} file tidak valid/korup/terlalu_sedikit ke: {INVALID_DIR}")

if not kept_classes or len(kept_classes) < 2:
    print(f"ERROR: Hanya ditemukan {len(kept_classes)} kelas valid setelah pembersihan.")
    print("Susun dataset menjadi Train/<class_name>/*.jpg (setiap kelas minimal 2 gambar).")
    sys.exit(1)

# buat folder sementara hanya berisi kelas yang valid, lalu split per-class ke train/val (pastikan 1 val bila memungkinkan)
CLEAN_DIR = os.path.join(BASE_DIR, "clean_train_temp")
if os.path.exists(CLEAN_DIR):
    shutil.rmtree(CLEAN_DIR)
os.makedirs(CLEAN_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(CLEAN_DIR, "train")
VAL_DIR = os.path.join(CLEAN_DIR, "val")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

for class_name, files in kept_classes:
    src_class = os.path.join(DATA_DIR, class_name)
    train_dst = os.path.join(TRAIN_DIR, class_name)
    val_dst = os.path.join(VAL_DIR, class_name)
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(val_dst, exist_ok=True)

    # deterministic shuffle
    files_sorted = sorted(files)
    random.shuffle(files_sorted)

    n = len(files_sorted)
    if n == 1:
        tr_files = files_sorted
        val_files = []
    elif n == 2:
        tr_files = files_sorted[:1]
        val_files = files_sorted[1:]
    else:
        k = max(1, int(n * VAL_SPLIT))
        if k >= n:
            k = 1
        val_files = files_sorted[:k]
        tr_files = files_sorted[k:]

    for fname in tr_files:
        shutil.copy2(os.path.join(src_class, fname), os.path.join(train_dst, fname))
    for fname in val_files:
        shutil.copy2(os.path.join(src_class, fname), os.path.join(val_dst, fname))

# debug output
class_counts = {}
for class_name, _ in kept_classes:
    train_count = len(os.listdir(os.path.join(TRAIN_DIR, class_name)))
    val_count = len(os.listdir(os.path.join(VAL_DIR, class_name)))
    class_counts[class_name] = {"train": train_count, "val": val_count}
print("Detected class folders (kept):", list(class_counts.keys()))
print("Images per class (train/val):", class_counts)
total_images = sum(v["train"] + v["val"] for v in class_counts.values())
total_val = sum(v["val"] for v in class_counts.values())
print("Total images (after cleaning):", total_images, "  Val samples:", total_val)

# generator dari folder bersih (explicit train/val dirs)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=(0.8, 1.2),
    shear_range=0.02,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

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

print("Kelas terdeteksi oleh generator:", train_generator.class_indices)
print("Train samples:", train_generator.samples)
print("Val samples:", validation_generator.samples)

try:
    x, y = next(train_generator)
    print("Batch shapes:", x.shape, y.shape)
except StopIteration:
    print("ERROR: Generator tidak menghasilkan batch. Periksa batch_size dan total gambar.")
except Exception as e:
    print("ERROR saat mengambil batch:", str(e))
