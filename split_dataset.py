import os, glob, shutil
from sklearn.model_selection import train_test_split

SRC = "Train"
OUT = "data_split"
VAL_RATIO = 0.25   # 1 dari 4 gambar masuk val
SEED = 42

# Bersihkan dulu folder data_split
if os.path.exists(OUT):
    shutil.rmtree(OUT)

os.makedirs(os.path.join(OUT, "train"), exist_ok=True)
os.makedirs(os.path.join(OUT, "val"), exist_ok=True)

classes = [d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))]
print("Classes:", len(classes))

for cls in classes:
    imgs = glob.glob(os.path.join(SRC, cls, "*"))
    if len(imgs) == 0:
        continue

    # Jika gambar sangat sedikit
    if len(imgs) == 1:
        tr, va = imgs, []
    elif len(imgs) == 2:
        tr = [imgs[0]]
        va = [imgs[1]]
    else:
        tr, va = train_test_split(imgs, test_size=1, random_state=SEED)

    train_dir = os.path.join(OUT, "train", cls)
    val_dir = os.path.join(OUT, "val", cls)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for p in tr: shutil.copy(p, train_dir)
    for p in va: shutil.copy(p, val_dir)

print("DONE! Dataset siap dipakai di folder: data_split")
