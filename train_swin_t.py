# train_swin_t.py
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# ---- config (edit jika perlu) ----
PROJECT_ROOT = r"C:\DATA JOY\Tugas\SEM 7\DL\TUBES DL"
TRAIN_DIR = os.path.join(PROJECT_ROOT, "clean_train_temp", "train")  # atau data_split/train
VAL_DIR   = os.path.join(PROJECT_ROOT, "clean_train_temp", "val")    # atau data_split/val
OUT_DIR   = os.path.join(PROJECT_ROOT, "swin_t_output")
IMG_SIZE  = 224
BATCH_SIZE = 8   # kecil karena dataset & kemungkinan GPU memori terbatas
NUM_EPOCHS = 12
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
BACKBONE = "swin_tiny_patch4_window7_224"  # timm name for Swin-Tiny
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# ---- transforms ----
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

def main():
    # safe num_workers / pin_memory settings for Windows / no GPU
    nw = min(NUM_WORKERS, os.cpu_count() or 1)
    pin = True if torch.cuda.is_available() else False

    # ---- datasets & loaders ----
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_tf)

    num_classes = len(train_ds.classes)
    print("Num classes:", num_classes)
    print("Train samples:", len(train_ds), "Val samples:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=nw, pin_memory=pin)

    # ---- model ----
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=num_classes)
    model.to(DEVICE)

    # training strategy: freeze backbone then unfreeze
    def freeze_backbone(m):
        for name, p in m.named_parameters():
            if 'head' not in name and 'classifier' not in name:
                p.requires_grad = False

    # if you want to first freeze, uncomment the next line:
    # freeze_backbone(model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # scheduler (reduce LR on plateau) — remove verbose for compatibility
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ---- training loop ----
    best_val_acc = 0.0
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - train")
        for inputs, labels in loop:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)
            loop.set_postfix(loss=running_loss/total, acc=running_corrects/total)

        epoch_loss = running_loss / total
        epoch_acc  = running_corrects / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - val"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_total += inputs.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_epoch_loss = val_loss / val_total if val_total>0 else 0.0
        val_epoch_acc  = val_corrects / val_total if val_total>0 else 0.0
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc)

        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f} val_loss={val_epoch_loss:.4f} val_acc={val_epoch_acc:.4f}")

        # scheduler step using val_loss
        scheduler.step(val_epoch_loss)

        # save best
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            save_path = os.path.join(OUT_DIR, "best_swin_t.pth")
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "classes": train_ds.classes,
                        "best_val_acc": best_val_acc}, save_path)
            print("Saved best model to", save_path)

        # optional: unfreeze after couple of epochs
        if epoch == 4:
            print("=> Unfreezing backbone for fine-tuning...")
            for p in model.parameters():
                p.requires_grad = True
            # reset optimizer to include all params
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR/5, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ---- final evaluation & report ----
    # load best model
    best_ckpt = os.path.join(OUT_DIR, "best_swin_t.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print("Loaded best checkpoint (val_acc=%.4f) from epoch %s" % (ckpt.get("best_val_acc",0.0), ckpt.get("epoch","?")))

    # compute detailed metrics
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    print("Final Val Accuracy:", acc)
    report = classification_report(y_true, y_pred, target_names=train_ds.classes, digits=4)
    print(report)

    # save report
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Val accuracy: {acc}\n\n")
        f.write(report)

    # plot loss/acc
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend(); plt.title("Loss"); plt.savefig(os.path.join(OUT_DIR, "loss_curve.png")); plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend(); plt.title("Accuracy"); plt.savefig(os.path.join(OUT_DIR, "acc_curve.png")); plt.close()

    print("Done. Outputs saved to", OUT_DIR)

if __name__ == "__main__":
    # Windows multiprocessing protection
    multiprocessing.freeze_support()
    main()
