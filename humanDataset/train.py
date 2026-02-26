import os
import csv
import time
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch.optim as optim

from albumentations.pytorch import ToTensorV2
import albumentations as albu

from model import HumanSegmentationDataset, calculate_metrics, CombinedLoss, get_images_and_masks

# ================= GLOBAL PARAMETERS =================
H, W = 256, 256
BATCH_SIZE = 12
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device Used: {DEVICE}")

gc.collect()

# ================= OUTPUT DIRECTORY =================
BASE_OUTPUT_DIR = "outputs"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ================= AUGMENTATIONS =================
train_transform = albu.Compose([
    albu.Resize(H, W),
    albu.HorizontalFlip(p=0.5),
    albu.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    albu.Normalize(mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = albu.Compose([
    albu.Resize(H, W),
    albu.Normalize(mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ================= LOAD DATASET =================
dataset_path = "/Users/aysebas/PycharmProjects/humanDataset/new_data"

train_imgs, train_masks = get_images_and_masks(dataset_path, "train")
val_imgs, val_masks = get_images_and_masks(dataset_path, "val")
test_imgs, test_masks = get_images_and_masks(dataset_path, "test")

print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

train_set = HumanSegmentationDataset(train_imgs, train_masks, transform=train_transform)
val_set = HumanSegmentationDataset(val_imgs, val_masks, transform=val_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# ================= TRAINING SETTINGS =================
epoch_list = [30, 50]
backbones = ["resnet50", "resnet101", "mobilenet_v2"]

# ================= TRAINING LOOP =================
for NUMEPOCS in epoch_list:

    epoch_dir = os.path.join(BASE_OUTPUT_DIR, f"epochs_{NUMEPOCS}")
    os.makedirs(epoch_dir, exist_ok=True)

    for backbone in backbones:

        print(f"\n=== Training DeepLabV3Plus | Backbone: {backbone} | Epochs: {NUMEPOCS} ===")

        run_dir = os.path.join(epoch_dir, backbone)
        os.makedirs(run_dir, exist_ok=True)

        # ===== MODEL =====
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            decoder={'dropout': 0.3, 'aspp_dropout': 0.5}
        ).to(DEVICE)

        # ===== LOSS / OPTIMIZER =====
        criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        # ===== CSV LOG =====
        log_file = os.path.join(run_dir, "logs.csv")
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(
                ["Epoch", "Train Loss", "Val Loss",
                 "Train Acc", "Val Acc",
                 "Train IoU", "Val IoU"]
            )

        best_val_iou = 0.0
        patience = 4
        counter = 0
        start_time = time.time()

        # ================= EPOCH LOOP =================
        for epoch in range(NUMEPOCS):

            # -------- TRAIN --------
            model.train()
            train_loss = 0.0
            train_acc_list, train_iou_list = [], []

            for images, masks, _ in train_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                for i in range(images.size(0)):
                    _, _, _, iou, _, acc = calculate_metrics(outputs[i], masks[i])
                    train_acc_list.append(acc)
                    train_iou_list.append(iou)

            train_loss /= len(train_loader)
            train_acc = np.mean(train_acc_list)
            train_iou = np.mean(train_iou_list)

            # -------- VALIDATION --------
            model.eval()
            val_loss = 0.0
            val_acc_list, val_iou_list = [], []

            with torch.no_grad():
                for images, masks, _ in val_loader:
                    images, masks = images.to(DEVICE), masks.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    for i in range(images.size(0)):
                        _, _, _, iou, _, acc = calculate_metrics(outputs[i], masks[i])
                        val_acc_list.append(acc)
                        val_iou_list.append(iou)

            val_loss /= len(val_loader)
            val_acc = np.mean(val_acc_list)
            val_iou = np.mean(val_iou_list)

            scheduler.step(val_loss)

            # -------- LOG --------
            print(
                f"Epoch {epoch+1}/{NUMEPOCS} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}"
            )

            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch+1, train_loss, val_loss,
                     train_acc, val_acc,
                     train_iou, val_iou]
                )

            # -------- SAVE BEST --------
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
                print(f"✅ Best model saved | Val IoU: {val_iou:.4f}")
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    print("⏹ Early stopping applied")
                    break

        print(f"🎯 Finished in {(time.time() - start_time)/60:.2f} minutes")

        # ================= PLOTS =================
        log_df = pd.read_csv(log_file)

        for col, name in [
            (("Train Loss", "Val Loss"), "loss_curve.png"),
            (("Train Acc", "Val Acc"), "accuracy_curve.png"),
            (("Train IoU", "Val IoU"), "iou_curve.png")
        ]:
            plt.figure(figsize=(8, 6))
            plt.plot(log_df["Epoch"], log_df[col[0]], label=col[0])
            plt.plot(log_df["Epoch"], log_df[col[1]], label=col[1])
            plt.xlabel("Epoch")
            plt.ylabel(col[0].split()[1])
            plt.title(f"{backbone.upper()} - {col[0].split()[1]}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, name))
            plt.close()
