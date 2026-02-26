import os
import cv2
import csv
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from model import HumanSegmentationDataset, calculate_metrics

# ================= GLOBAL PARAMETERS =================
H = 256
W = 256
BATCH_SIZE = 1
THRESHOLD = 0.4

# ================= BASE DIRECTORIES =================
BASE_OUTPUT_DIR = "outputs"          # train.py çıktıları
TEST_RESULTS_DIR = "test_results"    # test.py çıktıları

CSV_DIR = os.path.join(TEST_RESULTS_DIR, "csv")
VIS_DIR = os.path.join(TEST_RESULTS_DIR, "visuals")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# ================= TEST SCRIPT =================
if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device Used: {device}")

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    # ================= TRANSFORM =================
    val_transform = albu.Compose([
        albu.Resize(H, W),
        albu.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])

    # ================= DATASET =================
    dataset_path = "/Users/aysebas/PycharmProjects/humanDataset/new_data"

    test_imgs_dir = os.path.join(dataset_path, "images", "test")
    test_masks_dir = os.path.join(dataset_path, "masks", "test")

    test_imgs = sorted(os.path.join(test_imgs_dir, f) for f in os.listdir(test_imgs_dir))
    test_masks = sorted(os.path.join(test_masks_dir, f) for f in os.listdir(test_masks_dir))

    print(f"Total Test Images: {len(test_imgs)}")

    test_set = HumanSegmentationDataset(test_imgs, test_masks, transform=val_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # ================= EXPERIMENT SETTINGS =================
    epoch_list = [30, 50]
    backbones = ["resnet50", "resnet101", "mobilenet_v2"]

    # ================= TEST LOOP =================
    for epochs in epoch_list:
        for backbone in backbones:

            print(f"\n===== Testing | Backbone: {backbone} | Epochs: {epochs} =====")

            # -------- MODEL PATH --------
            model_path = os.path.join(
                BASE_OUTPUT_DIR,
                f"epochs_{epochs}",
                backbone,
                "best_model.pth"
            )

            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_path}")
                continue

            # -------- MODEL --------
            model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                decoder={'dropout': 0.3, 'aspp_dropout': 0.5}
            ).to(device)

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # -------- OUTPUT DIRS --------
            vis_epoch_dir = os.path.join(VIS_DIR, f"epochs_{epochs}", backbone)
            os.makedirs(vis_epoch_dir, exist_ok=True)

            csv_file = os.path.join(
                CSV_DIR,
                f"epochs_{epochs}_{backbone}_test_results.csv"
            )

            with open(csv_file, "w", newline="") as f:
                csv.writer(f).writerow([
                    "Image_Name", "Precision", "Recall",
                    "F1_Score", "IoU", "Specificity", "Accuracy"
                ])

            precision_list, recall_list, f1_list = [], [], []
            iou_list, specificity_list, accuracy_list = [], [], []

            # ================= INFERENCE =================
            start_time = time.time()
            with torch.no_grad():
                for images, masks, paths in test_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)

                    for i in range(images.size(0)):
                        p, r, f1, iou, s, acc = calculate_metrics(outputs[i], masks[i])

                        precision_list.append(p)
                        recall_list.append(r)
                        f1_list.append(f1)
                        iou_list.append(iou)
                        specificity_list.append(s)
                        accuracy_list.append(acc)

                        image_name = os.path.basename(paths[i])

                        with open(csv_file, "a", newline="") as f:
                            csv.writer(f).writerow(
                                [image_name, p, r, f1, iou, s, acc]
                            )

                        # -------- VISUALIZATION --------
                        original_img = cv2.cvtColor(
                            cv2.imread(paths[i]),
                            cv2.COLOR_BGR2RGB
                        )

                        pred_mask = torch.sigmoid(outputs[i]).cpu().squeeze().numpy()
                        pred_mask = (pred_mask > THRESHOLD).astype(np.uint8)

                        gt_mask = masks[i].cpu().squeeze().numpy()
                        gt_mask = (gt_mask > THRESHOLD).astype(np.uint8)

                        pred_mask = cv2.resize(
                            pred_mask,
                            (original_img.shape[1], original_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

                        gt_mask = cv2.resize(
                            gt_mask,
                            (original_img.shape[1], original_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

                        overlay = (original_img * np.stack([pred_mask]*3, axis=-1)).astype(np.uint8)

                        plt.figure(figsize=(16, 4))
                        plt.subplot(1, 4, 1); plt.imshow(original_img); plt.title("Original"); plt.axis("off")
                        plt.subplot(1, 4, 2); plt.imshow(gt_mask, cmap="gray"); plt.title("GT"); plt.axis("off")
                        plt.subplot(1, 4, 3); plt.imshow(pred_mask, cmap="gray"); plt.title("Prediction"); plt.axis("off")
                        plt.subplot(1, 4, 4); plt.imshow(overlay); plt.title("Overlay"); plt.axis("off")

                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(
                                vis_epoch_dir,
                                os.path.splitext(image_name)[0] + ".png"
                            ),
                            dpi=200
                        )
                        plt.close()

            # ================= SUMMARY =================
            print(f"Test finished in {(time.time() - start_time):.2f} seconds")
            print(f"IoU        : {np.mean(iou_list):.4f}")
            print(f"Precision  : {np.mean(precision_list):.4f}")
            print(f"Recall     : {np.mean(recall_list):.4f}")
            print(f"Specificity: {np.mean(specificity_list):.4f}")
            print(f"F1-Score   : {np.mean(f1_list):.4f}")
            print(f"Accuracy   : {np.mean(accuracy_list):.4f}")
            print("======================================")
