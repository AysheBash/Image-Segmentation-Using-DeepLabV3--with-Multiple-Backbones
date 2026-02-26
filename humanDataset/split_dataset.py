import os
import shutil
import random
from glob import glob
from tqdm import tqdm

# ====== AYARLAR ======
SOURCE_DIR = r"/Users/aysebas/PycharmProjects/humanDataset/people_segmentation"
OUTPUT_DIR = r"/Users/aysebas/PycharmProjects/humanDataset/new_data"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ====== OUTPUT KLASÖR YAPISI ======
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks", split), exist_ok=True)

# ====== Tüm görüntüleri topla ======
all_images = sorted(glob(os.path.join(SOURCE_DIR, "images", "*.png")))

# Shuffle
random.shuffle(all_images)

# Split sayıları
n_total = len(all_images)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

splits = {
    "train": all_images[:n_train],
    "val": all_images[n_train:n_train + n_val],
    "test": all_images[n_train + n_val:]
}

# ====== Dosyaları kopyala ======
for split, image_list in splits.items():
    print(f"\nProcessing {split.upper()} set...")
    out_img_dir = os.path.join(OUTPUT_DIR, "images", split)
    out_mask_dir = os.path.join(OUTPUT_DIR, "masks", split)

    for idx, img_path in enumerate(tqdm(image_list, leave=False)):
        # mask ismi image ile birebir eşleşiyor
        mask_path = os.path.join(SOURCE_DIR, "masks", os.path.basename(img_path))

        if not os.path.exists(mask_path):
            print(f"⚠️ Mask not found for {img_path}, skipping.")
            continue

        out_name = f"{idx}.png"

        shutil.copy2(img_path, os.path.join(out_img_dir, out_name))
        shutil.copy2(mask_path, os.path.join(out_mask_dir, out_name))

print("\n✅ DONE")
print("📁 Dataset ready for Human Segmentation (U-Net, DeepLabV3+, etc.)")
print("📁 Output path:", OUTPUT_DIR)
