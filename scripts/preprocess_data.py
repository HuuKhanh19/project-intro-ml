"""
Preprocess and organize dataset with undersampling strategy
Convert DICOM to PNG, organize by labels, split train/val/test
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pydicom
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Target class distribution (after undersampling)
TARGET_SAMPLES = {
    'Normal': 5000,
    'Pneumonia': 4500,
    'COVID': 3616,  # Keep all
    'Tuberculosis': 700,  # Keep all
    'Pneumothorax': 2379  # Keep all
}

print("=" * 80)
print("DATA PREPROCESSING - Undersampling + Organization")
print("=" * 80)

# Create processed directories
for split in ['train', 'val', 'test']:
    for label in TARGET_SAMPLES.keys():
        (PROCESSED_DIR / split / label).mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def convert_dicom_to_png(dicom_path, output_path):
    """Convert DICOM to PNG with proper windowing"""
    try:
        dcm = pydicom.dcmread(dicom_path)
        img_array = dcm.pixel_array
        
        # Apply lung window (level=-600, width=1500)
        img_array = np.clip(img_array, -600-750, -600+750)
        img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        
        img = Image.fromarray(img_array).convert('L')
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")
        return False

def copy_image(src_path, dest_path):
    """Copy image file (PNG/JPG)"""
    try:
        shutil.copy2(src_path, dest_path)
        return True
    except Exception as e:
        print(f"Error copying {src_path}: {e}")
        return False

def undersample_list(id_list, target_size):
    """Random undersample to target size"""
    if len(id_list) <= target_size:
        return id_list
    return random.sample(id_list, target_size)

def split_by_ratio(id_list, train_ratio=0.7, val_ratio=0.15):
    """Split list into train/val/test"""
    random.shuffle(id_list)
    
    n_total = len(id_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train = id_list[:n_train]
    val = id_list[n_train:n_train+n_val]
    test = id_list[n_train+n_val:]
    
    return train, val, test

# ============================================================================
# 1. Process RSNA (Normal + Pneumonia)
# ============================================================================
print("\n[1/5] Processing RSNA...")

rsna_dir = RAW_DIR / "rsna-pneumonia"
rsna_images_dir = rsna_dir / "stage_2_train_images"

# Load filtered IDs
with open(rsna_dir / "normal_ids.txt") as f:
    normal_ids = [line.strip() for line in f]

with open(rsna_dir / "pneumonia_ids.txt") as f:
    pneumonia_ids = [line.strip() for line in f]

# Undersample
normal_ids = undersample_list(normal_ids, TARGET_SAMPLES['Normal'])
pneumonia_ids = undersample_list(pneumonia_ids, TARGET_SAMPLES['Pneumonia'])

print(f"  Normal: {len(normal_ids)} (after undersampling)")
print(f"  Pneumonia: {len(pneumonia_ids)} (after undersampling)")

# Process Normal
normal_train, normal_val, normal_test = split_by_ratio(normal_ids)
for split_name, split_ids in [('train', normal_train), ('val', normal_val), ('test', normal_test)]:
    print(f"  Processing Normal {split_name}: {len(split_ids)} images...")
    for img_id in tqdm(split_ids, desc=f"  Normal {split_name}"):
        src = rsna_images_dir / f"{img_id}.dcm"
        dst = PROCESSED_DIR / split_name / "Normal" / f"{img_id}.png"
        convert_dicom_to_png(src, dst)

# Process Pneumonia
pneumonia_train, pneumonia_val, pneumonia_test = split_by_ratio(pneumonia_ids)
for split_name, split_ids in [('train', pneumonia_train), ('val', pneumonia_val), ('test', pneumonia_test)]:
    print(f"  Processing Pneumonia {split_name}: {len(split_ids)} images...")
    for img_id in tqdm(split_ids, desc=f"  Pneumonia {split_name}"):
        src = rsna_images_dir / f"{img_id}.dcm"
        dst = PROCESSED_DIR / split_name / "Pneumonia" / f"{img_id}.png"
        convert_dicom_to_png(src, dst)

# ============================================================================
# 2. Process COVID-19
# ============================================================================
print("\n[2/5] Processing COVID-19...")

covid_dir = RAW_DIR / "covid19" / "COVID-19_Radiography_Dataset" / "COVID" / "images"
covid_images = list(covid_dir.glob("*.png"))

print(f"  Total COVID images: {len(covid_images)}")

covid_train, covid_val, covid_test = split_by_ratio(covid_images)

for split_name, split_imgs in [('train', covid_train), ('val', covid_val), ('test', covid_test)]:
    print(f"  Processing COVID {split_name}: {len(split_imgs)} images...")
    for img_path in tqdm(split_imgs, desc=f"  COVID {split_name}"):
        dst = PROCESSED_DIR / split_name / "COVID" / img_path.name
        copy_image(img_path, dst)

# ============================================================================
# 3. Process Tuberculosis
# ============================================================================
print("\n[3/5] Processing Tuberculosis...")

tb_dir = RAW_DIR / "tuberculosis" / "TB_Chest_Radiography_Database" / "Tuberculosis"
tb_images = list(tb_dir.glob("*.png"))

print(f"  Total TB images: {len(tb_images)}")

tb_train, tb_val, tb_test = split_by_ratio(tb_images)

for split_name, split_imgs in [('train', tb_train), ('val', tb_val), ('test', tb_test)]:
    print(f"  Processing TB {split_name}: {len(split_imgs)} images...")
    for img_path in tqdm(split_imgs, desc=f"  TB {split_name}"):
        dst = PROCESSED_DIR / split_name / "Tuberculosis" / img_path.name
        copy_image(img_path, dst)

# ============================================================================
# 4. Process Pneumothorax
# ============================================================================
print("\n[4/5] Processing Pneumothorax...")

ptx_dir = RAW_DIR / "pneumothorax"

# Load positive IDs
with open(ptx_dir / "positive_ids.txt") as f:
    ptx_positive_ids = [line.strip() for line in f]

# Find images in png_images folder
ptx_images_dir = ptx_dir / "siim-acr-pneumothorax" / "png_images"
available_ptx = []

for img_id in ptx_positive_ids:
    img_path = ptx_images_dir / img_id
    if img_path.exists():
        available_ptx.append(img_path)

print(f"  Total Pneumothorax images: {len(available_ptx)}")

ptx_train, ptx_val, ptx_test = split_by_ratio(available_ptx)

for split_name, split_imgs in [('train', ptx_train), ('val', ptx_val), ('test', ptx_test)]:
    print(f"  Processing Pneumothorax {split_name}: {len(split_imgs)} images...")
    for img_path in tqdm(split_imgs, desc=f"  Pneumothorax {split_name}"):
        dst = PROCESSED_DIR / split_name / "Pneumothorax" / img_path.name
        copy_image(img_path, dst)

# ============================================================================
# 5. Generate Statistics
# ============================================================================
print("\n[5/5] Generating statistics...")

stats = defaultdict(lambda: defaultdict(int))

for split in ['train', 'val', 'test']:
    for label in TARGET_SAMPLES.keys():
        count = len(list((PROCESSED_DIR / split / label).glob("*.*")))
        stats[split][label] = count

# Save to CSV
stats_df = pd.DataFrame(stats).T
stats_df.to_csv(PROCESSED_DIR / "dataset_statistics.csv")

# Print summary
print("\n" + "=" * 80)
print("PREPROCESSING COMPLETED")
print("=" * 80)
print("\nDataset Distribution:")
print(stats_df)

print("\n" + "=" * 80)
print("NEXT: Run training with 4 models")
print("=" * 80)