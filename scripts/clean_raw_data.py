"""
Clean raw datasets following Zero-Overlap strategy
Remove duplicate/unnecessary folders and filter images
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")

print("=" * 80)
print("CLEANING RAW DATASETS - Zero Overlap Strategy")
print("=" * 80)

# ============================================================================
# 1. RSNA: Filter Normal + Pneumonia
# ============================================================================
print("\n[1/4] Processing RSNA Pneumonia...")
rsna_dir = RAW_DIR / "rsna-pneumonia"

# Load metadata
train_labels = pd.read_csv(rsna_dir / "stage_2_train_labels.csv")
detailed_info = pd.read_csv(rsna_dir / "stage_2_detailed_class_info.csv")

# Filter Pneumonia (Target = 1)
pneumonia_ids = train_labels[train_labels['Target'] == 1]['patientId'].unique()
print(f"  Pneumonia positive: {len(pneumonia_ids)} images")

# Filter Normal (class = 'Normal', NOT 'No Lung Opacity / Not Normal')
normal_ids = detailed_info[detailed_info['class'] == 'Normal']['patientId'].unique()
print(f"  Normal (clean): {len(normal_ids)} images")

# Save filtered IDs
with open(rsna_dir / "pneumonia_ids.txt", "w") as f:
    for pid in pneumonia_ids:
        f.write(f"{pid}\n")

with open(rsna_dir / "normal_ids.txt", "w") as f:
    for nid in normal_ids:
        f.write(f"{nid}\n")

print(f"  ✓ Saved filtered IDs")

# ============================================================================
# 2. COVID-19: Keep only COVID folder
# ============================================================================
print("\n[2/4] Processing COVID-19...")
covid_dir = RAW_DIR / "covid19"

covid_dataset = covid_dir / "COVID-19_Radiography_Dataset"
covid_images = covid_dataset / "COVID" / "images"

if covid_images.exists():
    num_covid = len(list(covid_images.glob("*.png")))
    print(f"  COVID-19 images: {num_covid}")
    
    # Remove other folders to avoid overlap
    folders_to_remove = ["Normal", "Viral Pneumonia", "Lung_Opacity"]
    for folder in folders_to_remove:
        folder_path = covid_dataset / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
            print(f"  ✓ Removed {folder} (overlap with RSNA)")
else:
    print("  ✗ COVID folder not found!")

# ============================================================================
# 3. Tuberculosis: Keep only Tuberculosis folder
# ============================================================================
print("\n[3/4] Processing Tuberculosis...")
tb_dir = RAW_DIR / "tuberculosis"

tb_dataset = tb_dir / "TB_Chest_Radiography_Database"
tb_images = tb_dataset / "Tuberculosis"

if tb_images.exists():
    num_tb = len(list(tb_images.glob("*.png")))
    print(f"  Tuberculosis images: {num_tb}")
    
    # Remove Normal folder
    normal_folder = tb_dataset / "Normal"
    if normal_folder.exists():
        shutil.rmtree(normal_folder)
        print(f"  ✓ Removed Normal (overlap with RSNA)")
else:
    print("  ✗ Tuberculosis folder not found!")

# ============================================================================
# 4. Pneumothorax: Filter positive cases only
# ============================================================================
print("\n[4/4] Processing Pneumothorax...")
ptx_dir = RAW_DIR / "pneumothorax"

# Try multiple possible CSV filenames
csv_files = ["train-rle.csv", "stage_1_train_images.csv", "siim-acr-pneumothorax/stage_1_train_images.csv"]
train_csv = None

for csv_name in csv_files:
    csv_path = ptx_dir / csv_name
    if csv_path.exists():
        train_csv = csv_path
        break

if train_csv:
    ptx_df = pd.read_csv(train_csv)
    print(f"  Found CSV: {train_csv.name}")
    print(f"  Columns: {list(ptx_df.columns)}")
    
    # Filter based on has_pneumo column (1 = has disease)
    if 'has_pneumo' in ptx_df.columns:
        positive_df = ptx_df[ptx_df['has_pneumo'] == 1]
        positive_ids = positive_df['new_filename'].unique()
    else:
        print("  ✗ 'has_pneumo' column not found!")
        positive_ids = []
    
    print(f"  Pneumothorax positive: {len(positive_ids)} images")
    
    # Save filtered IDs
    with open(ptx_dir / "positive_ids.txt", "w") as f:
        for img_id in positive_ids:
            f.write(f"{img_id}\n")
    
    print(f"  ✓ Saved filtered IDs")
else:
    print("  ✗ No CSV file found!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("CLEANING COMPLETED")
print("=" * 80)
print("\nDataset structure (Zero Overlap):")
print("  RSNA:")
print(f"    - Normal: {len(normal_ids)} images")
print(f"    - Pneumonia: {len(pneumonia_ids)} images")
print(f"  COVID-19: {num_covid if 'num_covid' in locals() else 0} images")
print(f"  Tuberculosis: {num_tb if 'num_tb' in locals() else 0} images")
print(f"  Pneumothorax: {len(positive_ids) if 'positive_ids' in locals() else 0} images")

print("\n" + "=" * 80)
print("NEXT: Run `python preprocess_data.py` to organize into label folders")
print("=" * 80)