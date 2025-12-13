"""
Clean and organize raw chest X-ray data from multiple sources.
Ensures zero-overlap and proper class distribution.
Output: data/cleaned/ (ready for train/val/test split)
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_status(message, color=Colors.GREEN):
    print(f"{color}{message}{Colors.END}")

def clean_rsna_data(raw_dir, cleaned_dir):
    """
    Clean RSNA Pneumonia Detection data.
    Extract Normal and Pneumonia classes.
    """
    print_status("\n[1/5] Processing RSNA Pneumonia Detection Challenge...", Colors.BLUE)
    
    rsna_path = raw_dir / "rsna_pneumonia"
    
    # Check required files
    labels_file = rsna_path / "stage_2_train_labels.csv"
    detailed_file = rsna_path / "stage_2_detailed_class_info.csv"
    images_dir = rsna_path / "stage_2_train_images"
    
    if not labels_file.exists() or not detailed_file.exists():
        print_status(f"ERROR: Missing CSV files in {rsna_path}", Colors.RED)
        return
    
    # Read CSV files
    labels_df = pd.read_csv(labels_file)
    detailed_df = pd.read_csv(detailed_file)
    
    # Create output directories
    normal_dir = cleaned_dir / "Normal"
    pneumonia_dir = cleaned_dir / "Pneumonia"
    normal_dir.mkdir(parents=True, exist_ok=True)
    pneumonia_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract Pneumonia cases (Target = 1)
    pneumonia_ids = labels_df[labels_df['Target'] == 1]['patientId'].unique()
    print(f"Found {len(pneumonia_ids)} Pneumonia cases")
    
    for patient_id in tqdm(pneumonia_ids, desc="Copying Pneumonia images"):
        src = images_dir / f"{patient_id}.dcm"
        if src.exists():
            dst = pneumonia_dir / f"rsna_{patient_id}.dcm"
            shutil.copy2(src, dst)
    
    # Extract Normal cases (class = 'Normal')
    # Important: Skip 'No Lung Opacity / Not Normal' to keep dataset clean
    normal_ids = detailed_df[detailed_df['class'] == 'Normal']['patientId'].unique()
    print(f"Found {len(normal_ids)} Normal cases")
    
    for patient_id in tqdm(normal_ids, desc="Copying Normal images"):
        src = images_dir / f"{patient_id}.dcm"
        if src.exists():
            dst = normal_dir / f"rsna_{patient_id}.dcm"
            shutil.copy2(src, dst)
    
    print_status(f"✓ RSNA: {len(normal_ids)} Normal, {len(pneumonia_ids)} Pneumonia", Colors.GREEN)

def clean_covid_data(raw_dir, cleaned_dir):
    """
    Clean COVID-19 Radiography Database.
    ONLY take COVID folder, remove Normal/Pneumonia/Lung_Opacity to avoid overlap.
    """
    print_status("\n[2/5] Processing COVID-19 Radiography Database...", Colors.BLUE)
    
    covid_src = raw_dir / "covid19_radiography" / "COVID-19_Radiography_Dataset" / "COVID" / "images"
    
    if not covid_src.exists():
        # Try without images subfolder
        covid_src = raw_dir / "covid19_radiography" / "COVID-19_Radiography_Dataset" / "COVID"
    
    if not covid_src.exists():
        print_status(f"ERROR: COVID folder not found", Colors.RED)
        print(f"Expected path: {covid_src}")
        return
    
    covid_dir = cleaned_dir / "COVID"
    covid_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all COVID images (try multiple extensions)
    image_files = (list(covid_src.glob("*.png")) + 
                   list(covid_src.glob("*.jpg")) + 
                   list(covid_src.glob("*.jpeg")))
    
    if len(image_files) == 0:
        print_status(f"WARNING: No images found in {covid_src}", Colors.YELLOW)
        return
    
    for img_file in tqdm(image_files, desc="Copying COVID images"):
        dst = covid_dir / f"covid_{img_file.name}"
        shutil.copy2(img_file, dst)
    
    print_status(f"✓ COVID: {len(image_files)} images", Colors.GREEN)

def clean_tuberculosis_data(raw_dir, cleaned_dir):
    """
    Clean Tuberculosis dataset (Yasser Hessein).
    ONLY take Tuberculosis folder, remove Normal to avoid overlap.
    """
    print_status("\n[3/5] Processing Tuberculosis Chest X-rays...", Colors.BLUE)
    
    # Updated path based on actual structure
    tb_src = raw_dir / "tuberculosis" / "Dataset of Tuberculosis Chest X-rays Images" / "TB Chest X-rays"
    
    if not tb_src.exists():
        print_status(f"ERROR: Tuberculosis folder not found", Colors.RED)
        print(f"Expected path: {tb_src}")
        return
    
    tb_dir = cleaned_dir / "Tuberculosis"
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all TB images (try multiple extensions)
    image_files = (list(tb_src.glob("*.png")) + 
                   list(tb_src.glob("*.jpg")) + 
                   list(tb_src.glob("*.jpeg")))
    
    if len(image_files) == 0:
        print_status(f"WARNING: No images found in {tb_src}", Colors.YELLOW)
        return
    
    for img_file in tqdm(image_files, desc="Copying TB images"):
        dst = tb_dir / f"tb_{img_file.name}"
        shutil.copy2(img_file, dst)
    
    print_status(f"✓ Tuberculosis: {len(image_files)} images", Colors.GREEN)

def clean_pneumothorax_data(raw_dir, cleaned_dir):
    """
    Clean Pneumothorax dataset (SIIM-ACR).
    Filter images with has_pneumo = 1 from BOTH train and test sets.
    """
    print_status("\n[4/5] Processing Pneumothorax (SIIM-ACR)...", Colors.BLUE)
    
    pneumo_path = raw_dir / "pneumothorax" / "siim-acr-pneumothorax"
    
    if not pneumo_path.exists():
        pneumo_path = raw_dir / "pneumothorax"
    
    # Read BOTH train and test CSV files
    train_csv = pneumo_path / "stage_1_train_images.csv"
    test_csv = pneumo_path / "stage_1_test_images.csv"
    
    dfs = []
    
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        print(f"Found train CSV with {len(train_df)} images")
        dfs.append(train_df)
    
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        print(f"Found test CSV with {len(test_df)} images")
        dfs.append(test_df)
    
    if not dfs:
        print_status(f"ERROR: No CSV files found in {pneumo_path}", Colors.RED)
        return
    
    # Combine train and test dataframes
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total images: {len(df)}")
    print(f"CSV columns: {df.columns.tolist()}")
    
    # Filter images with pneumothorax
    if 'has_pneumo' in df.columns:
        pneumo_df = df[df['has_pneumo'] == 1]
    elif ' EncodedPixels' in df.columns:
        pneumo_df = df[df[' EncodedPixels'] != '-1']
    elif 'EncodedPixels' in df.columns:
        pneumo_df = df[df['EncodedPixels'] != '-1']
    else:
        print_status(f"WARNING: Could not identify pneumothorax column", Colors.YELLOW)
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    pneumo_dir = cleaned_dir / "Pneumothorax"
    pneumo_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images directory
    images_dir = pneumo_path / "png_images"
    if not images_dir.exists():
        images_dir = pneumo_path / "dicom-images-train"
    if not images_dir.exists():
        images_dir = pneumo_path
    
    print(f"Looking for images in: {images_dir}")
    
    # Get image filenames - prefer 'new_filename' if available
    if 'new_filename' in pneumo_df.columns:
        image_files = pneumo_df['new_filename'].unique()
        print(f"Found {len(image_files)} pneumothorax cases (using new_filename)")
    elif 'ImageId' in pneumo_df.columns:
        image_files = pneumo_df['ImageId'].unique()
        print(f"Found {len(image_files)} pneumothorax cases (using ImageId)")
    elif 'image_id' in pneumo_df.columns:
        image_files = pneumo_df['image_id'].unique()
        print(f"Found {len(image_files)} pneumothorax cases (using image_id)")
    else:
        print_status(f"ERROR: Could not find image filename column", Colors.RED)
        print(f"Available columns: {pneumo_df.columns.tolist()}")
        return
    
    copied = 0
    for filename in tqdm(image_files, desc="Copying Pneumothorax images"):
        # If filename doesn't have extension, try adding common ones
        if '.' in str(filename):
            src = images_dir / filename
            if src.exists():
                dst = pneumo_dir / f"pneumo_{filename}"
                shutil.copy2(src, dst)
                copied += 1
        else:
            # Try different extensions
            for ext in ['.png', '.jpg', '.dcm', '.jpeg']:
                src = images_dir / f"{filename}{ext}"
                if src.exists():
                    dst = pneumo_dir / f"pneumo_{filename}{ext}"
                    shutil.copy2(src, dst)
                    copied += 1
                    break
    
    print_status(f"✓ Pneumothorax: {copied} images (from {len(image_files)} cases)", Colors.GREEN)

def generate_summary(cleaned_dir):
    """Generate summary statistics of cleaned data."""
    print_status("\n[5/5] Generating Summary...", Colors.BLUE)
    
    classes = ["Normal", "Pneumonia", "COVID", "Tuberculosis", "Pneumothorax"]
    summary = {}
    
    for cls in classes:
        cls_dir = cleaned_dir / cls
        if cls_dir.exists():
            count = len(list(cls_dir.glob("*")))
            summary[cls] = count
        else:
            summary[cls] = 0
    
    # Print summary table
    print("\n" + "="*50)
    print_status("DATASET SUMMARY", Colors.GREEN)
    print("="*50)
    
    total = 0
    for cls, count in summary.items():
        print(f"{cls:20s}: {count:6d} images")
        total += count
    
    print("-"*50)
    print(f"{'TOTAL':20s}: {total:6d} images")
    print("="*50)
    
    # Save summary to CSV
    summary_df = pd.DataFrame(list(summary.items()), columns=['Class', 'Count'])
    summary_file = cleaned_dir / "dataset_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Clean raw chest X-ray data')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Path to raw data directory')
    parser.add_argument('--cleaned_dir', type=str, default='data/cleaned',
                        help='Path to output cleaned data directory')
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    cleaned_dir = Path(args.cleaned_dir)
    
    print_status("="*60, Colors.GREEN)
    print_status("CHEST X-RAY DATA CLEANING PIPELINE", Colors.GREEN)
    print_status("="*60, Colors.GREEN)
    print(f"Raw data: {raw_dir}")
    print(f"Output: {cleaned_dir}\n")
    
    # Create cleaned directory
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean each dataset
    clean_rsna_data(raw_dir, cleaned_dir)
    clean_covid_data(raw_dir, cleaned_dir)
    clean_tuberculosis_data(raw_dir, cleaned_dir)
    clean_pneumothorax_data(raw_dir, cleaned_dir)
    
    # Generate summary
    summary = generate_summary(cleaned_dir)
    
    print_status("\n✓ Data cleaning completed successfully!", Colors.GREEN)
    print_status(f"\nCleaned data saved to: {cleaned_dir}", Colors.BLUE)
    print_status("\nNext step: Run preprocess_data.py to split train/val/test", Colors.YELLOW)

if __name__ == "__main__":
    main()