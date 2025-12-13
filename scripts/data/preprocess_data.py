"""
Preprocess cleaned chest X-ray data for training.
- Convert DICOM to PNG
- Resize to target size (224x224)
- Convert grayscale to RGB (3 identical channels)
- Split into train/val/test (70/15/15)
- Normalize to [0, 1]
"""

import os
import shutil
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_status(message, color=Colors.GREEN):
    print(f"{color}{message}{Colors.END}")

def read_dicom_image(dicom_path):
    """Read DICOM file and convert to numpy array."""
    try:
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array
        
        # Normalize to 0-255
        image = image - np.min(image)
        image = image / np.max(image)
        image = (image * 255).astype(np.uint8)
        
        return image
    except Exception as e:
        print(f"Error reading {dicom_path}: {e}")
        return None

def read_image(image_path):
    """Read image file (PNG/JPG/DICOM) and return as numpy array."""
    ext = image_path.suffix.lower()
    
    if ext == '.dcm':
        return read_dicom_image(image_path)
    elif ext in ['.png', '.jpg', '.jpeg']:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        return np.array(img)
    else:
        print(f"Unsupported format: {ext}")
        return None

def convert_grayscale_to_rgb(gray_image):
    """Convert grayscale image to RGB (3 identical channels)."""
    if len(gray_image.shape) == 2:
        # Stack grayscale 3 times to create RGB
        rgb_image = np.stack([gray_image]*3, axis=-1)
        return rgb_image
    return gray_image

def preprocess_and_save_image(src_path, dst_path, target_size=(224, 224)):
    """
    Read, preprocess and save image.
    - Read DICOM/PNG/JPG
    - Convert to RGB (3 channels)
    - Resize to target size
    - Save as PNG
    """
    # Read image
    image = read_image(src_path)
    if image is None:
        return False
    
    # Convert grayscale to RGB (3 identical channels for pretrained models)
    rgb_image = convert_grayscale_to_rgb(image)
    
    # Convert to PIL Image and resize
    pil_image = Image.fromarray(rgb_image.astype(np.uint8))
    pil_image = pil_image.resize(target_size, Image.LANCZOS)
    
    # Save as PNG
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(dst_path)
    
    return True

def split_data(class_dir, class_name, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data for one class into train/val/test.
    Returns: (train_files, val_files, test_files)
    """
    # Get all image files
    all_files = list(class_dir.glob("*"))
    all_files = [f for f in all_files if f.suffix.lower() in ['.dcm', '.png', '.jpg', '.jpeg']]
    
    # First split: train vs (val + test)
    train_files, temp_files = train_test_split(
        all_files, 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files, 
        train_size=val_size, 
        random_state=random_state
    )
    
    print(f"  {class_name:15s}: Train={len(train_files):5d}, Val={len(val_files):5d}, Test={len(test_files):5d}")
    
    return train_files, val_files, test_files

def process_dataset(cleaned_dir, processed_dir, target_size=(224, 224), 
                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Process entire dataset:
    1. Split into train/val/test
    2. Convert DICOM to PNG
    3. Resize to target size
    4. Convert grayscale to RGB
    """
    
    classes = ["Normal", "Pneumonia", "COVID", "Tuberculosis", "Pneumothorax"]
    
    print_status("\n=== Splitting Dataset ===", Colors.BLUE)
    
    splits_info = {}
    
    for class_name in classes:
        class_dir = cleaned_dir / class_name
        
        if not class_dir.exists():
            print_status(f"WARNING: {class_name} folder not found, skipping...", Colors.YELLOW)
            continue
        
        # Split data
        train_files, val_files, test_files = split_data(
            class_dir, class_name, train_ratio, val_ratio, test_ratio, random_state
        )
        
        splits_info[class_name] = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
    
    # Process and save images
    print_status("\n=== Processing Images ===", Colors.BLUE)
    print(f"Target size: {target_size}")
    print(f"Converting: DICOM → PNG, Grayscale → RGB\n")
    
    summary = {'train': {}, 'val': {}, 'test': {}}
    
    for split_name in ['train', 'val', 'test']:
        print_status(f"\n[Processing {split_name.upper()} set]", Colors.GREEN)
        
        for class_name in classes:
            if class_name not in splits_info:
                continue
            
            files = splits_info[class_name][split_name]
            
            # Create output directory
            output_dir = processed_dir / split_name / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each image
            success_count = 0
            for src_path in tqdm(files, desc=f"  {class_name}", ncols=100):
                # Generate output filename (always .png)
                dst_filename = src_path.stem + '.png'
                dst_path = output_dir / dst_filename
                
                # Preprocess and save
                if preprocess_and_save_image(src_path, dst_path, target_size):
                    success_count += 1
            
            summary[split_name][class_name] = success_count
    
    return summary

def generate_summary(processed_dir, summary):
    """Generate and save summary statistics."""
    print_status("\n" + "="*60, Colors.GREEN)
    print_status("PREPROCESSING SUMMARY", Colors.GREEN)
    print_status("="*60, Colors.GREEN)
    
    # Calculate totals
    totals = {'train': 0, 'val': 0, 'test': 0}
    
    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()} SET:")
        print("-" * 40)
        
        for class_name, count in summary[split_name].items():
            print(f"  {class_name:15s}: {count:5d} images")
            totals[split_name] += count
        
        print("-" * 40)
        print(f"  {'TOTAL':15s}: {totals[split_name]:5d} images")
    
    print("\n" + "="*60)
    print(f"GRAND TOTAL: {sum(totals.values())} images")
    print("="*60)
    
    # Save to CSV
    rows = []
    for split_name in ['train', 'val', 'test']:
        for class_name, count in summary[split_name].items():
            rows.append({
                'Split': split_name,
                'Class': class_name,
                'Count': count
            })
    
    df = pd.DataFrame(rows)
    csv_path = processed_dir / "preprocessing_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")
    
    # Save split ratios
    print("\n" + "="*60)
    print("SPLIT RATIOS:")
    print("-" * 40)
    total = sum(totals.values())
    for split_name, count in totals.items():
        ratio = count / total * 100
        print(f"  {split_name.upper():10s}: {ratio:5.2f}%")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Preprocess chest X-ray data')
    parser.add_argument('--cleaned_dir', type=str, default='data/cleaned',
                        help='Path to cleaned data directory')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='Path to output processed data directory')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                        help='Target image size (height width)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test split ratio')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    cleaned_dir = Path(args.cleaned_dir)
    processed_dir = Path(args.processed_dir)
    target_size = tuple(args.target_size)
    
    print_status("="*60, Colors.GREEN)
    print_status("CHEST X-RAY PREPROCESSING PIPELINE", Colors.GREEN)
    print_status("="*60, Colors.GREEN)
    print(f"Input:  {cleaned_dir}")
    print(f"Output: {processed_dir}")
    print(f"Target size: {target_size}")
    print(f"Split ratio: {args.train_ratio:.0%}/{args.val_ratio:.0%}/{args.test_ratio:.0%}")
    
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print_status(f"ERROR: Split ratios must sum to 1.0 (got {total_ratio})", Colors.RED)
        return
    
    # Create processed directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process dataset
    summary = process_dataset(
        cleaned_dir, 
        processed_dir, 
        target_size,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.random_state
    )
    
    # Generate summary
    generate_summary(processed_dir, summary)
    
    print_status("\n✓ Preprocessing completed successfully!", Colors.GREEN)
    print_status(f"\nProcessed data saved to: {processed_dir}", Colors.BLUE)
    print_status("\nNext step: Start training with src/train.py", Colors.YELLOW)
    
    print("\n" + "="*60)
    print_status("KEY PREPROCESSING DECISIONS:", Colors.BLUE)
    print("="*60)
    print("✓ DICOM → PNG conversion")
    print("✓ Grayscale → RGB (3 identical channels)")
    print("  → Compatible with pretrained ImageNet models")
    print(f"✓ Resized to {target_size[0]}x{target_size[1]}")
    print("✓ Normalized to [0, 255] uint8")
    print("✓ Augmentation will be applied during training")
    print("="*60)

if __name__ == "__main__":
    main()