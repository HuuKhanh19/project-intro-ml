#!/bin/bash

# Script to download chest X-ray datasets from Kaggle
# Prerequisites: 
# 1. Install kaggle CLI: pip install kaggle
# 2. Setup Kaggle API credentials: https://github.com/Kaggle/kaggle-api#api-credentials

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Starting Chest X-Ray Dataset Download ===${NC}\n"

# Create raw data directory
RAW_DATA_DIR="data/raw"
mkdir -p "$RAW_DATA_DIR"

# Check if Kaggle API is configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${RED}Error: Kaggle API credentials not found!${NC}"
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Move kaggle.json to ~/.kaggle/"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Function to download and extract
download_and_extract() {
    local dataset=$1
    local folder_name=$2
    
    echo -e "${YELLOW}Downloading: $dataset${NC}"
    
    cd "$RAW_DATA_DIR"
    kaggle datasets download -d "$dataset" --unzip -p "$folder_name"
    cd ../../
    
    echo -e "${GREEN}✓ Downloaded: $folder_name${NC}\n"
}

download_competition() {
    local competition=$1
    local folder_name=$2
    
    echo -e "${YELLOW}Downloading competition: $competition${NC}"
    
    cd "$RAW_DATA_DIR"
    mkdir -p "$folder_name"
    kaggle competitions download -c "$competition" -p "$folder_name"
    
    # Unzip all files in the competition folder
    cd "$folder_name"
    for file in *.zip; do
        if [ -f "$file" ]; then
            echo "Extracting $file..."
            unzip -q "$file"
            rm "$file"
        fi
    done
    cd ../../../
    
    echo -e "${GREEN}✓ Downloaded: $folder_name${NC}\n"
}

# 1. Download RSNA Pneumonia Detection Challenge (Normal + Pneumonia)
echo -e "${GREEN}[1/4] RSNA Pneumonia Detection Challenge${NC}"
download_competition "rsna-pneumonia-detection-challenge" "rsna_pneumonia"

# 2. Download COVID-19 Radiography Database
echo -e "${GREEN}[2/4] COVID-19 Radiography Database${NC}"
download_and_extract "tawsifurrahman/covid19-radiography-database" "covid19_radiography"

# 3. Download Tuberculosis Dataset (New source - Yasser Hessein)
echo -e "${GREEN}[3/4] Tuberculosis Chest X-rays${NC}"
download_and_extract "yasserhessein/tuberculosis-chest-x-rays-images" "tuberculosis"

# 4. Download Pneumothorax Dataset
echo -e "${GREEN}[4/4] Pneumothorax (SIIM-ACR)${NC}"
download_and_extract "vbookshelf/pneumothorax-chest-xray-images-and-masks" "pneumothorax"

# Display summary
echo -e "\n${GREEN}=== Download Complete ===${NC}"
echo "Data saved to: $RAW_DATA_DIR"
echo ""
echo "Directory structure:"
tree -L 2 "$RAW_DATA_DIR" 2>/dev/null || ls -R "$RAW_DATA_DIR"

echo -e "\n${YELLOW}Next step: Run clean_raw_data.py to clean and organize the data${NC}"
echo "Command: python scripts/clean_raw_data.py"