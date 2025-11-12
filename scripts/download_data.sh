#!/bin/bash

echo "======================================"
echo "  DOWNLOADING CHEST X-RAY DATASETS"
echo "======================================"

DATA_DIR="data/raw"
mkdir -p $DATA_DIR
cd $DATA_DIR

echo ""
echo "📥 [1/3] Downloading COVID-19 Pneumonia dataset..."
kaggle datasets download -d prashant268/chest-xray-covid19-pneumonia
if [ $? -eq 0 ]; then
    echo "✅ Downloaded successfully"
    unzip -q chest-xray-covid19-pneumonia.zip -d covid19-pneumonia
    rm chest-xray-covid19-pneumonia.zip
else
    echo "❌ Failed to download"
    exit 1
fi

echo ""
echo "📥 [2/3] Downloading Tuberculosis dataset..."
kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
if [ $? -eq 0 ]; then
    echo "✅ Downloaded successfully"
    unzip -q tuberculosis-tb-chest-xray-dataset.zip -d tuberculosis
    rm tuberculosis-tb-chest-xray-dataset.zip
else
    echo "❌ Failed to download"
    exit 1
fi

echo ""
echo "📥 [3/3] Downloading Pneumonia dataset..."
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
if [ $? -eq 0 ]; then
    echo "✅ Downloaded successfully"
    unzip -q chest-xray-pneumonia.zip -d pneumonia
    rm chest-xray-pneumonia.zip
else
    echo "❌ Failed to download"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ ALL DATASETS DOWNLOADED!"
echo "======================================"
echo ""
echo "📊 Dataset sizes:"
du -sh *
echo ""

cd ../..
