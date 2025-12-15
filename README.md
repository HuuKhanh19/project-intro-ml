# Phân Loại Bệnh Hô Hấp qua Ảnh X-quang Phổi

![Chest X-ray Classification](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)
![Python](https://img.shields.io/badge/Python-3.11-blue)

## 📋 Mục lục
- [Giới thiệu](#giới-thiệu)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
- [Training và Evaluation](#training-và-evaluation)
- [Kết quả](#kết-quả)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Giới thiệu

Đây là repository của dự án phân loại bệnh hô hấp sử dụng các mô hình Deep Learning dựa trên các model: **MLP**, **LeNet**, **DenseNet-121**, và **EfficientNet-B0**. Dự án cung cấp code training, pre-trained models, và công cụ đánh giá để chẩn đoán các bệnh phổi phổ biến.

### 🎯 Phân loại 5 loại bệnh:
- **Normal** (Bình thường)
- **Pneumonia** (Viêm phổi)
- **COVID-19**
- **Tuberculosis** (Lao phổi)
- **Pneumothorax** (Tràn khí màng phổi)

### 🔬 So sánh hiệu quả:
Dự án so sánh **4 models** với **2 loss functions** (Weighted Cross-Entropy và Focal Loss), tổng cộng **8 experiments**, đạt độ chính xác cao nhất **91.41%** trên test set với **DenseNet-121 + Focal Loss**.

---

## Cài đặt môi trường

### 1. Clone repository

```bash
git clone https://github.com/HuuKhanh19/project-intro-ml
cd chest-xray-classification
```

### 2. Tạo môi trường Python

```bash
conda create -n chest-xray python=3.11
conda activate chest-xray
```

### 3. Cài đặt PyTorch (CUDA 12.9)

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
```

### 4. Cài đặt các thư viện khác

```bash
pip install -r requirements.txt
```

---

## Chuẩn bị dữ liệu

### 1. Cấu hình Kaggle API

Để tải dữ liệu từ Kaggle, cần có API token:

1. Truy cập https://www.kaggle.com/settings/account
2. Nhấn **"Create New API Token"**
3. Di chuyển file `kaggle.json`:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Tải và xử lý dữ liệu

Download dữ liệu từ 4 nguồn Kaggle:

```bash
# Bước 1: Download (~30-60 phút)
chmod +x scripts/data/download_data.sh
./scripts/data/download_data.sh
```

Các dataset sử dụng:
- [RSNA Pneumonia Detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- [Tuberculosis Chest X-rays](https://www.kaggle.com/datasets/yasserhessein/tuberculosis-chest-x-rays-images)
- [Pneumothorax (SIIM-ACR)](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks)

```bash
# Bước 2: Làm sạch dữ liệu
python scripts/data/clean_raw_data.py

# Bước 3: Tiền xử lý (resize, split, normalize)
python scripts/data/preprocess_data.py
```

**Kết quả:** Dữ liệu được chia thành train/val/test (70/15/15):
```
data/processed/
├── train/     # 16,547 ảnh
├── val/       #  3,546 ảnh
└── test/      #  3,549 ảnh
```

---

## Training và Evaluation

### 🎓 Training một experiment

**Cách 1: Dùng experiment ID**
```bash
python scripts/experiments/train.py --experiment exp01_densenet121_weighted_ce
```

**Cách 2: Chỉ định model và loss**
```bash
python scripts/experiments/train.py --model densenet121 --loss weighted_ce
```

**Cách 3: Chạy training + evaluation (khuyên dùng)**
```bash
chmod +x scripts/experiments/run_experiment.sh
./scripts/experiments/run_experiment.sh exp01_densenet121_weighted_ce
```

### 🚀 Training tất cả experiments

Chạy 8 experiments (4 models × 2 loss functions):

```bash
chmod +x scripts/experiments/run_all.sh
./scripts/experiments/run_all.sh
```

**Lưu ý:** Quá trình training mất khoảng **10-15 giờ** tùy GPU.

### 📊 Evaluation

Đánh giá model đã train trên test set:

```bash
python scripts/experiments/evaluate.py \
    --checkpoint checkpoints/exp01_densenet121_weighted_ce/checkpoint_best.pth \
    --split test
```

**Output:**
- `metrics_test.json` - Các metrics (accuracy, precision, recall, F1, AUC)
- `confusion_matrix_test.png` - Ma trận nhầm lẫn
- `roc_curves_test.png` - ROC curves
- `classification_report_test.txt` - Báo cáo chi tiết

### 📈 So sánh kết quả

So sánh tất cả 8 experiments:

```bash
python scripts/experiments/compare_results.py
```

**Output:** File so sánh trong folder `results/`
- `summary.csv` - Bảng tổng hợp
- `accuracy_comparison.png` - Biểu đồ so sánh accuracy
- `metrics_comparison.png` - So sánh các metrics
- `model_comparison_by_loss.png` - So sánh theo loss function

### 📺 Xem TensorBoard

```bash
# Xem log một experiment
tensorboard --logdir checkpoints/exp01_densenet121_weighted_ce/logs

# Xem tất cả experiments
tensorboard --logdir checkpoints/
```

Truy cập: http://localhost:6006

---

## Kết quả

### 📊 So sánh 8 Experiments

| Model | Loss | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|:------|:-----|:------------:|:-------------:|:----------:|:------:|
| **DenseNet-121** | **Focal Loss** | **91.41** | **91.03** | **90.82** | **90.87** |
| DenseNet-121 | Weighted CE | 90.81 | 90.34 | 90.27 | 90.31 |
| EfficientNet-B0 | Weighted CE | 90.48 | 90.07 | 89.77 | 89.92 |
| EfficientNet-B0 | Focal Loss | 90.42 | 90.16 | 89.79 | 89.92 |
| LeNet | Weighted CE | 87.09 | 86.69 | 87.28 | 86.83 |
| LeNet | Focal Loss | 86.48 | 86.31 | 84.90 | 85.50 |
| MLP | Weighted CE | 61.45 | 55.83 | 57.63 | 55.47 |
| MLP | Focal Loss | 59.99 | 53.09 | 56.39 | 53.76 |

### 🎯 Kết luận

**Model tốt nhất:** DenseNet-121 với Focal Loss đạt **91.41% accuracy** trên test set.

**Nhận xét:**
- **Pretrained models** (DenseNet-121, EfficientNet-B0) vượt trội với accuracy **>90%**
- **Focal Loss** cho kết quả tốt hơn **Weighted CE** một chút (91.41% vs 90.81%)
- **LeNet** đạt kết quả khá tốt (~87%) cho model train from scratch
- **MLP** baseline cho kết quả thấp (~60%) do không tận dụng được đặc trưng không gian của ảnh

Bằng cách sử dụng repository này với các pretrained models, bạn có thể đạt được độ chính xác **>90%** trên test set, phù hợp để hỗ trợ chẩn đoán các bệnh phổi với độ tin cậy cao.

---

## 📚 Tài liệu tham khảo

### Papers

```bibtex
@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4700--4708},
  year={2017}
}

@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={6105--6114},
  year={2019}
}

@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2980--2988},
  year={2017}
}

@article{lecun1998gradient,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal={Proceedings of the IEEE},
  volume={86},
  number={11},
  pages={2278--2324},
  year={1998}
}
```

### Datasets

- **RSNA Pneumonia Detection Challenge**  
  https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

- **COVID-19 Radiography Database** (Tawsifur Rahman et al.)  
  https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

- **Tuberculosis Chest X-rays** (Yasser Hessein)  
  https://www.kaggle.com/datasets/yasserhessein/tuberculosis-chest-x-rays-images

- **Pneumothorax Challenge** (SIIM-ACR)  
  https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks

---