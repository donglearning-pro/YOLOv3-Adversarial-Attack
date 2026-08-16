<div align="center">

# 🎯 YOLOv3 Adversarial Attack (TOG)

![CI Status](https://img.shields.io/badge/CI-Passing-brightgreen?style=for-the-badge&logo=github-actions)
![Python Version](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![CUDA Version](https://img.shields.io/badge/CUDA-13.0-green?style=for-the-badge&logo=nvidia)
![GPU Hardware](https://img.shields.io/badge/GPU-NVIDIA%20RTX%205050%20(8GB)-76B900?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**Team 1 — SEAS Summer Camp Project**  
*White-box Targeted Objectness Gradient (TOG) Attacks on YOLOv3 Object Detection*

---

</div>

## 📋 Table of Contents
- [📌 Overview](#-overview)
- [📂 Directory & Dataset Structure](#-directory--dataset-structure)
- [🚀 Quick Start & Environment Setup](#-quick-start--environment-setup)
- [💻 Headless Execution (CLI)](#-headless-execution-cli)
- [📊 Experimental Results & Metrics](#-experimental-results--metrics)
- [🧪 Testing & Validation](#-testing--validation)
- [👥 Team & Acknowledgments](#-team--acknowledgments)
- [📜 Citations & References](#-citations--references)

## 📌 Overview

This repository explores the adversarial vulnerabilities of **YOLOv3** under **Targeted Objectness Gradient (TOG)** attack strategies. By generating imperceptible, bounded perturbation patterns ($L_\infty$), we demonstrate how real-time object detection pipelines can be manipulated to cause missing detections, false positives, or misclassifications.

## 📸 Visual Demonstration

Below is a side-by-side execution demo comparing raw YOLOv3 detections against the **TOG-Vanishing** adversarial attack on video input:

<div align="center">
  <video src="output_side_by_side.mp4" controls="controls" muted="muted" style="max-height: 480px;">
    Your browser does not support the video tag.
  </video>
  
  <p><i>Left: <b>BENIGN</b> &nbsp;|&nbsp; Right: <b>TOG-VANISHING ATTACK</b></i></p>
</div>

### 🌟 Hardware & Benchmark System Specs
All benchmark experiments and metrics were executed on the following setup:
* **Machine**: ASUS TUF GAMING F16
* **CPU**: Intel Core i7-14650HX (2.20 GHz)
* **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)
* **RAM**: 16.0 GB
* **Environment**: Windows 11 Home (25H2, Build 26200.9168) | **Python 3.12** | **CUDA 13.0**

---

## 📂 Directory & Dataset Structure

Organize your repository root directory as shown below before running training or attack scripts:

```text
YOLOv3-Adversarial-Attack/
├── data/                       
│   └── coco/                   # COCO root folder
│       ├── annotations/        # COCO annotation JSON files
│       └── val2017/            # MS COCO 2017 validation JPEG images
├── model/                      # YOLOv3 model configuration & weights
│   ├── class.names             # Class labeling file (80 COCO classes)
│   ├── yolov3                  # Darknet architecture configuration source (.cfg)
│   └── yolov3.weights          # Pre-trained Darknet weights (242,195 KB)
├── output_side_by_side.mp4     # Side-by-side demonstration output video
├── sample_video.mp4            # Sample test input video
├── main.ipynb                  # End-to-end execution & visualization notebook
├── tog_attacks.py              # Core TOG attack implementations & CLI interface
├── tog_model.py                # Gradient optimization loops & loss calculations
├── yolov3_model.py             # Darknet YOLOv3 wrapper & inference helper functions
├── utils.py                    # Pre/post-processing & visualization tools
├── test_smoke.py               # Automated unit & sanity tests
├── environment.yml             # Conda environment definition (Python 3.12 / CUDA 13.0)
├── Dockerfile                  # Runtime environment container definition
├── requirements.txt            # Python dependencies
└── README.md                   # Repository documentation

```

### 📥 Download Instructions

1. **MS COCO 2017 Validation Set**:
* Download: [COCO 2017 Val Images](https://drive.google.com/file/d/1VutbbQAgCn7__vfPy6qYtb0rLOIpc579/view?usp=sharing)
* Extract image files directly into `./data/coco/val2017/` and annotations into `./data/coco/annotations/`.


2. **YOLOv3 Weights & Config**:
* Download: [yolov3.weights](https://drive.google.com/file/d/1plerp95bu5GJjgXvMYTTMUUuSr33MwY0/view?usp=sharing)
* Place `yolov3.weights`, `yolov3` (config), and `class.names` into `./model/`.



---

## 🚀 Quick Start & Environment Setup

### Option 1: Conda Environment (Recommended)

```bash
# Create and activate Python 3.12 / CUDA 13 environment
conda env create -f environment.yml
conda activate tog-yolov3

```

### Option 2: Pip Setup

```bash
# Clone repository
git clone https://github.com/donglearning-pro/YOLOv3-Adversarial-Attack.git
cd YOLOv3-Adversarial-Attack

# Install dependencies
python -m pip install -r requirements.txt

```

### Option 3: Docker

```bash
# Build image
docker build -t tog-yolov3 .

# Run with GPU support
docker run --gpus all -it -v $(pwd):/workspace tog-yolov3

```

---

## 💻 Headless Execution (CLI)

Run attacks directly from the command line without Jupyter notebooks:

```bash
# Execute TOG-Vanishing attack
python tog_attacks.py \
  --weights model/yolov3.weights \
  --config model/yolov3 \
  --class_names model/class.names \
  --data_dir data/coco/val2017 \
  --attack_type vanishing \
  --eps 0.031 \
  --steps 10 \
  --output_dir ./results

# Run minimal system test
python -m unittest test_smoke.py

```

---

## 📊 Experimental Results & Metrics

Below are some empirical findings evaluated across the sampled COCO 2017 validation dataset on the RTX 5050 GPU setup.

### 1. Perturbation Bound Amplitude ($\epsilon$) Analysis
![Epsilon Analysis](assets/graph-1.png)

The relationship between perturbation bound $\epsilon$ (scaled over 255), Attack Success Rate (ASR), and Peak Signal-to-Noise Ratio (PSNR):

| Epsilon ($\epsilon$) | ASR (%) | PSNR (dB) | Visual Impact |
| --- | --- | --- | --- |
| **2/255** | 87.5% | 45.5 dB | Completely imperceptible |
| **4/255** | 97.0% | 39.8 dB | Near-imperceptible |
| **6/255** | 97.0% | 36.2 dB | Low perceptual noise |
| **8/255** | 100.0% | 33.5 dB | Optimal trade-off point |
| **10/255** | 98.0% | 31.5 dB | Minor visible grain |
| **12/255** | 97.5% | 29.8 dB | Visible grain |
| **14/255** | 100.0% | 28.2 dB | Moderately noisy |
| **16/255** | 98.8% | 27.0 dB | High visible perturbation |

---

### 2. Convergence Speed & Iteration Latency ($N$)
![Speed and Latency](assets/graph-2.png)

Runtime performance and convergence rate across iteration steps ($N$):

| Iteration Steps ($N$) | ASR (%) | Runtime Latency (s/img) |
| --- | --- | --- |
| **1** | 60.0% | 0.12 s |
| **2** | 77.5% | 0.19 s |
| **5** | 92.0% | 0.38 s |
| **10** | **99.0%** | **0.75 s** |
| **15** | **100.0%** | **0.78 s** |
| **20** | 100.0% | 0.98 s |
| **30** | 100.0% | 1.42 s |

---

### 3. Comparison Across 4 TOG Attack Variants
![TOG Attack Comparison](assets/graph-3.png)

| Attack Variant | Attack Success Rate (ASR) | $\Delta\text{mAP@50}$ Drop | Image Quality (PSNR) | Generation Time |
| --- | --- | --- | --- | --- |
| **TOG-Vanishing** | **100.0%** | **~0.80** | **34.1 dB** | **0.47 s/img** |
| **TOG-Fabrication** | **93.0%** | **~0.74** | **34.0 dB** | **0.48 s/img** |
| **TOG-Mislabeling** | **74.0%** | **~0.59** | **34.2 dB** | **0.61 s/img** |
| **TOG-Untargeted** | **86.5%** | **~0.69** | **34.0 dB** | **0.62 s/img** |

*TOG-Vanishing achieved the highest impact with complete bounding box suppression at 0.47s per image.*

---

### 4. Attack Sensitivity by Object Scale (COCO Standard)
![Object Sensivity](assets/graph-4.png)

Vulnerability comparison of object categories based on pixel bounding box area under **TOG-Vanishing**:

| Object Category | Bounding Box Pixel Area | ASR (%) | Vulnerability Level |
| --- | --- | --- | --- |
| **Small Objects** | $< 32^2\text{ px}$ | **100.00%** | Critical |
| **Medium Objects** | $32^2 \text{ to } 96^2\text{ px}$ | **100.00%** | Critical |
| **Large Objects** | $> 96^2\text{ px}$ | **95.67%** | High |

---

## 🧪 Testing & Validation

Run minimal environment checks to confirm model file loading, paths, and GPU acceleration:

```bash
python -m unittest test_smoke.py

```

---

## 👥 Team & Acknowledgments

* **Program**: SEAS (Summer in Engineering and Applied Sciences)
* **Team Members**:
* Dương Phương Đông *(Team Leader)*
* Hoàng Bình Minh
* Trần Hoài Thương
* Dương Xuân Quân
* Lê Quang Huy


* **Mentors**:
* Nguyễn Tiết Khôi Nguyên
* Nguyễn Xuân Minh Đức



---

## 📜 Citations & References

If you use this repository or build upon our work, please cite the following papers:

* **TOG Attack**: K.-H. Chow, L. Liu, S.-T. Liew, M. E. Gursoy, and S. Truex, "TOG: Targeted Adversarial Objectness Gradient Attacks on Real-time Object Detectors," *IEEE Transactions on Computers*, 2020.
* **YOLOv3**: J. Redmon and A. Farhadi, "YOLOv3: An Incremental Improvement," *arXiv preprint arXiv:1804.02767*, 2018.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```

---

### 🔨 Supporting Repository Files (Updated for Python 3.12 & CUDA 13.0)

#### 1. `environment.yml`
```yaml
name: tog-yolov3
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pytorch
  - torchvision
  - pytorch-cuda=13.0
  - numpy
  - opencv
  - pillow
  - matplotlib
  - scikit-image
  - tqdm
  - jupyter
  - pip
  - pip:
    - -r requirements.txt

```

#### 2. `Dockerfile`

```dockerfile
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3.12", "test_smoke.py"]

```

#### 3. `test_smoke.py`

```python
import os
import unittest
import torch

class TestTOGYOLOv3Environment(unittest.TestCase):
    def test_cuda_availability(self):
        """Verify CUDA 13.0 hardware acceleration support."""
        self.assertTrue(torch.cuda.is_available(), "CUDA is not available. GPU is required for fast execution.")

    def test_model_files_exist(self):
        """Verify model configuration, weights, and class files exist in model directory."""
        required_files = ["yolov3.weights", "yolov3", "class.names"]
        for file in required_files:
            file_path = os.path.join("model", file)
            self.assertTrue(os.path.exists(file_path), f"Required model file missing: {file_path}")

    def test_dataset_structure_exist(self):
        """Verify dataset hierarchy under data/coco/."""
        coco_dir = os.path.join("data", "coco")
        self.assertTrue(os.path.exists(coco_dir), f"Directory missing: {coco_dir}")

if __name__ == "__main__":
    unittest.main()

```

---
