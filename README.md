# Tri-Prompted Visual-Text Alignment with Ordinal-Aware Regression for Blind Image Quality Assessment

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)


## 📋 Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Cross-Dataset Evaluation](#cross-dataset-evaluation)
- [Model Checkpoints](#model-checkpoints)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)
- [Updates](#updates)

## 🎯 Introduction

This repository provides the **official PyTorch implementation** of the paper:

> **"Tri-Prompted Visual-Text Alignment with Ordinal-Aware Regression for Blind Image Quality Assessment"**


### Architecture Overview

![image-20260302163011547](https://lsky.ruotao.tech/i/2026/03/02/69a54a983a9c6.png)

*Fig. The framework of the proposed method.*


## 🛠️ Installation

### Prerequisites

- *NVIDIA GPU with CUDA support (recommended: RTX 3090 with 24GB VRAM)*
- *Python 3.8+*
- *CUDA 11.8 (recommended)*
- *Anaconda or Miniconda*

### Step 1: Clone the Repository

```bash
git clone https://github.com/Why0322/ICME-TP-IQA
cd TP-IQA
```

### Step 2: Create Conda Environment

```bash
conda create -n TP_IQA python=3.8 -y
conda activate TP_IQA
```

### Step 3: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Core dependencies include:**
- PyTorch 2.4.1 + CUDA 11.8
- torchvision
- numpy, scipy
- timm
- tensorboard
- einops

## 🧩 Dataset Preparation

### Supported Datasets

This project supports the following IQA datasets:

| Dataset | Images | Type | Download Link |
|---------|--------|------|---------------|
| **LIVE** | 29 reference + 779 distorted | Synthetic | [Link](https://live.ece.utexas.edu/research/quality/subjective.htm) |
| **CSIQ** | 30 reference + 866 distorted | Synthetic | [Link](http://vision.okstate.edu/csiq) |
| **TID2013** | 25 reference + 3000 distorted | Synthetic | [Link](http://www.ponomarenko.info/tid2013.htm) |
| **KADID-10k** | 81 reference + 10,125 distorted | Synthetic | [Link](http://database.mmsp-kn.de/kadid-10k-database.html) |
| **LIVE-C** | 1,162 images | Authentic | [Link](https://live.ece.utexas.edu/research/ChallengeDB/index.html) |
| **KonIQ-10k** | 10,073 images | Authentic | [Link](http://database.mmsp-kn.de/koniq-10k-database.html) |
| **SPAQ** | 11,125 images | Authentic | [Link](https://github.com/h4nwei/SPAQ) |
| **BID** | 585 images | Authentic | [Link](https://github.com/zwx8981/UNIQUE#link-to-download-the-bid-dataset) |

### Directory Structure

The project uses a two-level directory structure:

#### 1. Dataset Images (stored in `../datasets/`)

Download and organize the actual dataset images in the parent directory:

```
../datasets/
├── LIVEC/
│   └── Images/                    # 1,162 authentically distorted images
├── tid2013/
│   ├── distorted_images/          # 3,000 distorted images
├── CSIQ/
│   ├── dist_imgs/                 # Distorted images
├── KADID-10K/
│   └── images/                    # 10,125 images
├── KonIQ-10K/
│   └── 1024x768/                  # 10,073 images
├── SPAQ/
│   └── TestImage/                 # 11,125 images
└── ...                            # Other datasets
```

#### 2. Data Processing and Labels Scripts (in `IQA/`)

The `IQA/` folder in the project contains dataset-specific processing scripts and label files:

```
IQA/
├── iqa_dataset_clip.py       # Data Processing
├── build.py				  # Dataset Construction 
├── clive_all_clip_with_quality.csv
 							  # Livec dataset labels 
└── ...                       # Other dataset labels
```

The `iqa_dataset_clip.py` file contains the processing functions for all datasets.

Files such as (`clive_all_clip_with_quality.csv , csiq_all_clip_with_quality.csv`,) contain all image labels.

**Label Example:**

```
Images/611.JPG	38.489474	
realistic blur landscape poor

Images/211.bmp	42.686047	
other realistic	human	fair
```

## 🚀 Usage

### Training

#### Single Dataset Training

To train the model on a specific dataset (e.g., LIVEC):

```bash
bash livec.sh
```

**Key training features:**

- Automatically creates output directories (e.g., `log/livec/`)
- Saves checkpoints and logs during training

#### Training Configuration

Edit the training script or create a configuration file to customize like:
- Dataset selection
- Batch size
- etc.

### Cross-Dataset Evaluation

Cross-validation of the dataset can be configured in the following Python code.:

```bash
eval.py
```

This script supports all cross-dataset combinations, such as:
- LIVE → CSIQ
- CSIQ → TID2013
- LIVE-C → TID2013
- etc.

Results will be saved in `log/livec/`

## 💾 Model Checkpoints

Pre-trained model checkpoints will be available soon. They will include:

- ✅ Single-dataset models
- ✅ Cross-dataset models
- ✅ Single-distortion-type models
- 🙌 coming soon

**Download links:** Coming soon

## 📊 Results

### Performance on Individual Datasets

![image-Result on individual datasets](ipic\image-20260310143536086.png)

## 📖 Citation

If you find this work useful for your research, please consider citing:

```bibtex

```

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Contact: [hongyuwang@stu.scau.edu.cn]

## 🔄 Updates

- **2026-03-22**: Initial release

---

**Star ⭐ this repository if you find it helpful!**
