# S2B: Sketch-to-BERT Classification System

This repository contains reproducible experimental code for the paper currently under review at ICDCS 2026.

## 📋 Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Environment Variables Configuration](#environment-variables-configuration)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Overview

S2B is a classification framework that integrates sketch-based filtering and BERT-based models, and operates in three modes:
- **S2B-C**: Cascade mode (Sketch → BERT)
- **S2B-S**: Sketch-only mode
- **S2B-B**: BERT-only mode

## Environment Setup

### 1. Python Version
- Python 3.12.2

### 2. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the following packages:

```
pandas==2.3.3
pyarrow==23.0.0
python-dotenv==1.2.1
scikit-learn==1.8.0
torch==2.5.1+cu124
torchaudio==2.5.1+cu124
torchvision==0.20.1+cu124
tqdm==4.67.1
transformers==4.57.6
```

**Note**: The PyTorch packages in `requirements.txt` are built with CUDA 12.4. If you need a different CUDA version or CPU-only installation, modify the PyTorch package versions accordingly.

### 3. Hugging Face Model Cache

BERT models are cached in the `./hf_cache` directory. You may either pre-download the required models or remove the `local_files_only=True` option in `models/s2b.py` (line 25) and `models/pl_bert_model.py` (line 4).

## Environment Variables Configuration

Create a `.env` file in the project root and configure the following environment variables.

### Required Environment Variables

#### Basic Settings
```bash
# Dimension settings
DIM=30                    # packet max length

# Number of labels
LABEL=120                   # Number of classification classes

# Mode settings
MODE=s2b-c                 # Execution mode: s2b-c, s2b-s, s2b-b
OPENWORLD=False            # Open-world setting: True/False
IS_TRAIN=True              # Training mode: True (training), False (inference)

# Device
DEVICE=cuda                # cuda or cpu
```

#### Data Paths
```bash
# Preprocessed data (parquet format)
TRAIN_DF=/path/to/train.parquet
VALID_DF=/path/to/valid.parquet
TEST_DF=/path/to/test.parquet
OPEN_TEST_DF=/path/to/openworld_test.parquet

# Raw data (before preprocessing)
RAW_TRAIN_DF=/path/to/raw_train.csv
RAW_VALID_DF=/path/to/raw_valid.csv
RAW_TEST_DF=/path/to/raw_test.csv
OPEN_RAW_TEST_DF=/path/to/raw_openworld_test.csv

# Raw data column names
RAW_X_LABEL=pcaket_length      # Input data column name
RAW_Y_LABEL=label              # Label column name

# Dataset path
DATASET_PATH=/path/to/dataset
```

#### BERT Model Settings
```bash
# Base BERT model (Hugging Face model name)
BASE_BERT_MODEL=bert-base-uncased

# Trained model path (required for inference)
BERT_MODEL_PATH=/path/to/bert_model
```

#### Training Hyperparameters

**Sketch Model**
```bash
SKETCH_BATCH_SIZE=32
SKETCH_LR=0.001
SKETCH_EPOCH=10
```

**PL-BERT Model**
```bash
PL_BERT_BATCH_SIZE=32
PL_BERT_EPOCH=10
PL_BERT_LR=0.00005
```

#### Output and Model Paths
```bash
# Output directory (models are saved here during training)
OUTPUT_PATH=./outputs

# Trained Sketch model path (required for inference)
SKETCH_MODEL_PATH=/path/to/sketch_model.pth
```

### Example .env File

```bash
# Basic settings
DIM=30
LABEL=120
MODE=s2b-c
OPENWORLD=False
IS_TRAIN=True
DEVICE=cuda

# Data paths
TRAIN_DF=./datas/train.parquet
VALID_DF=./datas/valid.parquet
TEST_DF=./datas/test.parquet
OPEN_TEST_DF=./datas/openworld_test.parquet

RAW_TRAIN_DF=./datas/raw_train.csv
RAW_VALID_DF=./datas/raw_valid.csv
RAW_TEST_DF=./datas/raw_test.csv
OPEN_RAW_TEST_DF=./datas/raw_openworld_test.csv

RAW_X_LABEL=x
RAW_Y_LABEL=y
DATASET_PATH=./datas

# BERT settings
BASE_BERT_MODEL=bert-base-uncased
BERT_MODEL_PATH=./outputs/1234567890/bert_model

# Training hyperparameters
SKETCH_BATCH_SIZE=32
SKETCH_LR=0.001
SKETCH_EPOCH=10

PL_BERT_BATCH_SIZE=16
PL_BERT_EPOCH=3
PL_BERT_LR=0.00005

# Output paths
OUTPUT_PATH=./outputs
SKETCH_MODEL_PATH=./outputs/1234567890/learned_sketch_model.pth
```

## Data Preparation

### 1. Raw Data Format

Raw data should be in CSV format. The column names in your CSV file **must exactly match** the values you set in the `.env` file for `RAW_X_LABEL` and `RAW_Y_LABEL`.

**Example:**
- If `.env` has `RAW_X_LABEL=packet_length` and `RAW_Y_LABEL=label`
- Then your CSV file must have columns named `packet_length` and `label`

The columns should contain:
- `{RAW_X_LABEL}`: Input data (packet length sequence)
- `{RAW_Y_LABEL}`: Label (integer type)

### 2. Data Preprocessing

Preprocessed data is saved in parquet format. You can use the `preprocess()` function to perform preprocessing (currently commented out in `main.py`).

## Usage

### 1. Training

Set `IS_TRAIN=True` in the `.env` file and run:

```bash
python main.py
```

Training process:
1. Train frequency filter
2. Train learned filter (Sketch model)
3. Train PL-BERT model

Trained models are saved in the `OUTPUT_PATH/{timestamp}/` directory:
- `learned_sketch_model.pth`: Sketch model
- `checkpoint-{batch}/`: BERT model directory

### 2. Inference

Set `IS_TRAIN=False` in the `.env` file and specify the trained model paths:

```bash
# Modify .env file
IS_TRAIN=False
SKETCH_MODEL_PATH=./outputs/1234567890/learned_sketch_model.pth
BERT_MODEL_PATH=./outputs/1234567890/checkpoint-{batch}
```

Then run:
```bash
python main.py
```

Inference process:
1. Load models
2. Set Sketch OOD threshold
3. Set Sketch finalize threshold
4. Set PL-BERT OOD threshold
5. Perform inference on test data

### 3. Mode-Specific Behavior

#### s2b-c (Cascade)
- First attempts classification with Sketch model
- Re-classifies with BERT model if confidence is low
- Includes OOD detection when open-world is enabled

#### s2b-s (Sketch-only)
- Uses only Sketch model
- Fastest inference speed

#### s2b-b (BERT-only)
- Uses only BERT model
- Highest accuracy (typically)

## Troubleshooting

### Model Loading Errors
- Verify model paths are correct
- Check that model files exist
- Verify `IS_TRAIN` setting is correct

### Out of Memory
- Reduce `PL_BERT_BATCH_SIZE` and `SKETCH_BATCH_SIZE`
- Change to `DEVICE=cpu` to use CPU

### Preprocessing Errors
- Verify parquet file paths
- Check data format: column names in the raw data must match the values specified in `.env` file (`RAW_X_LABEL` and `RAW_Y_LABEL`)