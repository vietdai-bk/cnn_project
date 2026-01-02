# Simple Image Classification Model (MNIST-style)

## Overview
This repository implements a **complete image classification pipeline** using PyTorch, including:

- Model training (`train.py`)
- Model evaluation (`test.py`)
- Single-image inference (`inference.py`)

The project is designed to be **simple, clean, and easy to extend**, suitable for MNIST-style datasets or custom image classification tasks.

---

## Dataset Structure

The dataset should be organized in a **folder-per-class** format:

```text
dataset_root/
├── class1/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── class2/
│   ├── image_001.jpg
│   └── ...
├── class3/
│   └── ...
```

Example for MNIST:

```text
mnist_dataset/
└── trainingSet/
    ├── 0/
    ├── 1/
    ├── 2/
    └── ...
```

---

## Environment Setup

It is recommended to use **conda** or **virtualenv**.

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py   --dataset path/to/dataset   --epoch number/of/epochs  --checkpoint_folder checkpoints --batch_size num/batchsize   --lr learing rate   --SGD
```
Example:
```
python train.py --dataset mnist_dataset/trainingSet  --epoch 10 --batch_size 64 --lr 1e-4 --SGD
```

### Training Arguments

| Argument | Description |
|--------|------------- |
| `--dataset` | Path to the training dataset |
| `--epoch` | Number of training epochs |
| `--batch_size` | Batch size |
| `--lr` | Learning rate |
| `--SGD` | Use SGD optimizer (default is Adam) |
| `--checkpoint_folder` | Model trained save here |

---

## Evaluation

```bash
python test.py   --test_data mnist_dataset/trainingSet   --model_path checkpoints/SimpleModel.pt   --acc   --cfm
```

### Options
- `--acc` : Compute accuracy
- `--cfm` : Generate confusion matrix

---

## Inference

```bash
python inference.py   --model_path checkpoints/SimpleModel.pt   --image_path image_test/img_1.jpg
```

---

## Notes
- Model architecture is defined in `models/SimpleModel.py`
- Dataset loader is implemented in `datasets/MyDataset.py`
- The project can be easily adapted to other datasets by changing the dataset directory structure

---
