# WTHaar-Net: A Hybrid Classical–Quantum Approach (ECCV 2026)

**Source code for the ECCV 2026 paper _“WTHaar-Net: A Hybrid Classical–Quantum Approach”_**

📄 **Paper link:** _To be added_

---

## Repository Overview

This repository contains the official implementation of **WTHaar-Net**, including both classical and hybrid quantum–classical components.

---

## Quantum Implementation (QCHT)

The **`QCHT`** folder contains the implementation of the **Quantum–Classical Hadamard Transform (QCHT)** executed on the **IBM Quantum** cloud platform:

- **Platform:** [IBM Quantum](https://quantum-computing.ibm.com/)
- **Framework:** Qiskit
- **Purpose:** Quantum validation of the Hadamard Transform

The results demonstrate that **QCHT produces exactly the same outputs as the classical Hadamard Transform**, confirming its correctness.

---

## Folder Structure

### `QCHT`
- Quantum testing code using **Qiskit**.
- Implements the hybrid **Quantum–Classical Hadamard Transform**.

### `CIFAR10`
- Training and testing code on the **CIFAR-10** dataset.
- Supported models:
  - **WTHaar-Net**
  - **Hadamard**
  - **WHT**
  - **ResNet**

### `TinyImageNet`
- Training and testing code on the **TinyImageNet** dataset.
- Supported models:
  - **WTHaar-Net**
  - **Hadamard (natural)**
  - **ResNet**

---

## Running Experiments

### 1. Dataset Selection

Select one of the following datasets:

- **TinyImageNet**

- **ImageNet**

---

### 2. Network Selection

Select one of the following network architectures:

- **wthaar_resnet50** - Input size: **224 × 224**

- **wht_resnet50** - Input size: **224 × 224**

- **resnet50** - Input size: **224 × 224**

> **Note:** **WHTResNet50x3** denotes the **3-path HT-ResNet50** architecture (Input size: **224 × 224**).

---

### Installation

Navigate to your selected experiment folder and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

### Training

To train a model, run `main.py` with the desired model architecture and the path to the dataset:

```bash
python main.py -a wthaar_resnet50 -b 128
```

---

### Testing

To test a trained model, run `test.py` flag:

```bash
python test.py -a wthaar_resnet50 -b 128 -b10 32
```

---

### Use Dummy Data

ImageNet dataset is large and time-consuming to download. To get started quickly, run `main.py` using dummy data by "--dummy". It's also useful for training speed benchmark. Note that the loss or accuracy is useless in this case.

```bash
python main.py -a resnet18 --dummy
```
---

### Multi-processing Distributed Data Parallel Training

You should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

### Single node, multiple GPUs:

```bash
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
```

### Multiple nodes:

Node 0:

```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
```

Node 1:

```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]
```

## Usage

```bash
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE] [--rank RANK]
               [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed] [--dummy]
               [DIR]

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset (default: imagenet)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: alexnet | convnext_base | convnext_large | convnext_small | convnext_tiny | densenet121 | densenet161 | densenet169 | densenet201 | efficientnet_b0 |
                        efficientnet_b1 | efficientnet_b2 | efficientnet_b3 | efficientnet_b4 | efficientnet_b5 | efficientnet_b6 | efficientnet_b7 | googlenet | inception_v3 | mnasnet0_5 | mnasnet0_75 |
                        mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | mobilenet_v3_large | mobilenet_v3_small | regnet_x_16gf | regnet_x_1_6gf | regnet_x_32gf | regnet_x_3_2gf | regnet_x_400mf | regnet_x_800mf |
                        regnet_x_8gf | regnet_y_128gf | regnet_y_16gf | regnet_y_1_6gf | regnet_y_32gf | regnet_y_3_2gf | regnet_y_400mf | regnet_y_800mf | regnet_y_8gf | resnet101 | resnet152 | resnet18 |
                        resnet34 | resnet50 | resnext101_32x8d | resnext50_32x4d | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | vit_b_16 | vit_b_32 | vit_l_16 | vit_l_32 | wide_resnet101_2 | wide_resnet50_2 (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel
                        training
  --dummy               use fake data to benchmark

```