# MNIST Experiments - Symmetry-Induced Score Matching

This subdirectory contains implementations and experiments for applying symmetry-induced score matching techniques to MNIST digit data, focusing on rotation invariance and bridge diffusion models.

## Directory Structure

```
sism/mnist/
├── README.md                    # This file
├── __init__.py                  # Package initialization
├── mnist_data.py               # Data loading and preprocessing
├── train_bridge.py             # Bridge diffusion model training
├── train_gsm.py                # Generative Score Matching training
├── train_classifier.py         # MNIST classifier training
├── run_test_bridge.py          # Bridge model inference and evaluation
├── run_test_gsm.py             # GSM model inference and evaluation
├── visualize.ipynb             # Interactive visualization notebook
├── experiments/                # Trained model checkpoints and configs
├── MNIST/                      # Raw MNIST dataset
```

## Core Components

### Data Handling (`mnist_data.py`)

- **MNISTDataModule**: PyTorch Lightning data module with configurable transforms and batch processing

### Training Scripts

#### Bridge Diffusion Training (`train_bridge.py`)
Brownian Bridge Diffusion Models (BBDM) for MNIST reconstruction with configurable schedulers, MLP/CNN architectures, and context-aware diffusion.
This model predicts changes in the entire pixel space of dimension $\mathbb{R}^{28\times28}$

**Usage:**
```bash
mkdir experiments/
python train_bridge.py --batch_size 64 --lr 2e-4 --num_epochs 100 --save_dir experiments/BBDM
```

#### Generalized Score Matching Training (`train_gsm.py`)
Generalized score matching (GSM) model that only predicts the infinitesimal rotation angle $\theta \in \mathbb{R}^1$ to change the image.

**Usage:**
```bash
mkdir experiments/
python train_gsm.py --batch_size 64 --lr 2e-4 --num_epochs 100 --use-conv --save_dir experiments/GSM
```

#### MNIST Classifier Training (`train_classifier.py`)
CNN classifier with two convolutional blocks (32, 64 filters) and fully connected layers for digit recognition.

**Usage:**
```bash
mkdir experiments/
python train_classifier.py --batch_size 64 --lr 1e-3 --num_epochs 10 --save_dir experiments/classifier
```

### Evaluation and Testing

#### Bridge Model Testing (`run_test_bridge.py`)
Evaluation script with model loading, test image generation, bridge sampling, and visualization capabilities.

**Usage:**
```bash
# Basic testing, only reconstruct 10 digits
python run_test_bridge.py --model-path experiments/BBDM/models/model_id_0_sin_epoch_99_T=1000.ckpt \
    --T 1000 \
    --device cpu \
    --save-path experiments/BBDM \
    --run-test
# Generate more images. Generates 800 images per digit class, totaling 8000 images.
python run_test_bridge.py --model-path experiments/BBDM/models/model_id_0_sin_epoch_99_T=1000.ckpt \
    --T 1000 \
    --device cpu \
    --save-path experiments/BBDM \
```

#### GSM Model Testing (`run_test_gsm.py`)
Evaluation for Group-based Score Matching with rotation reconstruction, denoising, and trajectory visualization.

```bash
# Basic testing, only reconstruct 10 digits
python run_test_gsm.py --model-path experiments/GSM/models/model_epoch_99_T=100.ckpt \
    --T 10 \
    --device cpu \
    --save-path experiments/GSM \
    --run-test
# Generate more images. Generates 800 images per digit class, totaling 8000 images.
python run_test_gsm.py --model-path experiments/GSM/models/model_epoch_99_T=100.ckpt \
    --T 10 \
    --device cpu \
    --save-path experiments/GSM_tmp \
```
#### Evaluate

Evaluates generated samples from (`run_test_bridge.py`) and (`run_test_gsm.py`) by computing the predictiona accuracy of samples digits if they follow the ground-truth digit.  
Furthermore, FID score is computed.


```bash
python evaluate_bridge_gsm.py --gsm-samples experiments/GSM/generated_valset.pth --bridge-samples experiments/BBDM/generated_valset.pth --classifier experiments/classifier/mnist_classifier_best.pth
```

Should return

| Method | Average Accuracy | Average FID |
|--------|------------------|-------------|
| GSM | 0.9280 ± 0.0402 | 130.8464 ± 21.8529 |
| Bridge | 0.8048 ± 0.0963 | 133.3733 ± 18.9774 |

## Visualization

- **Interactive Notebook** (`visualize.ipynb`): Model output visualization