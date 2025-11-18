# QM9 Molecular Conformer Generation

This directory contains the implementation for molecular conformer generation experiments on the QM9 dataset using our Lie group representation diffusion framework.

## Overview

We apply our generalized score matching approach to generate molecular conformers by modeling $G=(SO(3) \times \mathbb{R}^+)^N $-guided transformations for a molecule with $N$ atoms.
Specifically, each point/atom is subject to a rotation and dilation transformation to move it in space.
The method operates on molecular graphs and learns to generate realistic 3D molecular structures, and we compare against standard Fisher score matching $G=T(3)^N$.

## Dataset

The QM9 dataset contains ~134k small organic molecules with up to 9 heavy atoms (C, N, O, F). Each molecule includes:
- Atomic coordinates and types
- Quantum mechanical properties
- Ground truth conformations

The dataset is automatically downloaded and processed during the first run.

## Usage

### Training

Run the training script from the repository root:

```bash
mamba activate sism
python sism/qm9/run_train.py --conf config/base_qm9.yaml
```

### Configuration

The main configuration file is `config/base_qm9.yaml`. Key parameters include:
- Model architecture settings
- Training hyperparameters 

### Key Files

- `run_train.py` - Main training script
- `model.py` - Lie group diffusion model implementation
- `data.py` - QM9 dataset loading and preprocessing
- `utils.py` - Utility functions for molecular operations

## Results

Our approach demonstrates that both Fisher score matching model (`fisher_dsm=True`) and our proposed generalized score matching model (`fisher_dsm=false`) are able to perform conformer generation. (See details in the Paper.)