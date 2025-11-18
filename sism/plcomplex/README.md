# CrossDocked2020 Molecular Docking

This directory contains the implementation for protein-ligand docking experiments on the CrossDocked2020 dataset using our Lie group representation diffusion framework.

## Overview

We apply our generalized score matching approach to model **global $SE(3)$ transformations** of entire ligands for molecular docking. Unlike the QM9 conformer generation experiment where each atom has individual degrees of freedom under $G=(SO(3) \times \mathbb{R}^+)^N$, here we treat the ligand as a rigid body subject to a single $SE(3)$ transformation (rotation and translation) to find optimal binding poses in protein binding sites.

This represents a different modeling paradigm:
- **QM9**: Individual atom flexibility for conformer generation ($N$ independent transformations)
- **CrossDocked**: Global ligand rigidity for docking (single $SE(3)$ transformation)

## Baseline Comparisons

We compare our Lie group approach (`TrainerSphere`) against:
- **Riemannian Score-Based Generative Models** (`TrainerRSGM`): Direct diffusion on the $SO(3)$ manifold with additional translation group $T(3)$
- **Unconstrained Brownian Bridge** (`TrainerFisherBridge`): Standard Fisher score matching where all $N$ atoms have individual degrees of freedom as $G=T(3)^N$

## Dataset

The CrossDocked2020 dataset contains protein-ligand complexes with:
- Protein structures and binding sites
- Ligand conformations and binding poses
- Ground truth docking configurations

**Note**: The processed dataset is currently not provided in this repository.

## Usage

### Training

Run the training script from the repository root:

```bash
mamba activate sism
python sism/plcomplex/run_train.py --conf config/base_crossdocked.yaml
```

### Configuration

The main configuration file is `config/base_crossdocked.yaml`. Key parameters include:
- Model architecture settings
- Training hyperparameters
- Global $SE(3)$ transformation specifications

### Key Files

- `run_train.py` - Main training script
- `model.py` - Lie group diffusion model implementation with forward and reverse dynamics
- `data.py` - CrossDocked2020 dataset loading and preprocessing
- `utils.py` - Utility functions for docking operations

## Results

Our approach demonstrates improved docking performance by modeling global $SE(3)$ transformations that respect the rigid-body constraints of protein-ligand interactions, focusing on finding optimal binding poses rather than internal conformational changes.