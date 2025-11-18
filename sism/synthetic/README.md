## Running the benchmarks

#### 2D toy distributions

```bash
python 2d_toydatasets.py
```

##### 3D toy distributions

```bash
python 3d_toydatasets.py
```

## Results

### Comparison of GSM and Fisher Score matching on 2D and 3D synthetic datasets

| Dataset | Method | MMD | W2 |
|---------|--------|-----|-----|
| MoG (2D) | GSM | 0.12 | 2.78 |
| MoG (2D) | Fisher | 0.08 | 1.61 |
| Concentric Circles (2D) | GSM | 0.11 | 2.05 |
| Concentric Circles (2D) | Fisher | 0.03 | 0.76 |
| Line (2D) | GSM | 0.11 | 0.57 |
| Line (2D) | Fisher | 0.06 | 0.39 |
| MoG (3D) | GSM | 0.19 | 1.50 |
| MoG (3D) | Fisher | 0.11 | 0.82 |
| Torus (3D) | GSM | 0.15 | 1.00 |
| Torus (3D) | Fisher | 0.06 | 0.57 |
| Möbius Strip (3D) | GSM | 0.26 | 0.47 |
| Möbius Strip (3D) | Fisher | 0.05 | 0.16 |