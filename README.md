
# TGO-IV: Developmental Geometry Observatory

This repository implements **TGO-IV**, a **Developmental Geometry Observatory** for ViT-Small/16 on ImageNet-100.

TGO-IV studies how the **representation manifold of a single image** develops through transformer layers, using three families of observables:

1. **Trajectory analysis**
2. **Jacobian analysis**
3. **Topology analysis**

The central research question is:

> **How does a Transformer learn?**

---

## What is analyzed

### 1) Trajectory analysis
- **Representation Trajectory**: one image, one pretrained model, layers
- **Population Trajectory**: all images, one pretrained model, layers
- **Learning Trajectory**: one image, epochs, per layer
- **Manifold Learning Trajectory**: dataset mean, epochs, per layer

### 2) Jacobian analysis
- **Local Deformation Field**
- **Jacobian Field Evolution**
- **Jacobian Heterogeneity**

### 3) Topology analysis
- **Persistence Diagram Evolution**
- **Barcode Evolution**
- **Betti Curve Evolution**
- **Bottleneck Distance Evolution**
- **Wasserstein Distance Evolution**

---

## Research motivation

- **TGO-I** studied representational motion.
- **TGO-II** studied local geometry.
- **TGO-III** studied semantic organization.
- **TGO-IV** studies the topology of the representation manifold and how it changes across layers and training checkpoints.

---

## Files

- `tgo_iv/main.py` — entry point
- `tgo_iv/config.py` — dataclass-based configuration
- `tgo_iv/dataset.py` — ImageNet-100 loaders and fixed subsets
- `tgo_iv/hooks.py` — ViT activation capture
- `tgo_iv/model.py` — ViT wrapper
- `tgo_iv/trajectories.py` — trajectory observables
- `tgo_iv/jacobian.py` — affine deformation / Jacobian observables
- `tgo_iv/topology.py` — persistent homology utilities
- `tgo_iv/metrics.py` — observable summaries and aggregation helpers
- `tgo_iv/visualization.py` — plotting helpers
- `tgo_iv/trainer.py` — training and observatory pipeline
- `tgo_iv/utils.py` — logging, seeding, JSON helpers

---

## Run

```bash
python -m tgo_iv.main --config configs/vit_small_imagenet100.yaml
```

Optional resume:

```bash
python -m tgo_iv.main \
    --config configs/vit_small_imagenet100.yaml \
    --resume results_tgo_iv/checkpoints/last.pth
```

---

## Expected data layout

```text
/path/to/imagenet100/train/<class_name>/*.JPEG
/path/to/imagenet100/val/<class_name>/*.JPEG
```

If you want a fixed class list, provide a text file with one class name per line and point `data.class_subset_file` to it.

---

## Output layout

```text
results_tgo_iv/
├── checkpoints/
│   ├── best.pth
│   └── last.pth
│
├── summaries/
│   ├── epoch_001.json
│   ├── epoch_002.json
│   └── ...
│
├── trajectories/
│   ├── representation/
│   ├── population/
│   ├── learning/
│   └── manifold_learning/
│
├── jacobians/
│   ├── local_deformation_field/
│   ├── jacobian_field_evolution/
│   └── heterogeneity/
│
├── topology/
│   ├── persistence_diagrams/
│   ├── barcodes/
│   ├── betti_curves/
│   ├── bottleneck_distance/
│   └── wasserstein_distance/
│
├── global_analysis/
└── logs/
```

---

## Notes

- The representation object for TGO-IV is **the token-level representation manifold of a single image**.
- For a ViT-S/16, one image produces a token matrix of shape approximately `197 x 384` at each layer.
- Persistent homology is computed on the token point cloud at each layer.
- Trajectory and Jacobian observables are tracked layer-wise; learning observables are tracked across epochs.
- If `ripser` / `persim` are unavailable, install the repository requirements first.

---

## Default experiment interpretation

- **Trajectory analysis**: how representations move.
- **Jacobian analysis**: how each layer locally deforms the token manifold.
- **Topology analysis**: how the manifold's global topology evolves through the network and during training.
