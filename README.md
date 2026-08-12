# TGO-IV — Developmental Topology Observatory

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2608.09997)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](https://pytorch.org/)

**TGO-IV — Developmental Topology Observatory** is the fourth observatory of the **Transformer Geometry Observatory (TGO)** framework.

TGO-IV studies the topological evolution of Vision Transformer representations across **network depth** and **training maturity**. The central object of analysis is the token representation point cloud produced by each Transformer layer.

The study uses **Persistent Homology** and **Vietoris–Rips filtrations** to characterize how the topology of these representation point clouds changes throughout the network and during training.

## Paper

**Transformer Geometry Observatory — TGO-IV: Developmental Topology Observatory**

[arXiv:2608.09997](https://arxiv.org/abs/2608.09997)

## Core Idea

For a Vision Transformer layer $l$, the token representations are represented as

$$
X^{(l)} = {x_1^{(l)}, x_2^{(l)}, \ldots, x_N^{(l)}},
\qquad x_i^{(l)} \in \mathbb{R}^{D}.
$$

For ViT-Small/16 at $224 \times 224$:

$$
N = 197,
\qquad
D = 384.
$$

Thus, each Transformer layer produces a point cloud of **197 token representations in a 384-dimensional embedding space**.

TGO-IV operates directly on this representation space rather than first projecting the representations using PCA, t-SNE, or UMAP.

For a filtration parameter $\epsilon$, the Vietoris–Rips complex is constructed from the pairwise distances between representation points:

$$
|x_i-x_j| \leq \epsilon.
$$

As $\epsilon$ increases, the Vietoris–Rips complexes form a filtration:

$$
VR(\epsilon_1)
\subseteq
VR(\epsilon_2)
\subseteq
\cdots
\subseteq
VR(\epsilon_m).
$$

Persistent Homology tracks when topological structures **appear, persist, and disappear** across this filtration.

The resulting persistence structure provides a representation of the topology of the observed Transformer point cloud.

## What TGO-IV Measures

TGO-IV studies the representation point cloud through several topological observables:

* **Persistence Diagrams**
* **Barcode Diagrams**
* **Betti Curves**
* **Persistence Landscapes**
* **Bottleneck Distance**
* **Wasserstein Distance**

The primary homology dimensions examined are

$$
H_0,\quad H_1,\quad H_2.
$$

Here:

* $H_0$ captures connected components.
* $H_1$ captures loop-like structures.
* $H_2$ captures higher-dimensional cavities.

## Development Across Depth

For a fixed training checkpoint, TGO-IV computes the persistent topology of the representation produced at each Transformer layer.

The resulting layer-wise topological signatures are compared to determine how the representation changes as information propagates through the network.

For consecutive layers $l$ and $l+1$, the topological difference is quantified using distances between their persistence diagrams.

The Bottleneck Distance is defined as

$$
d_B(D_1,D_2)

\inf_{\gamma}
\sup_{x \in D_1}
|x-\gamma(x)|_\infty.
$$

The Wasserstein Distance provides a complementary measure of the overall discrepancy between persistence diagrams.

Together, these measures quantify how strongly the topology of the representation changes between neighbouring Transformer layers.

## Development Across Training

TGO-IV also studies the temporal evolution of representation topology.

The same analysis is repeated at multiple training checkpoints:

```text
Epoch 1
Epoch 5
Epoch 10
Epoch 20
Epoch 50
Epoch 100
```

This allows the evolution of persistent topological structure to be studied as the network moves from initialization toward a trained representation.

The analysis therefore considers two developmental axes:

```text
                    Training
                       ↑
                       │
                       │
                       │
Depth ─────────────────────────→
```

More explicitly:

```text
Transformer Depth
       │
       ▼
Layer-wise Representation
       │
       ▼
Token Point Cloud
       │
       ▼
Vietoris–Rips Filtration
       │
       ▼
Persistent Homology
       │
       ├── Persistence Diagrams
       ├── Barcode Diagrams
       ├── Betti Curves
       ├── Persistence Landscapes
       ├── Bottleneck Distance
       └── Wasserstein Distance
```

The resulting observables are then analyzed across training checkpoints.

## Experimental Configuration

The primary experimental configuration is:

| Component                  | Configuration          |
| -------------------------- | ---------------------- |
| Architecture               | ViT-Small/16           |
| Dataset                    | ImageNet-100           |
| Input Resolution           | $224 \times 224$       |
| Number of Tokens           | 197                    |
| Embedding Dimension        | 384                    |
| Transformer Blocks         | 12                     |
| Training Epochs            | 100                    |
| Analysis Set               | 1000 validation images |
| Detailed Topology Analysis | Fixed validation image |
| Batch Size                 | 128                    |
| Learning Rate              | $10^{-3}$              |
| Weight Decay               | 0.05                   |
| Scheduler                  | Cosine Annealing       |
| Minimum Learning Rate      | $10^{-6}$              |
| Label Smoothing            | 0.1                    |
| Gradient Clipping          | 1.0                    |
| Mixed Precision            | AMP                    |
| Hardware                   | NVIDIA Quadro RTX 6000 |

A fixed validation subset is used throughout the longitudinal analysis to maintain consistency between training checkpoints.

## Topological Observables

### Persistence Diagrams

A persistence diagram represents each topological feature by its birth and death filtration values:

$$
(b_i,d_i).
$$

The persistence of a feature is

$$
d_i-b_i.
$$

Features with larger persistence survive over a larger range of filtration scales.

### Barcode Diagrams

Barcode representations display each persistent feature as an interval:

$$
[b_i,d_i).
$$

This provides a direct visualization of the lifetime of topological structures across the filtration.

### Betti Curves

For homology dimension $k$, the Betti number at filtration value $\epsilon$ is

$$
\beta_k(\epsilon).
$$

It represents the number of $k$-dimensional topological features present at that filtration scale.

TGO-IV primarily analyzes

$$
\beta_0(\epsilon),\quad
\beta_1(\epsilon),\quad
\beta_2(\epsilon).
$$

### Persistence Landscapes

Persistence diagrams are additionally represented using persistence landscapes, providing a functional representation of persistent topology that can be compared across layers and training checkpoints.

### Bottleneck Distance

The Bottleneck Distance emphasizes the largest topological discrepancy between two persistence diagrams.

TGO-IV uses it to quantify the maximum persistent-topological change between representation layers.

### Wasserstein Distance

The Wasserstein Distance measures the aggregate discrepancy between persistence diagrams.

It complements the Bottleneck Distance by capturing the overall redistribution of persistent features.

## Main Observations

TGO-IV investigates several empirical observations concerning the developmental topology of Transformer representations.

### Topological Stabilization

The topological distance between neighbouring Transformer layers generally decreases as training progresses.

This indicates that the persistent topological signatures of successive representations become increasingly similar during optimization.

However, the evolution is not perfectly uniform across depth. Localized changes remain around specific layer transitions.

### Increasing Representation Connectivity

The dominant topological signal is observed in $H_0$.

As training progresses, connected components tend to merge at smaller filtration scales.

This corresponds to increasing connectivity within the observed representation point cloud.

### Limited Persistent Higher-Order Structure

Persistent $H_1$ structures remain comparatively sparse.

Similarly, persistent $H_2$ structures are limited in the observed representation point clouds.

The dominant topological evolution is therefore associated primarily with **connectivity rather than large-scale higher-order cavities**.

### Layer-wise Topological Reorganization

Although the overall topological differences between neighbouring layers decrease during training, specific layer transitions continue to exhibit stronger topological changes.

This suggests that representation development is not uniformly distributed across Transformer depth.

## Developmental Topology Hypothesis

The observations motivate the following empirical hypothesis:

> Transformer representations undergo progressive topological stabilization during training, with the representation point cloud becoming increasingly cohesive while retaining non-trivial internal structure. The magnitude of topological transformation between neighbouring layers decreases with training maturity, while localized layer transitions continue to perform stronger representational reorganization.

This is an empirical hypothesis derived from the observed persistent topological signatures.

## Relationship to the TGO Framework

TGO-IV extends the previous observatories of the Transformer Geometry Observatory.

```text
TGO-I
Spectral Geometry Observatory
        │
        ▼
TGO-II
Representation Geometry Observatory
        │
        ▼
TGO-III
Semantic Geometry Observatory
        │
        ▼
TGO-IV
Developmental Topology Observatory
```

The observatories examine different properties of Transformer representations:

$$
\text{Spectral}
\rightarrow
\text{Geometric}
\rightarrow
\text{Semantic}
\rightarrow
\text{Topological}.
$$

TGO-IV therefore does not replace the measurements of the previous observatories.

It adds a topological perspective to the analysis of representation development.

## Methodological Consideration

Each ViT-Small/16 layer provides only

$$
197
$$

token representations in

$$
\mathbb{R}^{384}.
$$

The resulting point cloud is therefore a sparse sample of the underlying representation structure.

Consequently, the persistent homology computed by TGO-IV should not be interpreted as a guaranteed reconstruction of an underlying representation manifold.

The analysis instead focuses on the **comparative evolution of persistent topological signatures of the observed representation point clouds** across depth and training.

## Repository Structure

```text
TGO-IV/
│
├── configs/
│   └── ...
│
├── models/
│   └── ...
│
├── topology/
│   ├── persistence_diagrams/
│   ├── barcode_diagrams/
│   ├── betti_curves/
│   ├── persistence_landscapes/
│   ├── bottleneck_distance/
│   └── wasserstein_distance/
│
├── experiments/
│   ├── layerwise/
│   ├── training/
│   └── longitudinal/
│
├── analysis/
│   └── ...
│
├── results/
│   └── ...
│
├── requirements.txt
└── README.md
```

## Reproducibility

The experiments use fixed configurations, deterministic evaluation subsets, and recorded training checkpoints to enable comparison of representation topology across training.

The principal longitudinal checkpoints are:

```text
1 → 5 → 10 → 20 → 50 → 100
```

At each checkpoint, the layer-wise token representations are extracted from the Vision Transformer and passed through the topological analysis pipeline.

For more details and in-depth review of the entire method, please check my pre-print out on arXiv: [https://arxiv.org/abs/2608.09997]. As always, I would appreciate your inputs and feedbacks; of you find this study useful in you own work, please consider citing. :)

## Citation

If you use TGO-IV or the associated experimental methodology, please cite:

```bibtex
@article{kapil2026tgoiv,
  title   = {Transformer Geometry Observatory TGO-IV: Developmental Topology Observatory},
  author  = {Kapil, Kaustubh and Upla, Kishor P.},
  year    = {2026},
  eprint  = {2608.09997},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

