# TGO-IV — Developmental Topology Observatory

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2608.09997)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![Domain](https://img.shields.io/badge/Domain-Topological%20Data%20Analysis-blue)](https://en.wikipedia.org/wiki/Topological_data_analysis)

TGO-IV: **Developmental Topology Observatory** is the fourth observatory in the **Transformer Geometry Observatory (TGO)** framework.

TGO-IV studies how the topology of Vision Transformer representations develops across network depth and training time. Rather than treating intermediate representations as isolated objects, it models the token embeddings produced by each Transformer layer as a representation point cloud and tracks the evolution of its persistent topological signatures.

The study uses **Persistent Homology** and **Vietoris–Rips simplicial complexes** to characterize this evolution directly in the original embedding space.

## Paper

**TGO-IV: Developmental Topology Observatory**

Kaustubh Kapil, Kishor P. Upla

[Paper — arXiv:2608.09997](https://arxiv.org/abs/2608.09997)

## Motivation

Previous observatories in the TGO framework examined Transformer representations from different perspectives:

* **TGO-I — Spectral Geometry Observatory:** spectral evolution, covariance eigenspectra, Effective Rank, Stable Rank, and Spectral Entropy.
* **TGO-II — Representation Geometry Observatory:** representational similarity and intrinsic dimensionality.
* **TGO-III — Semantic Geometry Observatory:** semantic organization and class-level representation structure.
* **TGO-IV — Developmental Topology Observatory:** persistent topology and topological evolution.

These observatories provide complementary descriptions of representation development.

TGO-IV asks a different question:

> **Does the topology of Transformer representations evolve systematically during the forward pass and throughout training?**

In particular, it investigates whether topological transformations are distributed continuously across depth or concentrated around particular Transformer layers, and whether the persistent topology of neighbouring representations becomes more stable as optimization progresses.

## Core Idea

For a Vision Transformer layer (l), the token representations are represented as

[
X^{(l)} =
{x_1^{(l)},x_2^{(l)},\ldots,x_N^{(l)}},
\qquad
x_i^{(l)}\in\mathbb{R}^{D}.
]

For ViT-Small/16 at (224\times224):

[
N=197,\qquad D=384.
]

Thus, each layer produces a point cloud of 197 token representations in a 384-dimensional embedding space.

Instead of projecting this point cloud using PCA, t-SNE, or UMAP, TGO-IV operates directly in the original representation space.

A Vietoris–Rips complex is constructed using a filtration parameter (\epsilon):

[
|x_i-x_j|\leq\epsilon
]

defines connectivity between points.

Increasing (\epsilon) produces a filtration

[
VR(\epsilon_1)
\subseteq
VR(\epsilon_2)
\subseteq
\cdots
\subseteq
VR(\epsilon_m).
]

Persistent Homology then tracks when topological structures appear, persist, and disappear across this filtration.

## Topological Observatories

TGO-IV implements a collection of complementary observables.

### Persistence Diagrams

Represent topological features using birth and death filtration values:

[
(b_i,d_i).
]

The persistence of a feature is related to

[
d_i-b_i.
]

Longer-lived features correspond to structures that persist across a larger range of filtration scales.

### Barcode Diagrams

Represent each persistent feature as an interval from its birth scale to its death scale.

TGO-IV examines the evolution of:

* (H_0): connected components
* (H_1): loops
* (H_2): higher-dimensional cavities

### Betti Curves

For homology dimension (k),

[
\beta_k(\epsilon)
]

counts the number of (k)-dimensional topological features present at filtration radius (\epsilon).

The primary observables are:

[
\beta_0,\quad\beta_1,\quad\beta_2.
]

### Persistence Landscapes

Persistence diagrams are converted into functional representations suitable for comparing persistent topology across layers and training epochs.

### Bottleneck Distance

The Bottleneck Distance measures the largest discrepancy between two persistence diagrams:

[
d_B(D_1,D_2).
]

TGO-IV uses it to quantify the largest topological change between consecutive Transformer layers.

### Wasserstein Distance

The Wasserstein Distance measures the cumulative discrepancy between persistence diagrams.

It therefore complements the Bottleneck Distance by considering the overall distribution of topological changes rather than only the largest one.

## Experimental Configuration

The primary experimental setup is:

| Component                  | Configuration          |
| -------------------------- | ---------------------- |
| Architecture               | ViT-Small/16           |
| Dataset                    | ImageNet-100           |
| Input resolution           | (224\times224)         |
| Transformer blocks         | 12                     |
| Tokens                     | 197                    |
| Embedding dimension        | 384                    |
| Training epochs            | 100                    |
| Analysis subset            | 1000 validation images |
| Detailed topology tracking | 1 validation image     |
| Optimizer                  | AdamW                  |
| Initial learning rate      | (10^{-3})              |
| Weight decay               | 0.05                   |
| Label smoothing            | 0.1                    |
| Gradient clipping          | 1.0                    |
| Scheduler                  | Cosine Annealing       |
| Minimum learning rate      | (10^{-6})              |
| Mixed precision            | AMP                    |
| Hardware                   | NVIDIA Quadro RTX 6000 |

The same fixed validation subset is used throughout training to preserve longitudinal consistency.

Selected epochs are retained for temporal analysis:

[
{1,5,10,20,50,100}.
]

## Analysis Pipeline

The TGO-IV pipeline follows:

```text
Vision Transformer
       │
       ▼
Layer-wise Token Representations
       │
       ▼
Representation Point Cloud
       │
       ▼
Pairwise Distances
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
       │
       ▼
Developmental Topology Analysis
       │
       ├── Across Transformer Depth
       └── Across Training Time
```

## Main Observations

TGO-IV reports several consistent developmental trends.

### 1. Progressive Topological Stabilization

The Bottleneck Distance between consecutive Transformer layers decreases throughout training.

Early in training, neighbouring layers exhibit substantially different persistent topological signatures.

As optimization progresses, these differences become smaller.

However, localized peaks remain around particular layer transitions, notably around Layers 3, 10, and 12.

This indicates that topological evolution does not become uniformly distributed across the network.

### 2. Increasing Representation Connectivity

The topology is dominated by (H_0).

Across training, the (\beta_0) curves shift toward smaller filtration radii.

Therefore, connected components merge earlier during the filtration.

This corresponds to an increasingly cohesive representation point cloud.

### 3. Limited Persistent (H_1) Structure

(H_1) features remain comparatively sparse.

Loops appear over limited filtration ranges, but there is no substantial increase in persistent loop structures throughout training.

### 4. Sparse Higher-Order Topology

(H_2) is largely absent in the primary observations, with higher-order features remaining comparatively limited.

The analysis therefore suggests that the dominant persistent topological evolution is associated with connectivity rather than large-scale higher-order cavities.

### 5. Decreasing Inter-Layer Topological Difference

Both Bottleneck and Wasserstein distances decrease as training progresses.

The Wasserstein Distance exhibits a smoother depth-wise evolution, while localized peaks remain at particular Transformer transitions.

Together, these observations indicate progressively smaller topological modifications between successive representations as optimization approaches convergence.

## Progressive Topological Stabilization Hypothesis

The observations motivate the following empirical hypothesis:

> **Transformer representations evolve through progressive topological stabilization, in which the representation point cloud becomes increasingly compact while preserving non-trivial geometric organization. During optimization, higher-order topological structures are transiently established and subsequently refined, whereas successive Transformer layers perform progressively smaller cumulative topological transformations as convergence is approached.**

This hypothesis is an interpretation of the persistent topology observed in the constructed simplicial complexes.

It is not a claim that the experiments recover the intrinsic topology of the underlying representation manifold.

## Important Methodological Constraint

A critical limitation of this experiment is the dimensionality and sparsity of the point cloud.

Each ViT-Small/16 layer provides only

[
197
]

token representations in

[
\mathbb{R}^{384}.
]

This is an extremely sparse sampling of a potentially high-dimensional representation structure.

Consequently, the Vietoris–Rips complexes constructed in TGO-IV should **not** be interpreted as faithful reconstructions of the true representation manifold.

The topology measured by TGO-IV is therefore treated as a **topological measurement paradigm over the observed representation point cloud**.

The claims of this repository concern the comparative evolution of persistent topological signatures across layers and training epochs.

## Relation to the TGO Framework

TGO-IV extends the previous observatories:

```text
TGO-I
Spectral Geometry
        │
        ▼
TGO-II
Representation Geometry
        │
        ▼
TGO-III
Semantic Geometry
        │
        ▼
TGO-IV
Developmental Topology
```

The resulting perspective is:

[
\text{Spectrum}
\rightarrow
\text{Geometry}
\rightarrow
\text{Semantics}
\rightarrow
\text{Topology}.
]

Each observatory measures a different property of Transformer representation development.

TGO-IV does not replace the preceding observatories. It adds a topological perspective that is not captured by spectral statistics, representational similarity, intrinsic dimensionality, or semantic separability alone.

## Repository Structure

The repository is organized around the experimental stages of TGO-IV:

```text
TGO-IV/
│
├── configs/
│   └── ...
│
├── data/
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

The exact implementation structure may differ depending on the experiment configuration.

## Reproducibility

The experimental protocol fixes the evaluation subsets and random seed so that representation topology can be compared consistently across training.

The principal longitudinal checkpoints are:

```text
Epoch 1
Epoch 5
Epoch 10
Epoch 20
Epoch 50
Epoch 100
```

All topological measurements are performed from the layer-wise token representations generated by the trained Vision Transformer. 

For more details and in-depth review of the entire method, please check my pre-print out on arXiv: [https://arxiv.org/abs/2608.09997]. As always, I would appreciate your inputs and feedbacks; of you find this study useful in you own work, please consider citing. :)

## Citation

If you use this repository or the TGO-IV methodology, please cite:

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

