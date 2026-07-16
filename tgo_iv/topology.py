from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from ripser import ripser  # type: ignore
except Exception:  # pragma: no cover
    ripser = None

try:
    from persim import bottleneck as persim_bottleneck  # type: ignore
    from persim import wasserstein as persim_wasserstein  # type: ignore
except Exception:  # pragma: no cover
    persim_bottleneck = None
    persim_wasserstein = None


def compute_diagrams(X: np.ndarray, maxdim: int = 2, thresh: Optional[float] = None) -> List[np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D point cloud, got {X.shape}")
    if X.shape[0] < 2:
        return [np.zeros((0, 2), dtype=np.float64) for _ in range(maxdim + 1)]
    if ripser is None:
        raise ImportError(
            "ripser is not installed. Install the repository requirements to compute persistent homology."
        )
    kwargs = {"maxdim": maxdim}
    if thresh is not None:
        kwargs["thresh"] = thresh
    out = ripser(X, **kwargs)
    return [np.asarray(d, dtype=np.float64) for d in out["dgms"]]


def finite_pairs(diagram: np.ndarray) -> np.ndarray:
    diagram = np.asarray(diagram, dtype=np.float64)
    if diagram.size == 0:
        return diagram.reshape(0, 2)
    mask = np.isfinite(diagram[:, 1])
    return diagram[mask]


def diagram_max_value(diagrams: Sequence[np.ndarray], default: float = 1.0) -> float:
    max_val = 0.0
    for dgm in diagrams:
        dgm = np.asarray(dgm, dtype=np.float64)
        if dgm.size == 0:
            continue
        finite = dgm[np.isfinite(dgm[:, 1])]
        if finite.size == 0:
            continue
        max_val = max(max_val, float(np.max(finite)))
    return max(max_val, float(default))


def recommended_eps_grid(diagrams: Sequence[np.ndarray], num: int = 64, padding: float = 0.05, minimum: float = 1e-8) -> np.ndarray:
    max_val = diagram_max_value(diagrams, default=minimum)
    max_val = max(float(max_val) * (1.0 + float(padding)), float(minimum))
    return np.linspace(0.0, max_val, int(num), dtype=np.float64)


def persistence_lengths(diagram: np.ndarray) -> np.ndarray:
    diag = finite_pairs(diagram)
    if diag.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.maximum(diag[:, 1] - diag[:, 0], 0.0)


def persistence_entropy(diagram: np.ndarray, eps: float = 1e-12) -> float:
    lengths = persistence_lengths(diagram)
    if lengths.size == 0:
        return 0.0
    total = float(lengths.sum())
    if total <= 0:
        return 0.0
    p = lengths / total
    p = p[p > 0]
    return float(-(p * np.log(p + eps)).sum())


def diagram_statistics(diagrams: Sequence[np.ndarray]) -> dict:
    stats = {}
    for dim, dgm in enumerate(diagrams):
        lengths = persistence_lengths(dgm)
        stats[f"dim_{dim}_count"] = int(lengths.size)
        stats[f"dim_{dim}_mean_persistence"] = float(lengths.mean()) if lengths.size else 0.0
        stats[f"dim_{dim}_max_persistence"] = float(lengths.max()) if lengths.size else 0.0
        stats[f"dim_{dim}_entropy"] = persistence_entropy(dgm)
    return stats


def betti_curve(diagram: np.ndarray, eps_grid: np.ndarray) -> np.ndarray:
    diag = finite_pairs(diagram)
    eps_grid = np.asarray(eps_grid, dtype=np.float64)
    if diag.size == 0:
        return np.zeros_like(eps_grid, dtype=np.float64)
    births = diag[:, 0][:, None]
    deaths = diag[:, 1][:, None]
    alive = (births <= eps_grid[None, :]) & (eps_grid[None, :] < deaths)
    return alive.sum(axis=0).astype(np.float64)


def betti_curves(diagrams: Sequence[np.ndarray], eps_grid: np.ndarray) -> Dict[int, np.ndarray]:
    return {dim: betti_curve(dgm, eps_grid) for dim, dgm in enumerate(diagrams)}


def _pair_cost_matrix(A: np.ndarray, B: np.ndarray, metric: str = "linf") -> np.ndarray:
    A = finite_pairs(A)
    B = finite_pairs(B)
    if A.size == 0 and B.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if A.size == 0:
        return np.zeros((0, len(B)), dtype=np.float64)
    if B.size == 0:
        return np.zeros((len(A), 0), dtype=np.float64)
    diff = A[:, None, :] - B[None, :, :]
    if metric == "linf":
        return np.max(np.abs(diff), axis=-1)
    return np.linalg.norm(diff, axis=-1)


def _augmented_assignment_cost(A: np.ndarray, B: np.ndarray, power: float = 1.0, metric: str = "linf") -> np.ndarray:
    A = finite_pairs(A)
    B = finite_pairs(B)
    m, n = len(A), len(B)
    if m == 0 and n == 0:
        return np.zeros((0,), dtype=np.float64)
    pair = _pair_cost_matrix(A, B, metric=metric)
    cost = np.zeros((m + n, n + m), dtype=np.float64)

    if m and n:
        cost[:m, :n] = pair ** power

    if m:
        diag_a = 0.5 * np.maximum(A[:, 1] - A[:, 0], 0.0) ** power
        cost[:m, n:] = diag_a[:, None]
    if n:
        diag_b = 0.5 * np.maximum(B[:, 1] - B[:, 0], 0.0) ** power
        cost[m:, :n] = diag_b[None, :]
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind]


def bottleneck_distance(A: np.ndarray, B: np.ndarray) -> float:
    if persim_bottleneck is not None:
        try:
            return float(persim_bottleneck(finite_pairs(A), finite_pairs(B)))
        except Exception:
            pass
    selected = _augmented_assignment_cost(A, B, power=1.0, metric="linf")
    return float(selected.max()) if selected.size else 0.0


def wasserstein_distance(A: np.ndarray, B: np.ndarray, order: float = 1.0) -> float:
    if persim_wasserstein is not None:
        try:
            return float(persim_wasserstein(finite_pairs(A), finite_pairs(B), matching=False, order=order))
        except Exception:
            pass
    selected = _augmented_assignment_cost(A, B, power=order, metric="l2")
    return float(np.sum(selected) ** (1.0 / order)) if selected.size else 0.0


def layerwise_distances(diagrams_by_layer: Sequence[Sequence[np.ndarray]], dim: int = 1) -> dict:
    bottlenecks = []
    wassers = []
    for a, b in zip(diagrams_by_layer[:-1], diagrams_by_layer[1:]):
        dA = a[dim] if len(a) > dim else np.zeros((0, 2))
        dB = b[dim] if len(b) > dim else np.zeros((0, 2))
        bottlenecks.append(bottleneck_distance(dA, dB))
        wassers.append(wasserstein_distance(dA, dB))
    return {
        "bottleneck": np.asarray(bottlenecks, dtype=np.float64),
        "wasserstein": np.asarray(wassers, dtype=np.float64),
    }
