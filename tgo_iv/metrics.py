
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .jacobian import jacobian_spectrum, summarize_affine_map
from .topology import diagram_statistics, layerwise_distances, persistence_entropy
from .trajectories import (
    cls_series_from_cache,
    population_trajectory_statistics,
    samplewise_step_statistics,
    stack_series,
    step_lengths,
    trajectory_statistics,
    token_cloud_series_from_cache,
)


def summarize_trajectory(series: np.ndarray) -> dict:
    stats = trajectory_statistics(series)
    return {
        "points": stats["points"],
        "dimension": stats["dimension"],
        "total_length": stats["total_length"],
        "mean_velocity": stats["mean_velocity"],
        "mean_curvature": stats["mean_curvature"],
        "max_step_length": stats["max_step_length"],
        "max_curvature": stats["max_curvature"],
        "step_lengths": stats["step_lengths"],
        "curvature": stats["curvature"],
    }


def summarize_population(series_list: Sequence[np.ndarray]) -> dict:
    stats = population_trajectory_statistics(series_list)
    return {
        "mean_series": stats["mean_series"],
        "step_length_mean": stats["sample_length_mean"],
        "step_length_std": stats["sample_length_std"],
        "curvature_mean": stats["sample_curvature_mean"],
        "curvature_std": stats["sample_curvature_std"],
    }


def summarize_jacobian(jacobian_dicts: Sequence[dict]) -> dict:
    sigma_max = np.array([j["sigma_max"] for j in jacobian_dicts], dtype=np.float64)
    sigma_min = np.array([j["sigma_min"] for j in jacobian_dicts], dtype=np.float64)
    cond = np.array([j["condition_number"] for j in jacobian_dicts], dtype=np.float64)
    logdet = np.array([j["log_abs_det"] for j in jacobian_dicts], dtype=np.float64)
    resid = np.array([j["residual_mse"] for j in jacobian_dicts], dtype=np.float64)
    return {
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "condition_number": cond,
        "log_abs_det": logdet,
        "residual_mse": resid,
        "sigma_max_mean": float(np.nanmean(sigma_max)) if sigma_max.size else 0.0,
        "sigma_min_mean": float(np.nanmean(sigma_min)) if sigma_min.size else 0.0,
        "condition_number_mean": float(np.nanmean(cond)) if cond.size else 0.0,
        "log_abs_det_mean": float(np.nanmean(logdet)) if logdet.size else 0.0,
        "residual_mse_mean": float(np.nanmean(resid)) if resid.size else 0.0,
    }


def summarize_topology(diagrams_by_layer: Sequence[Sequence[np.ndarray]], eps_grid: np.ndarray, distances_dim: int = 1) -> dict:
    per_layer = [diagram_statistics(dgms) for dgms in diagrams_by_layer]
    bottleneck = []
    wasserstein = []
    for i in range(len(diagrams_by_layer) - 1):
        from .topology import bottleneck_distance, wasserstein_distance
        dA = diagrams_by_layer[i][distances_dim] if len(diagrams_by_layer[i]) > distances_dim else np.zeros((0, 2))
        dB = diagrams_by_layer[i + 1][distances_dim] if len(diagrams_by_layer[i + 1]) > distances_dim else np.zeros((0, 2))
        bottleneck.append(bottleneck_distance(dA, dB))
        wasserstein.append(wasserstein_distance(dA, dB))
    return {
        "per_layer": per_layer,
        "bottleneck": np.asarray(bottleneck, dtype=np.float64),
        "wasserstein": np.asarray(wasserstein, dtype=np.float64),
    }
