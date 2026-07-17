from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _center(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.size == 0:
        return X
    return X - X.mean(axis=0, keepdims=True)


def project_series_to_2d(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 2 or series.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if series.shape[1] <= 2:
        out = np.zeros((series.shape[0], 2), dtype=np.float64)
        out[:, : series.shape[1]] = series
        return out
    X = _center(series)
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        proj = X @ vt[:2].T
        if proj.shape[1] < 2:
            padded = np.zeros((proj.shape[0], 2), dtype=np.float64)
            padded[:, : proj.shape[1]] = proj
            return padded
        return proj.astype(np.float64, copy=False)
    except Exception:
        return X[:, :2].astype(np.float64, copy=False)


def extract_cls_vector(layer_name: str, tensor: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor)
    if tensor.ndim == 2:
        if tensor.shape[0] == 1:
            return tensor[0]
        return tensor.mean(axis=0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected 2D or 3D tensor for {layer_name}, got {tensor.shape}")
    if tensor.shape[1] == 0:
        return np.zeros((tensor.shape[-1],), dtype=np.float64)
    return tensor[0, 0, :]


def extract_token_cloud(layer_name: str, tensor: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor)
    if tensor.ndim == 3:
        return tensor[0]
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"Expected 2D or 3D tensor for {layer_name}, got {tensor.shape}")


def cls_series_from_cache(cache: Dict[str, np.ndarray], layer_names: Sequence[str]) -> np.ndarray:
    vectors = []
    for name in layer_names:
        if name not in cache:
            continue
        vectors.append(extract_cls_vector(name, cache[name]))
    if not vectors:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(vectors, axis=0).astype(np.float64, copy=False)


def token_cloud_series_from_cache(cache: Dict[str, np.ndarray], layer_names: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in layer_names:
        if name not in cache:
            continue
        out[name] = extract_token_cloud(name, cache[name]).astype(np.float64, copy=False)
    return out


def step_vectors(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 2 or series.shape[0] < 2:
        return np.zeros((0, series.shape[-1] if series.ndim == 2 else 0), dtype=np.float64)
    return np.diff(series, axis=0)


def step_lengths(series: np.ndarray) -> np.ndarray:
    steps = step_vectors(series)
    if steps.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(steps, axis=1)


def curvature(series: np.ndarray) -> np.ndarray:
    steps = step_vectors(series)
    if steps.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(np.diff(steps, axis=0), axis=1)


def trajectory_statistics(series: np.ndarray) -> dict:
    series = np.asarray(series, dtype=np.float64)
    lengths = step_lengths(series)
    curv = curvature(series)
    cumulative_lengths = np.cumsum(lengths) if lengths.size else np.zeros((0,), dtype=np.float64)
    return {
        "points": int(series.shape[0]) if series.ndim == 2 else 0,
        "dimension": int(series.shape[1]) if series.ndim == 2 and series.shape[0] else 0,
        "step_lengths": lengths,
        "cumulative_lengths": cumulative_lengths,
        "curvature": curv,
        "total_length": float(lengths.sum()) if lengths.size else 0.0,
        "mean_velocity": float(lengths.mean()) if lengths.size else 0.0,
        "mean_curvature": float(curv.mean()) if curv.size else 0.0,
        "max_step_length": float(lengths.max()) if lengths.size else 0.0,
        "max_curvature": float(curv.max()) if curv.size else 0.0,
    }


def stack_series(series_list: Sequence[np.ndarray]) -> np.ndarray:
    series_list = [np.asarray(s, dtype=np.float64) for s in series_list if np.asarray(s).ndim == 2]
    if not series_list:
        return np.zeros((0, 0, 0), dtype=np.float64)
    return np.stack(series_list, axis=0)


def samplewise_step_statistics(series_list: Sequence[np.ndarray]) -> dict:
    stack = stack_series(series_list)
    if stack.ndim != 3 or stack.shape[0] == 0:
        return {
            "length_mean": np.zeros((0,), dtype=np.float64),
            "length_std": np.zeros((0,), dtype=np.float64),
            "curvature_mean": np.zeros((0,), dtype=np.float64),
            "curvature_std": np.zeros((0,), dtype=np.float64),
            "length_matrix": np.zeros((0, 0), dtype=np.float64),
            "curvature_matrix": np.zeros((0, 0), dtype=np.float64),
        }
    lengths = np.linalg.norm(np.diff(stack, axis=1), axis=2)
    curv = np.linalg.norm(np.diff(np.diff(stack, axis=1), axis=1), axis=2) if stack.shape[1] >= 3 else np.zeros((stack.shape[0], 0))
    return {
        "length_mean": np.nanmean(lengths, axis=0),
        "length_std": np.nanstd(lengths, axis=0),
        "curvature_mean": np.nanmean(curv, axis=0) if curv.size else np.zeros((0,)),
        "curvature_std": np.nanstd(curv, axis=0) if curv.size else np.zeros((0,)),
        "length_matrix": lengths,
        "curvature_matrix": curv,
    }


def population_mean_series(series_list: Sequence[np.ndarray]) -> np.ndarray:
    stack = stack_series(series_list)
    if stack.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return stack.mean(axis=0)


def population_trajectory_statistics(series_list: Sequence[np.ndarray]) -> dict:
    mean_series = population_mean_series(series_list)
    sample_stats = samplewise_step_statistics(series_list)
    stats = trajectory_statistics(mean_series)
    stats.update({
        "mean_series": mean_series,
        "sample_length_mean": sample_stats["length_mean"],
        "sample_length_std": sample_stats["length_std"],
        "sample_curvature_mean": sample_stats["curvature_mean"],
        "sample_curvature_std": sample_stats["curvature_std"],
    })
    return stats
