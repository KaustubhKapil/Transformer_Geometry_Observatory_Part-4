
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def fit_affine_jacobian(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {X.shape} and {Y.shape}")
    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: {X.shape} vs {Y.shape}")
    n, d = X.shape
    if n < 2 or d < 2:
        A = np.eye(d, dtype=np.float64)
        b = np.zeros((d,), dtype=np.float64)
        return A, b, 0.0
    X_aug = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)
    XtX = X_aug.T @ X_aug + ridge * np.eye(d + 1, dtype=np.float64)
    XtY = X_aug.T @ Y
    try:
        W = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
    A = W[:-1, :].T
    b = W[-1, :]
    pred = X @ A.T + b
    residual = float(np.mean((pred - Y) ** 2))
    return A.astype(np.float64, copy=False), b.astype(np.float64, copy=False), residual


def jacobian_spectrum(J: np.ndarray) -> dict:
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        raise ValueError(f"Expected a matrix, got {J.shape}")
    try:
        s = np.linalg.svd(J, compute_uv=False)
    except np.linalg.LinAlgError:
        s = np.zeros((min(J.shape),), dtype=np.float64)
    if s.size == 0:
        s = np.zeros((1,), dtype=np.float64)
    s = np.clip(s, 1e-12, None)
    sigma_max = float(s.max())
    sigma_min = float(s.min())
    cond = float(sigma_max / sigma_min) if sigma_min > 0 else float("inf")
    logdet = float(np.sum(np.log(s))) if np.all(np.isfinite(s)) else float("-inf")
    stable_rank = float((np.linalg.norm(J, ord="fro") ** 2) / (sigma_max**2 + 1e-12))
    return {
        "singular_values": s,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "condition_number": cond,
        "log_abs_det": logdet,
        "stable_rank": stable_rank,
        "fro_norm": float(np.linalg.norm(J, ord="fro")),
    }


def summarize_affine_map(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-6) -> dict:
    A, b, residual = fit_affine_jacobian(X, Y, ridge=ridge)
    spec = jacobian_spectrum(A)
    spec.update({
        "matrix": A,
        "bias": b,
        "residual_mse": residual,
    })
    return spec


def layerwise_affine_jacobians(layer_series: Sequence[np.ndarray], ridge: float = 1e-6) -> List[dict]:
    out: List[dict] = []
    for x, y in zip(layer_series[:-1], layer_series[1:]):
        out.append(summarize_affine_map(x, y, ridge=ridge))
    return out


def batch_jacobian_heterogeneity(batch_layer_series: Sequence[Sequence[np.ndarray]], ridge: float = 1e-6) -> dict:
    per_sample = []
    for sample_series in batch_layer_series:
        if len(sample_series) >= 2:
            per_sample.append(layerwise_affine_jacobians(sample_series, ridge=ridge))
    if not per_sample:
        return {}
    n_layers = len(per_sample[0])
    out = {
        "sigma_max_mean": [],
        "sigma_max_std": [],
        "sigma_min_mean": [],
        "sigma_min_std": [],
        "condition_mean": [],
        "condition_std": [],
        "logdet_mean": [],
        "logdet_std": [],
        "residual_mean": [],
        "residual_std": [],
    }
    for li in range(n_layers):
        vals = [sample[li] for sample in per_sample if len(sample) > li]
        sigma_max = np.array([v["sigma_max"] for v in vals], dtype=np.float64)
        sigma_min = np.array([v["sigma_min"] for v in vals], dtype=np.float64)
        cond = np.array([v["condition_number"] for v in vals], dtype=np.float64)
        logdet = np.array([v["log_abs_det"] for v in vals], dtype=np.float64)
        resid = np.array([v["residual_mse"] for v in vals], dtype=np.float64)
        out["sigma_max_mean"].append(float(np.nanmean(sigma_max)))
        out["sigma_max_std"].append(float(np.nanstd(sigma_max)))
        out["sigma_min_mean"].append(float(np.nanmean(sigma_min)))
        out["sigma_min_std"].append(float(np.nanstd(sigma_min)))
        out["condition_mean"].append(float(np.nanmean(cond)))
        out["condition_std"].append(float(np.nanstd(cond)))
        out["logdet_mean"].append(float(np.nanmean(logdet)))
        out["logdet_std"].append(float(np.nanstd(logdet)))
        out["residual_mean"].append(float(np.nanmean(resid)))
        out["residual_std"].append(float(np.nanstd(resid)))
    return out
