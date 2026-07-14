
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def _save_fig(fig, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_line(values: Iterable[float], path: str | Path, title: str = "", xlabel: str = "Layer", ylabel: str = "Value"):
    vals = list(values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(vals) + 1), vals, linewidth=2, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def plot_multi_line(series: Dict[str, Iterable[float]], path: str | Path, title: str = "", xlabel: str = "Layer", ylabel: str = "Value"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, vals in series.items():
        vals = list(vals)
        ax.plot(range(1, len(vals) + 1), vals, linewidth=1.8, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    _save_fig(fig, path)


def plot_heatmap(mat: np.ndarray, path: str | Path, title: str = "", cmap: str = "viridis", vmin=None, vmax=None, xticklabels=None, yticklabels=None, xlabel: str = "", ylabel: str = ""):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=7)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, path)


def plot_bar(labels: List[str], values: Iterable[float], path: str | Path, title: str = "", ylabel: str = ""):
    vals = list(values)
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(labels)), 4.2))
    ax.bar(range(len(labels)), vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, path)


def plot_box(data_by_label: Dict[str, Iterable[float]], path: str | Path, title: str = "", ylabel: str = ""):
    labels = list(data_by_label.keys())
    data = [list(v) for v in data_by_label.values()]
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(labels)), 4.5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    _save_fig(fig, path)


def plot_persistence_diagram(diagrams: Sequence[np.ndarray], path: str | Path, title: str = ""):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    max_val = 0.0
    for dim, dgm in enumerate(diagrams):
        dgm = np.asarray(dgm, dtype=np.float64)
        if dgm.size == 0:
            continue
        finite = dgm[np.isfinite(dgm[:, 1])]
        if finite.size == 0:
            continue
        max_val = max(max_val, float(np.max(finite)))
        ax.scatter(finite[:, 0], finite[:, 1], s=16, alpha=0.8, label=f"H{dim}", color=colors[dim % len(colors)])
    if max_val <= 0:
        max_val = 1.0
    ax.plot([0, max_val], [0, max_val], "--", color="gray", linewidth=1)
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    _save_fig(fig, path)


def plot_barcode(diagrams: Sequence[np.ndarray], path: str | Path, title: str = ""):
    fig, axes = plt.subplots(len(diagrams), 1, figsize=(8.5, max(2.5, 1.6 * len(diagrams))), sharex=True)
    if len(diagrams) == 1:
        axes = [axes]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    max_death = 0.0
    for dim, dgm in enumerate(diagrams):
        ax = axes[dim]
        dgm = np.asarray(dgm, dtype=np.float64)
        finite = dgm[np.isfinite(dgm[:, 1])] if dgm.size else np.zeros((0, 2))
        y = 0
        for birth, death in finite:
            ax.hlines(y, birth, death, colors=colors[dim % len(colors)], linewidth=2)
            y += 1
            max_death = max(max_death, float(death))
        ax.set_ylabel(f"H{dim}", rotation=0, labelpad=20)
        ax.grid(True, axis="x", alpha=0.2)
    axes[-1].set_xlabel("ε")
    axes[0].set_title(title)
    axes[-1].set_xlim(0, max_death * 1.05 if max_death > 0 else 1.0)
    _save_fig(fig, path)


def plot_betti_curves(curves: Dict[int, np.ndarray], eps_grid: np.ndarray, path: str | Path, title: str = ""):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for dim, curve in curves.items():
        ax.plot(eps_grid, curve, linewidth=2, label=f"β{dim}")
    ax.set_xlabel("ε")
    ax.set_ylabel("Betti number")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_fig(fig, path)


def plot_distance_series(values: Iterable[float], path: str | Path, title: str = "", xlabel: str = "Layer", ylabel: str = "Distance"):
    vals = list(values)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(range(1, len(vals) + 1), vals, linewidth=2, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)
