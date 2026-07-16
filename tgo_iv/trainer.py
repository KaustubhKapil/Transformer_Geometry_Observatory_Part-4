
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import Config
from .hooks import CLS_LAYER_NAMES, TOKEN_LAYER_NAMES
from .jacobian import batch_jacobian_heterogeneity, layerwise_affine_jacobians
from .metrics import summarize_jacobian, summarize_population, summarize_topology, summarize_trajectory
from .topology import betti_curves, compute_diagrams, diagram_max_value, recommended_eps_grid, diagram_statistics
from .trajectories import (
    cls_series_from_cache,
    population_trajectory_statistics,
    project_series_to_2d,
    token_cloud_series_from_cache,
    trajectory_statistics,
)
from .utils import ensure_dir, is_main_process, save_json, save_npz, setup_logging, unwrap_model
from .visualization import (
    plot_bar,
    plot_barcode,
    plot_betti_curves,
    plot_box,
    plot_distance_series,
    plot_heatmap,
    plot_line,
    plot_multi_line,
    plot_persistence_diagram,
)


class Trainer:
    def __init__(self, cfg: Config, model, train_loader, val_loader, analysis_loader, trajectory_loader, topology_loader, device):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.analysis_loader = analysis_loader
        self.trajectory_loader = trajectory_loader
        self.topology_loader = topology_loader
        self.device = device

        self.output_dir = ensure_dir(cfg.output_dir)
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.summaries_dir = ensure_dir(self.output_dir / "summaries")
        self.logs_dir = ensure_dir(self.output_dir / "logs")

        self.trajectories_dir = ensure_dir(self.output_dir / "trajectories")
        self.trajectory_rep_dir = ensure_dir(self.trajectories_dir / "representation")
        self.trajectory_pop_dir = ensure_dir(self.trajectories_dir / "population")
        self.trajectory_learn_dir = ensure_dir(self.trajectories_dir / "learning")
        self.trajectory_manifold_dir = ensure_dir(self.trajectories_dir / "manifold_learning")

        self.jacobians_dir = ensure_dir(self.output_dir / "jacobians")
        self.jacobian_local_dir = ensure_dir(self.jacobians_dir / "local_deformation_field")
        self.jacobian_evo_dir = ensure_dir(self.jacobians_dir / "jacobian_field_evolution")
        self.jacobian_hetero_dir = ensure_dir(self.jacobians_dir / "heterogeneity")

        self.topology_dir = ensure_dir(self.output_dir / "topology")
        self.topology_pd_dir = ensure_dir(self.topology_dir / "persistence_diagrams")
        self.topology_bar_dir = ensure_dir(self.topology_dir / "barcodes")
        self.topology_betti_dir = ensure_dir(self.topology_dir / "betti_curves")
        self.topology_bottleneck_dir = ensure_dir(self.topology_dir / "bottleneck_distance")
        self.topology_wasserstein_dir = ensure_dir(self.topology_dir / "wasserstein_distance")

        self.global_dir = ensure_dir(self.output_dir / "global_analysis")

        self.logger = setup_logging(self.output_dir)
        self.best_acc = -1.0
        self.scaler = GradScaler(enabled=bool(cfg.train.amp))
        self.optim = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.layer_names = CLS_LAYER_NAMES
        self.token_layer_names = TOKEN_LAYER_NAMES

        self.history_epochs: List[int] = []
        self.history_selected_cls: List[np.ndarray] = []
        self.history_population_mean: List[np.ndarray] = []
        self.history_jacobian_sigma_max: List[np.ndarray] = []
        self.history_jacobian_condition: List[np.ndarray] = []
        self.history_jacobian_logdet: List[np.ndarray] = []
        self.history_jacobian_residual: List[np.ndarray] = []
        self.history_topology_mean_persistence: List[np.ndarray] = []
        self.history_topology_entropy: List[np.ndarray] = []
        self.history_topology_bottleneck: List[np.ndarray] = []
        self.history_topology_wasserstein: List[np.ndarray] = []

    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.train.opt.lower() == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.cfg.train.lr,
                momentum=self.cfg.train.momentum,
                weight_decay=self.cfg.train.weight_decay,
            )
        return torch.optim.AdamW(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)

    def _build_scheduler(self):
        epochs = max(int(self.cfg.train.epochs), 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=epochs, eta_min=self.cfg.train.min_lr)

    def save_checkpoint(self, epoch: int, train_loss: float, val_acc: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_loss": float(train_loss),
            "validation_accuracy": float(val_acc),
            "best_accuracy": float(self.best_acc),
            "config": self.cfg.to_dict(),
        }
        if self.cfg.checkpoint.save_last:
            torch.save(state, self.checkpoint_dir / "last.pth")
        if is_best and self.cfg.checkpoint.save_best:
            torch.save(state, self.checkpoint_dir / "best.pth")

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location="cpu")
        unwrap_model(self.model).load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_acc = float(ckpt.get("best_accuracy", -1.0))
        return ckpt

    def train(self):
        self.model.to(self.device)
        if int(self.cfg.train.epochs) <= 0:
            self.logger.info("Epochs set to 0; running analysis-only mode.")
            self.analyze_snapshot(0)
            self.final_global_analysis()
            return

        snapshot_set = set(int(x) for x in self.cfg.analysis.snapshot_epochs)
        for epoch in range(1, int(self.cfg.train.epochs) + 1):
            train_loss = self.train_one_epoch(epoch)
            val_acc = self.validate(epoch)
            if epoch in snapshot_set:
                self.analyze_snapshot(epoch)
            self.scheduler.step()

            is_best = val_acc > self.best_acc
            self.best_acc = max(self.best_acc, val_acc)
            self.save_checkpoint(epoch, train_loss, val_acc, is_best=is_best)

            self.logger.info(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | best={self.best_acc:.4f}"
            )

        self.final_global_analysis()

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total = 0
        pbar = tqdm(self.train_loader, desc=f"Train {epoch:03d}", disable=not is_main_process())
        for images, targets, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optim.zero_grad(set_to_none=True)
            with autocast(enabled=bool(self.cfg.train.amp)):
                logits = self.model(images)
                loss = F.cross_entropy(logits, targets, label_smoothing=self.cfg.train.label_smoothing)
            self.scaler.scale(loss).backward()
            if self.cfg.train.clip_grad and self.cfg.train.clip_grad > 0:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad)
            self.scaler.step(self.optim)
            self.scaler.update()
            total_loss += float(loss.item()) * images.size(0)
            total += images.size(0)
            pbar.set_postfix(loss=float(loss.item()))
        return total_loss / max(total, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        correct = 0
        total = 0
        pbar = tqdm(self.val_loader, desc=f"Val {epoch:03d}", disable=not is_main_process())
        for images, targets, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.numel()
        return correct / max(total, 1)

    @torch.no_grad()
    def _capture_single_series(self, loader):
        self.model.eval()
        hooks_mgr = self.model.hooks_mgr
        for images, targets, _ in loader:
            images = images.to(self.device, non_blocking=True)
            _ = self.model(images)
            cache = {k: v.detach().cpu().numpy() for k, v in hooks_mgr.cache.items()}
            hooks_mgr.clear()
            return cls_series_from_cache(cache, self.layer_names), token_cloud_series_from_cache(cache, self.token_layer_names)
        return np.zeros((0, 0), dtype=np.float64), {}

    @torch.no_grad()
    def _capture_population_series(self, loader):
        self.model.eval()
        hooks_mgr = self.model.hooks_mgr
        cls_series_list: List[np.ndarray] = []
        token_series_list: List[List[np.ndarray]] = []

        for images, targets, _ in tqdm(loader, desc="Population capture", disable=not is_main_process()):
            images = images.to(self.device, non_blocking=True)
            _ = self.model(images)
            cache = {k: v.detach().cpu().numpy() for k, v in hooks_mgr.cache.items()}
            hooks_mgr.clear()

            cls_series = cls_series_from_cache(cache, self.layer_names)
            if cls_series.size:
                cls_series_list.append(cls_series)

            token_series = token_cloud_series_from_cache(cache, self.token_layer_names)
            if token_series:
                token_series_list.append([token_series[name] for name in self.token_layer_names if name in token_series])

        return cls_series_list, token_series_list

    @torch.no_grad()
    def analyze_snapshot(self, epoch: int):
        selected_series, _ = self._capture_single_series(self.trajectory_loader)
        _, selected_tokens = self._capture_single_series(self.topology_loader)
        population_series_list, population_token_series_list = self._capture_population_series(self.analysis_loader)

        if selected_series.size:
            self.history_selected_cls.append(selected_series)
        if population_series_list:
            pop_mean_series = np.stack(population_series_list, axis=0).mean(axis=0)
            self.history_population_mean.append(pop_mean_series)
        else:
            pop_mean_series = np.zeros((0, 0), dtype=np.float64)

        traj_stats = summarize_trajectory(selected_series)
        pop_stats = summarize_population(population_series_list) if population_series_list else {}
        learning_stats = self._history_learning_stats()
        manifold_stats = self._history_manifold_stats()

        jacobian_dicts = []
        if selected_tokens:
            layer_series = [selected_tokens[name] for name in self.token_layer_names if name in selected_tokens]
            jacobian_dicts = layerwise_affine_jacobians(layer_series, ridge=self.cfg.analysis.jacobian_ridge)
        jacobian_stats = summarize_jacobian(jacobian_dicts)
        if jacobian_dicts:
            self.history_jacobian_sigma_max.append(np.asarray([j["sigma_max"] for j in jacobian_dicts], dtype=np.float64))
            self.history_jacobian_condition.append(np.asarray([j["condition_number"] for j in jacobian_dicts], dtype=np.float64))
            self.history_jacobian_logdet.append(np.asarray([j["log_abs_det"] for j in jacobian_dicts], dtype=np.float64))
            self.history_jacobian_residual.append(np.asarray([j["residual_mse"] for j in jacobian_dicts], dtype=np.float64))

        hetero_stats = batch_jacobian_heterogeneity(
            population_token_series_list[: self.cfg.analysis.jacobian_heterogeneity_size],
            ridge=self.cfg.analysis.jacobian_ridge,
        )

        diagrams_by_layer = []
        for name in self.token_layer_names:
            if name not in selected_tokens:
                continue
            diagrams_by_layer.append(
                compute_diagrams(selected_tokens[name], maxdim=int(self.cfg.analysis.topology_maxdim))
            )
        topology_scale = diagram_max_value(diagrams_by_layer, default=1.0)
        eps_grid = recommended_eps_grid(diagrams_by_layer, num=int(self.cfg.analysis.topology_eps_grid), padding=0.05)
        topology_stats = summarize_topology(diagrams_by_layer, eps_grid)
        if diagrams_by_layer:
            per_layer_stats = [diagram_statistics(dgms) for dgms in diagrams_by_layer]
            mean_persistence = np.array(
                [[stats[f"dim_{d}_mean_persistence"] for d in range(int(self.cfg.analysis.topology_maxdim) + 1)] for stats in per_layer_stats],
                dtype=np.float64,
            )
            entropy = np.array(
                [[stats[f"dim_{d}_entropy"] for d in range(int(self.cfg.analysis.topology_maxdim) + 1)] for stats in per_layer_stats],
                dtype=np.float64,
            )
            self.history_topology_mean_persistence.append(mean_persistence)
            self.history_topology_entropy.append(entropy)
            self.history_topology_bottleneck.append(topology_stats["bottleneck"])
            self.history_topology_wasserstein.append(topology_stats["wasserstein"])

        summary = {
            "epoch": epoch,
            "trajectory": {
                **traj_stats,
                "step_lengths": traj_stats["step_lengths"].tolist(),
                "curvature": traj_stats["curvature"].tolist(),
            },
            "population": {
                **{k: float(v) for k, v in pop_stats.items() if np.isscalar(v)},
                "mean_series_shape": list(pop_mean_series.shape),
            },
            "learning": learning_stats,
            "manifold_learning": manifold_stats,
            "jacobian": {
                **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in jacobian_stats.items() if k not in {"matrix", "bias", "singular_values"}},
            },
            "jacobian_heterogeneity": hetero_stats,
            "topology": {
                "per_layer": topology_stats["per_layer"],
                "bottleneck": topology_stats["bottleneck"].tolist(),
                "wasserstein": topology_stats["wasserstein"].tolist(),
            },
        }
        save_json(summary, self.summaries_dir / f"epoch_{epoch:03d}.json")
        if self.cfg.analysis.save_raw_diagrams and diagrams_by_layer:
            save_npz(
                self.topology_dir / f"epoch_{epoch:03d}_diagrams.npz",
                **{
                    f"layer_{i:02d}_H{dim}": dgm
                    for i, dgms in enumerate(diagrams_by_layer)
                    for dim, dgm in enumerate(dgms)
                },
            )
        if self.cfg.analysis.save_raw_jacobians and jacobian_dicts:
            save_npz(
                self.jacobians_dir / f"epoch_{epoch:03d}_selected.npz",
                **{f"layer_{i:02d}": j["matrix"] for i, j in enumerate(jacobian_dicts)},
            )

        self._save_trajectory_plots(epoch, selected_series, pop_mean_series, learning_stats, manifold_stats)
        self._save_jacobian_plots(epoch, jacobian_dicts, hetero_stats)
        self._save_topology_plots(epoch, diagrams_by_layer, topology_stats, topology_scale, eps_grid)

        return summary

    def _history_learning_stats(self) -> dict:
        if len(self.history_selected_cls) < 2:
            return {}
        stack = np.stack(self.history_selected_cls, axis=0)  # [S, L, D]
        step_heatmap = np.linalg.norm(np.diff(stack, axis=0), axis=2)  # [S-1, L]
        curvature_heatmap = np.linalg.norm(np.diff(np.diff(stack, axis=0), axis=0), axis=2) if stack.shape[0] >= 3 else np.zeros((0, stack.shape[1]))
        return {
            "step_length_heatmap": step_heatmap,
            "curvature_heatmap": curvature_heatmap,
            "step_length_mean_per_layer": np.nanmean(step_heatmap, axis=0),
            "step_length_mean_per_epoch": np.nanmean(step_heatmap, axis=1),
            "curvature_mean_per_layer": np.nanmean(curvature_heatmap, axis=0) if curvature_heatmap.size else np.zeros((stack.shape[1],)),
        }

    def _history_manifold_stats(self) -> dict:
        if len(self.history_population_mean) < 2:
            return {}
        stack = np.stack(self.history_population_mean, axis=0)  # [S, L, D]
        step_heatmap = np.linalg.norm(np.diff(stack, axis=0), axis=2)  # [S-1, L]
        curvature_heatmap = np.linalg.norm(np.diff(np.diff(stack, axis=0), axis=0), axis=2) if stack.shape[0] >= 3 else np.zeros((0, stack.shape[1]))
        return {
            "step_length_heatmap": step_heatmap,
            "curvature_heatmap": curvature_heatmap,
            "step_length_mean_per_layer": np.nanmean(step_heatmap, axis=0),
            "step_length_mean_per_epoch": np.nanmean(step_heatmap, axis=1),
            "curvature_mean_per_layer": np.nanmean(curvature_heatmap, axis=0) if curvature_heatmap.size else np.zeros((stack.shape[1],)),
        }

    def _save_trajectory_plots(self, epoch: int, selected_series: np.ndarray, population_series: np.ndarray, learning_stats: dict, manifold_stats: dict):
        if selected_series.size:
            stats = trajectory_statistics(selected_series)
            plot_line(
                stats["step_lengths"],
                self.trajectory_rep_dir / f"epoch_{epoch:03d}_representation_step_length.png",
                title=f"Representation Trajectory Step Length (Epoch {epoch:03d})",
                xlabel="Layer transition",
                ylabel="||h_{l+1}-h_l||",
            )
            if stats["curvature"].size:
                plot_line(
                    stats["curvature"],
                    self.trajectory_rep_dir / f"epoch_{epoch:03d}_representation_curvature.png",
                    title=f"Representation Trajectory Curvature (Epoch {epoch:03d})",
                    xlabel="Layer transition",
                    ylabel="Curvature",
                )
            plot_trajectory_path(
                project_series_to_2d(selected_series),
                self.trajectory_rep_dir / f"epoch_{epoch:03d}_representation_path.png",
                title=f"Representation Trajectory Path (Epoch {epoch:03d})",
                xlabel="PC1",
                ylabel="PC2",
            )
        if population_series.size:
            pop_stats = trajectory_statistics(population_series)
            plot_line(
                pop_stats["step_lengths"],
                self.trajectory_pop_dir / f"epoch_{epoch:03d}_population_step_length.png",
                title=f"Population Trajectory Step Length (Epoch {epoch:03d})",
                xlabel="Layer transition",
                ylabel="Mean ||Δh||",
            )
            plot_trajectory_path(
                project_series_to_2d(population_series),
                self.trajectory_pop_dir / f"epoch_{epoch:03d}_population_path.png",
                title=f"Population Trajectory Path (Epoch {epoch:03d})",
                xlabel="PC1",
                ylabel="PC2",
            )
        if learning_stats:
            plot_heatmap(
                learning_stats["step_length_heatmap"],
                self.trajectory_learn_dir / f"epoch_{epoch:03d}_learning_step_heatmap.png",
                title=f"Learning Trajectory Step Length Heatmap (Epoch {epoch:03d})",
                xlabel="Layer transition",
                ylabel="Epoch snapshot",
            )
        if manifold_stats:
            plot_heatmap(
                manifold_stats["step_length_heatmap"],
                self.trajectory_manifold_dir / f"epoch_{epoch:03d}_manifold_step_heatmap.png",
                title=f"Manifold Learning Step Length Heatmap (Epoch {epoch:03d})",
                xlabel="Layer transition",
                ylabel="Epoch snapshot",
            )

    def _save_jacobian_plots(self, epoch: int, jacobian_dicts: list[dict], hetero_stats: dict):
        if not jacobian_dicts:
            return
        sigma_max = [j["sigma_max"] for j in jacobian_dicts]
        sigma_min = [j["sigma_min"] for j in jacobian_dicts]
        cond = [j["condition_number"] for j in jacobian_dicts]
        logdet = [j["log_abs_det"] for j in jacobian_dicts]
        resid = [j["residual_mse"] for j in jacobian_dicts]
        plot_line(sigma_max, self.jacobian_local_dir / f"epoch_{epoch:03d}_sigma_max.png", title=f"Local Deformation Field: σ_max (Epoch {epoch:03d})", xlabel="Layer transition", ylabel="σ_max")
        plot_line(sigma_min, self.jacobian_local_dir / f"epoch_{epoch:03d}_sigma_min.png", title=f"Local Deformation Field: σ_min (Epoch {epoch:03d})", xlabel="Layer transition", ylabel="σ_min")
        plot_line(cond, self.jacobian_local_dir / f"epoch_{epoch:03d}_condition.png", title=f"Local Deformation Field: Condition Number (Epoch {epoch:03d})", xlabel="Layer transition", ylabel="κ(J)")
        plot_line(logdet, self.jacobian_local_dir / f"epoch_{epoch:03d}_logdet.png", title=f"Local Deformation Field: log|det(J)| (Epoch {epoch:03d})", xlabel="Layer transition", ylabel="log|det(J)|")
        plot_line(resid, self.jacobian_local_dir / f"epoch_{epoch:03d}_residual.png", title=f"Local Deformation Field: Residual MSE (Epoch {epoch:03d})", xlabel="Layer transition", ylabel="Residual MSE")
        if hetero_stats:
            if hetero_stats.get("sigma_max_mean"):
                plot_line(
                    hetero_stats["sigma_max_mean"],
                    self.jacobian_hetero_dir / f"epoch_{epoch:03d}_sigma_max_mean.png",
                    title=f"Jacobian Heterogeneity: Mean σ_max (Epoch {epoch:03d})",
                    xlabel="Layer transition",
                    ylabel="Mean σ_max",
                )
            if hetero_stats.get("sigma_max_std"):
                plot_line(
                    hetero_stats["sigma_max_std"],
                    self.jacobian_hetero_dir / f"epoch_{epoch:03d}_sigma_max_std.png",
                    title=f"Jacobian Heterogeneity: Std σ_max (Epoch {epoch:03d})",
                    xlabel="Layer transition",
                    ylabel="Std σ_max",
                )
            if hetero_stats.get("condition_mean"):
                plot_line(
                    hetero_stats["condition_mean"],
                    self.jacobian_hetero_dir / f"epoch_{epoch:03d}_condition_mean.png",
                    title=f"Jacobian Heterogeneity: Mean Condition Number (Epoch {epoch:03d})",
                    xlabel="Layer transition",
                    ylabel="Mean κ(J)",
                )

    def _save_topology_plots(self, epoch: int, diagrams_by_layer: Sequence[Sequence[np.ndarray]], topology_stats: dict, topology_scale: float, eps_grid: np.ndarray):
        if not diagrams_by_layer:
            return
        # One figure per selected layer index
        selected_indices = [i for i in self.cfg.analysis.plot_layers if i < len(diagrams_by_layer)]
        for idx in selected_indices:
            dgm = diagrams_by_layer[idx]
            plot_persistence_diagram(
                dgm,
                self.topology_pd_dir / f"epoch_{epoch:03d}_layer_{idx:02d}_persistence_diagram.png",
                title=f"Persistence Diagram Evolution (Epoch {epoch:03d}, Layer {idx:02d})",
                max_value=topology_scale,
            )
            plot_barcode(
                dgm,
                self.topology_bar_dir / f"epoch_{epoch:03d}_layer_{idx:02d}_barcode.png",
                title=f"Barcode Evolution (Epoch {epoch:03d}, Layer {idx:02d})",
                max_value=topology_scale,
            )
            curves = {dim: curve for dim, curve in betti_curves(dgm, eps_grid).items()}
            plot_betti_curves(
                curves,
                eps_grid,
                self.topology_betti_dir / f"epoch_{epoch:03d}_layer_{idx:02d}_betti.png",
                title=f"Betti Curve Evolution (Epoch {epoch:03d}, Layer {idx:02d})",
            )
        if topology_stats.get("bottleneck") is not None and len(topology_stats["bottleneck"]):
            plot_distance_series(
                topology_stats["bottleneck"],
                self.topology_bottleneck_dir / f"epoch_{epoch:03d}_bottleneck.png",
                title=f"Bottleneck Distance Evolution (Epoch {epoch:03d})",
                xlabel="Layer transition",
                ylabel="Distance",
            )
        if topology_stats.get("wasserstein") is not None and len(topology_stats["wasserstein"]):
            plot_distance_series(
                topology_stats["wasserstein"],
                self.topology_wasserstein_dir / f"epoch_{epoch:03d}_wasserstein.png",
                title=f"Wasserstein Distance Evolution (Epoch {epoch:03d})",
                xlabel="Layer transition",
                ylabel="Distance",
            )

    def final_global_analysis(self):
        if not self.history_selected_cls:
            return

        if len(self.history_selected_cls) >= 2:
            stack = np.stack(self.history_selected_cls, axis=0)
            step_heatmap = np.linalg.norm(np.diff(stack, axis=0), axis=2)
            plot_heatmap(
                step_heatmap,
                self.global_dir / "learning_trajectory_heatmap.png",
                title="Learning Trajectory Heatmap",
                xlabel="Layer transition",
                ylabel="Snapshot",
            )
            if stack.shape[0] >= 3:
                curv_heatmap = np.linalg.norm(np.diff(np.diff(stack, axis=0), axis=0), axis=2)
                plot_heatmap(
                    curv_heatmap,
                    self.global_dir / "learning_curvature_heatmap.png",
                    title="Learning Curvature Heatmap",
                    xlabel="Layer transition",
                    ylabel="Snapshot",
                )
            selected_layers = [i for i in self.cfg.analysis.plot_layers if i < stack.shape[1]]
            for layer_idx in selected_layers:
                series = stack[:, layer_idx, :]
                plot_trajectory_path(
                    project_series_to_2d(series),
                    self.global_dir / f"learning_trajectory_layer_{layer_idx:02d}.png",
                    title=f"Learning Trajectory Path (Layer {layer_idx:02d})",
                    xlabel="PC1",
                    ylabel="PC2",
                )

        if len(self.history_population_mean) >= 2:
            stack = np.stack(self.history_population_mean, axis=0)
            step_heatmap = np.linalg.norm(np.diff(stack, axis=0), axis=2)
            plot_heatmap(
                step_heatmap,
                self.global_dir / "manifold_learning_heatmap.png",
                title="Manifold Learning Heatmap",
                xlabel="Layer transition",
                ylabel="Snapshot",
            )
            if stack.shape[0] >= 3:
                curv_heatmap = np.linalg.norm(np.diff(np.diff(stack, axis=0), axis=0), axis=2)
                plot_heatmap(
                    curv_heatmap,
                    self.global_dir / "manifold_curvature_heatmap.png",
                    title="Manifold Curvature Heatmap",
                    xlabel="Layer transition",
                    ylabel="Snapshot",
                )
            selected_layers = [i for i in self.cfg.analysis.plot_layers if i < stack.shape[1]]
            for layer_idx in selected_layers:
                series = stack[:, layer_idx, :]
                plot_trajectory_path(
                    project_series_to_2d(series),
                    self.global_dir / f"manifold_trajectory_layer_{layer_idx:02d}.png",
                    title=f"Manifold Learning Trajectory Path (Layer {layer_idx:02d})",
                    xlabel="PC1",
                    ylabel="PC2",
                )

        if self.history_jacobian_sigma_max:
            sigma_max_stack = np.stack(self.history_jacobian_sigma_max, axis=0)
            cond_stack = np.stack(self.history_jacobian_condition, axis=0)
            logdet_stack = np.stack(self.history_jacobian_logdet, axis=0)
            plot_heatmap(
                sigma_max_stack,
                self.global_dir / "jacobian_sigma_max_heatmap.png",
                title="Jacobian σ_max Heatmap",
                xlabel="Layer transition",
                ylabel="Snapshot",
            )
            plot_heatmap(
                cond_stack,
                self.global_dir / "jacobian_condition_heatmap.png",
                title="Jacobian Condition Number Heatmap",
                xlabel="Layer transition",
                ylabel="Snapshot",
            )
            plot_heatmap(
                logdet_stack,
                self.global_dir / "jacobian_logdet_heatmap.png",
                title="Jacobian log|det(J)| Heatmap",
                xlabel="Layer transition",
                ylabel="Snapshot",
            )

        if self.history_topology_mean_persistence:
            mean_persistence_stack = np.stack(self.history_topology_mean_persistence, axis=0)  # [S, L, D]
            entropy_stack = np.stack(self.history_topology_entropy, axis=0)
            # use H1 if available, otherwise H0
            dim_idx = 1 if mean_persistence_stack.shape[-1] > 1 else 0
            plot_heatmap(
                mean_persistence_stack[:, :, dim_idx],
                self.global_dir / "topology_mean_persistence_heatmap.png",
                title=f"Topology Mean Persistence Heatmap (H{dim_idx})",
                xlabel="Layer",
                ylabel="Snapshot",
            )
            plot_heatmap(
                entropy_stack[:, :, dim_idx],
                self.global_dir / "topology_entropy_heatmap.png",
                title=f"Topology Entropy Heatmap (H{dim_idx})",
                xlabel="Layer",
                ylabel="Snapshot",
            )

        if self.history_topology_bottleneck:
            bottleneck_stack = np.stack(self.history_topology_bottleneck, axis=0)
            wasserstein_stack = np.stack(self.history_topology_wasserstein, axis=0)
            plot_heatmap(
                bottleneck_stack,
                self.global_dir / "topology_bottleneck_heatmap.png",
                title="Bottleneck Distance Heatmap",
                xlabel="Layer transition",
                ylabel="Snapshot",
            )
            plot_heatmap(
                wasserstein_stack,
                self.global_dir / "topology_wasserstein_heatmap.png",
                title="Wasserstein Distance Heatmap",
                xlabel="Layer transition",
                ylabel="Snapshot",
            )
