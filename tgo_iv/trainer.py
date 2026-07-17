from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import Config
from .hooks import CLS_LAYER_NAMES
from .trajectories import cls_series_from_cache, population_trajectory_statistics, trajectory_statistics
from .utils import ensure_dir, is_main_process, setup_logging, unwrap_model
from .visualization import plot_line, plot_multi_line


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

        self.logger = setup_logging(self.output_dir)
        self.best_acc = -1.0
        self.scaler = GradScaler(enabled=bool(cfg.train.amp))
        self.optim = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.layer_names = CLS_LAYER_NAMES

        self.learning_length_history: List[np.ndarray] = []
        self.learning_velocity_history: List[np.ndarray] = []
        self.learning_curvature_history: List[np.ndarray] = []

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
            rep_series = self._capture_single_series(self.trajectory_loader)
            pop_stats = self._capture_population_stats()
            self._save_representation_plots(0, rep_series)
            self._save_population_plots(0, pop_stats["mean_series"])
            self._save_learning_plots()
            return

        snapshot_set = set(int(x) for x in self.cfg.analysis.snapshot_epochs)
        for epoch in range(1, int(self.cfg.train.epochs) + 1):
            train_loss = self.train_one_epoch(epoch)
            val_acc = self.validate(epoch)

            rep_series = self._capture_single_series(self.trajectory_loader)
            rep_stats = trajectory_statistics(rep_series)
            self.learning_length_history.append(rep_stats["cumulative_lengths"])
            self.learning_velocity_history.append(rep_stats["step_lengths"])
            self.learning_curvature_history.append(rep_stats["curvature"])

            if epoch in snapshot_set:
                self._save_representation_plots(epoch, rep_series)
                pop_stats = self._capture_population_stats()
                self._save_population_plots(epoch, pop_stats["mean_series"])

            self.scheduler.step()

            is_best = val_acc > self.best_acc
            self.best_acc = max(self.best_acc, val_acc)
            self.save_checkpoint(epoch, train_loss, val_acc, is_best=is_best)

            self.logger.info(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | best={self.best_acc:.4f}"
            )

        self._save_learning_plots()

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
            return cls_series_from_cache(cache, self.layer_names)
        return np.zeros((0, 0), dtype=np.float64)

    @torch.no_grad()
    def _capture_population_stats(self):
        self.model.eval()
        hooks_mgr = self.model.hooks_mgr
        cls_series_list = []

        for images, targets, _ in tqdm(self.analysis_loader, desc="Population capture", disable=not is_main_process()):
            images = images.to(self.device, non_blocking=True)
            _ = self.model(images)
            cache = {k: v.detach().cpu().numpy() for k, v in hooks_mgr.cache.items()}
            hooks_mgr.clear()

            cls_series = cls_series_from_cache(cache, self.layer_names)
            if cls_series.size:
                cls_series_list.append(cls_series)

        if not cls_series_list:
            return {"mean_series": np.zeros((0, 0), dtype=np.float64)}

        return population_trajectory_statistics(cls_series_list)

    def _save_representation_plots(self, epoch: int, series: np.ndarray):
        if not series.size:
            return
        stats = trajectory_statistics(series)
        plot_line(
            stats["cumulative_lengths"],
            self.trajectory_rep_dir / f"epoch_{epoch:03d}_representation_length.png",
            title=f"Representation Trajectory Length (Epoch {epoch:03d})",
            xlabel="Layer transition",
            ylabel="Cumulative length",
        )
        plot_line(
            stats["step_lengths"],
            self.trajectory_rep_dir / f"epoch_{epoch:03d}_representation_velocity.png",
            title=f"Representation Trajectory Velocity (Epoch {epoch:03d})",
            xlabel="Layer transition",
            ylabel="Velocity",
        )
        plot_line(
            stats["curvature"],
            self.trajectory_rep_dir / f"epoch_{epoch:03d}_representation_curvature.png",
            title=f"Representation Trajectory Curvature (Epoch {epoch:03d})",
            xlabel="Layer transition",
            ylabel="Curvature",
        )

    def _save_population_plots(self, epoch: int, mean_series: np.ndarray):
        if not mean_series.size:
            return
        stats = trajectory_statistics(mean_series)
        plot_line(
            stats["cumulative_lengths"],
            self.trajectory_pop_dir / f"epoch_{epoch:03d}_population_length.png",
            title=f"Population Trajectory Length (Epoch {epoch:03d})",
            xlabel="Layer transition",
            ylabel="Cumulative length",
        )
        plot_line(
            stats["step_lengths"],
            self.trajectory_pop_dir / f"epoch_{epoch:03d}_population_velocity.png",
            title=f"Population Trajectory Velocity (Epoch {epoch:03d})",
            xlabel="Layer transition",
            ylabel="Velocity",
        )
        plot_line(
            stats["curvature"],
            self.trajectory_pop_dir / f"epoch_{epoch:03d}_population_curvature.png",
            title=f"Population Trajectory Curvature (Epoch {epoch:03d})",
            xlabel="Layer transition",
            ylabel="Curvature",
        )

    def _save_learning_plots(self):
        if not self.learning_length_history:
            return

        length_stack = np.stack(self.learning_length_history, axis=0)
        velocity_stack = np.stack(self.learning_velocity_history, axis=0)
        curvature_stack = np.stack(self.learning_curvature_history, axis=0)

        def to_series_dict(stack: np.ndarray, prefix: str):
            return {f"{prefix}{i + 1:02d}": stack[:, i].tolist() for i in range(stack.shape[1])}

        plot_multi_line(
            to_series_dict(length_stack, "L"),
            self.trajectory_learn_dir / "learning_trajectory_length.png",
            title="Learning Trajectory Length",
            xlabel="Epoch",
            ylabel="Cumulative length",
        )
        plot_multi_line(
            to_series_dict(velocity_stack, "L"),
            self.trajectory_learn_dir / "learning_trajectory_velocity.png",
            title="Learning Trajectory Velocity",
            xlabel="Epoch",
            ylabel="Velocity",
        )
        if curvature_stack.size:
            plot_multi_line(
                to_series_dict(curvature_stack, "L"),
                self.trajectory_learn_dir / "learning_trajectory_curvature.png",
                title="Learning Trajectory Curvature",
                xlabel="Epoch",
                ylabel="Curvature",
            )
