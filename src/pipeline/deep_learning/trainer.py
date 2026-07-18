"""Training engine for the Deep Learning module.

Production-grade mini-batch trainer: device selection (CUDA → MPS → CPU),
optional mixed precision (CUDA only), gradient clipping, configurable loss
(BCE / BCEWithLogits / weighted BCE / focal), optimizer (Adam / AdamW /
SGD / RMSProp), LR scheduling (plateau / cosine / step / exponential),
early stopping with best-weight restoration, NaN detection, and
best/last checkpoint saving + resuming.
"""
from __future__ import annotations

import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from ingestion.logging_config import get_logger

from .base import (DLError, EpochRecord, TrainingHistory, count_parameters,
                   resolve_device)

log = get_logger("dl.trainer")

CHECKPOINT_VERSION = 1


# ── losses ──────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al., 2017)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class BCEOnProbs(nn.Module):
    """Plain BCE — applies sigmoid to logits first (config key ``bce``)."""

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy(
            torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7), targets)


def build_loss(cfg: dict | None, pos_weight: float = 1.0,
               device: torch.device | None = None) -> nn.Module:
    """Config → loss module. All losses take (logits, targets)."""
    cfg = cfg or {}
    name = str(cfg.get("name", "bce_with_logits")).lower()
    if name == "bce":
        return BCEOnProbs()
    if name == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        w = torch.tensor(float(cfg.get("pos_weight") or pos_weight),
                         device=device)
        return nn.BCEWithLogitsLoss(pos_weight=w)
    if name == "focal":
        return FocalLoss(gamma=float(cfg.get("gamma", 2.0)),
                         alpha=float(cfg.get("alpha", 0.25)))
    raise DLError(f"unsupported loss '{name}' (supported: bce, "
                  f"bce_with_logits, weighted_bce, focal)")


# ── optimizers & schedulers ─────────────────────────────────────────────────
OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
              "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}


def build_optimizer(model: nn.Module, cfg: dict | None) -> torch.optim.Optimizer:
    cfg = dict(cfg or {})
    name = str(cfg.pop("name", "adamw")).lower()
    cls = OPTIMIZERS.get(name)
    if cls is None:
        raise DLError(f"unsupported optimizer '{name}' "
                      f"(supported: {sorted(OPTIMIZERS)})")
    lr = float(cfg.pop("lr", cfg.pop("learning_rate", 1e-3)))
    kwargs = {"lr": lr}
    if "weight_decay" in cfg:
        kwargs["weight_decay"] = float(cfg["weight_decay"])
    if name == "sgd" and "momentum" in cfg:
        kwargs["momentum"] = float(cfg["momentum"])
    return cls(model.parameters(), **kwargs)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict | None,
                    max_epochs: int):
    """Config → LR scheduler (or None). Returns (scheduler, is_plateau)."""
    cfg = cfg or {}
    name = str(cfg.get("name", "none")).lower()
    if name in ("none", ""):
        return None, False
    if name in ("plateau", "reduce_on_plateau", "reducelronplateau"):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 5))), True
    if name in ("cosine", "cosine_annealing"):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.get("t_max", max_epochs))), False
    if name in ("step", "steplr"):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(cfg.get("step_size", 10)),
            gamma=float(cfg.get("gamma", 0.5))), False
    if name in ("exponential", "exponentiallr"):
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(cfg.get("gamma", 0.95))), False
    raise DLError(f"unsupported scheduler '{name}' (supported: plateau, "
                  f"cosine, step, exponential, none)")


# ── early stopping ──────────────────────────────────────────────────────────
class EarlyStopping:
    """Stop when val loss hasn't improved by ``min_delta`` for ``patience``
    epochs; keeps the best state dict for automatic restoration."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 enabled: bool = True) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.enabled = bool(enabled)
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state: dict | None = None
        self.counter = 0

    def step(self, epoch: int, val_loss: float,
             model: nn.Module) -> bool:
        """Record this epoch; return True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = {k: v.detach().clone()
                               for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        self.counter += 1
        return self.enabled and self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            log.info("restored best weights (epoch %d, val_loss=%.6f)",
                     self.best_epoch, self.best_loss)


# ── trainer ─────────────────────────────────────────────────────────────────
class Trainer:
    """Config-driven mini-batch trainer for one network."""

    def __init__(self, cfg: dict | None = None,
                 checkpoint_dir: str = "models/deep_learning",
                 evaluator=None) -> None:
        cfg = cfg or {}
        train_cfg = cfg.get("training", {})
        self.max_epochs = int(train_cfg.get("epochs", 50))
        self.grad_clip = float(train_cfg.get("gradient_clip", 1.0))
        self.mixed_precision = bool(train_cfg.get("mixed_precision", True))
        self.log_every = int(train_cfg.get("log_every", 1))
        self.device = resolve_device(train_cfg.get("device", "auto"))
        self.loss_cfg = cfg.get("loss", {})
        self.optimizer_cfg = cfg.get("optimizer", {})
        self.scheduler_cfg = cfg.get("scheduler", {})
        es_cfg = cfg.get("early_stopping", {})
        self.es_kwargs = {"patience": int(es_cfg.get("patience", 10)),
                          "min_delta": float(es_cfg.get("min_delta", 1e-4)),
                          "enabled": bool(es_cfg.get("enabled", True))}
        ckpt_cfg = cfg.get("checkpoint", {})
        self.checkpoint_dir = ckpt_cfg.get("dir", checkpoint_dir)
        self.save_checkpoints = bool(ckpt_cfg.get("enabled", True))
        self.evaluator = evaluator
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ── main loop ─────────────────────────────────────────────────────────────
    def fit(self, name: str, model: nn.Module, train_loader: DataLoader,
            val_loader: DataLoader, pos_weight: float = 1.0,
            resume_from: str | None = None) -> tuple[TrainingHistory, dict]:
        """Train ``model`` in place; return (history, checkpoint paths)."""
        model = model.to(self.device)
        criterion = build_loss(self.loss_cfg, pos_weight, self.device)
        if isinstance(criterion, nn.Module):
            criterion = criterion.to(self.device)
        optimizer = build_optimizer(model, self.optimizer_cfg)
        scheduler, plateau = build_scheduler(optimizer, self.scheduler_cfg,
                                             self.max_epochs)
        amp = (self.mixed_precision and self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        stopper = EarlyStopping(**self.es_kwargs)
        history = TrainingHistory()
        start_epoch = 1
        if resume_from:
            start_epoch = 1 + self._load_checkpoint(
                resume_from, model, optimizer, scheduler)

        log.info("training %s on %s: %d params, epochs %d..%d, amp=%s",
                 name, self.device, count_parameters(model), start_epoch,
                 self.max_epochs, amp)
        started = time.perf_counter()
        for epoch in range(start_epoch, self.max_epochs + 1):
            tick = time.perf_counter()
            train_loss = self._train_epoch(model, train_loader, criterion,
                                           optimizer, scaler, amp)
            val_loss, val_metrics = self._validate(name, model, val_loader,
                                                   criterion)
            if not (train_loss == train_loss and val_loss == val_loss):
                raise DLError(f"{name}: NaN loss at epoch {epoch} "
                              f"(train={train_loss}, val={val_loss}) — "
                              "lower the learning rate or check the data")
            if scheduler is not None:
                scheduler.step(val_loss) if plateau else scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            record = EpochRecord(epoch=epoch, train_loss=train_loss,
                                 val_loss=val_loss, val_metrics=val_metrics,
                                 lr=lr, seconds=time.perf_counter() - tick)
            history.add(record)
            if epoch % self.log_every == 0:
                log.info("%s epoch %3d/%d | train %.5f | val %.5f | "
                         "lr %.2e | %.2fs", name, epoch, self.max_epochs,
                         train_loss, val_loss, lr, record.seconds)

            should_stop = stopper.step(epoch, val_loss, model)
            if stopper.best_epoch == epoch and self.save_checkpoints:
                self._save_checkpoint(name, "best", model, optimizer,
                                      scheduler, epoch, history)
            if should_stop:
                history.stopped_early = True
                log.info("%s: early stopping at epoch %d "
                         "(best epoch %d, val_loss=%.6f)", name, epoch,
                         stopper.best_epoch, stopper.best_loss)
                break

        history.best_epoch = stopper.best_epoch
        history.best_val_loss = stopper.best_loss
        history.total_seconds = time.perf_counter() - started
        checkpoints = {}
        if self.save_checkpoints:
            checkpoints["last"] = self._save_checkpoint(
                name, "last", model, optimizer, scheduler,
                history.epochs[-1].epoch if history.epochs else 0, history)
        stopper.restore(model)                    # best weights for eval
        if self.save_checkpoints:
            checkpoints["best"] = os.path.join(
                self.checkpoint_dir, f"{name}_best.pt")
        log.info("%s: trained %d epochs in %.2fs (best epoch %d)", name,
                 len(history.epochs), history.total_seconds,
                 history.best_epoch)
        return history, checkpoints

    # ── epoch helpers ─────────────────────────────────────────────────────────
    def _train_epoch(self, model, loader, criterion, optimizer, scaler,
                     amp: bool) -> float:
        model.train()
        total, n = 0.0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                loss = criterion(model(X), y)
            scaler.scale(loss).backward()
            if self.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(y)
            n += len(y)
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self, name: str, model, loader,
                  criterion) -> tuple[float, dict[str, float]]:
        model.eval()
        total, n = 0.0, 0
        logits_all, y_all = [], []
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = model(X)
            total += float(criterion(logits, y)) * len(y)
            n += len(y)
            logits_all.append(logits.cpu())
            y_all.append(y.cpu())
        val_loss = total / max(n, 1)
        metrics: dict[str, float] = {}
        if self.evaluator is not None:
            proba = torch.sigmoid(torch.cat(logits_all)).numpy()
            y_true = torch.cat(y_all).numpy()
            try:
                ev = self.evaluator.evaluate(name, "val_epoch", y_true,
                                             proba)
                metrics = {k: ev.metrics[k]
                           for k in ("roc_auc", "f1", "accuracy")
                           if k in ev.metrics}
            except Exception:  # noqa: BLE001 - metrics are best-effort here
                pass
        return val_loss, metrics

    # ── checkpoints ───────────────────────────────────────────────────────────
    def _save_checkpoint(self, name: str, tag: str, model, optimizer,
                         scheduler, epoch: int,
                         history: TrainingHistory) -> str:
        path = os.path.join(self.checkpoint_dir, f"{name}_{tag}.pt")
        torch.save({
            "version": CHECKPOINT_VERSION,
            "name": name,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (scheduler.state_dict()
                                if scheduler is not None else None),
            "history": history.as_dict(),
        }, path)
        log.info("checkpoint saved: %s (epoch %d)", path, epoch)
        return path

    def _load_checkpoint(self, path: str, model, optimizer,
                         scheduler) -> int:
        """Restore training state; return the epoch it was saved at."""
        state = load_checkpoint(path, self.device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        log.info("resumed from %s (epoch %d)", path, state["epoch"])
        return int(state["epoch"])


def load_checkpoint(path: str, device=None) -> dict:
    """Load + validate a checkpoint file (raises :class:`DLError`)."""
    if not os.path.exists(path):
        raise DLError(f"checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device or "cpu",
                           weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any load failure = corrupt
        raise DLError(f"corrupt checkpoint {path}: {exc}") from exc
    required = ("version", "model_state", "epoch")
    missing = [k for k in required if k not in state]
    if missing:
        raise DLError(f"corrupt checkpoint {path}: missing keys {missing}")
    return state
