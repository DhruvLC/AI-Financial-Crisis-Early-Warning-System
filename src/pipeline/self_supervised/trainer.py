"""Training engine for the Self-Supervised Learning module.

Contrastive pretraining loop built on the deep-learning trainer's
infrastructure — :func:`build_optimizer`, :func:`build_scheduler`,
:class:`EarlyStopping`, :func:`load_checkpoint`, device selection
(CUDA → MPS → CPU), mixed precision on CUDA, gradient clipping, NaN
detection, best/last checkpointing and resuming — with the two-view
forward the SimCLR objective needs:

    (v1, v2) → encoder → projection head → SSL loss(p1, p2)

Checkpoints store both the encoder and the projection head so pretraining
can resume exactly; the downstream artefacts only ever load the encoder.
"""
from __future__ import annotations

import os
import time

import torch
from torch import nn

from ingestion.logging_config import get_logger
from pipeline.deep_learning.trainer import (EarlyStopping,  # noqa: F401
                                            build_optimizer,
                                            build_scheduler,
                                            load_checkpoint)

from .base import (EpochRecord, SSLError, TrainingHistory, count_parameters,
                   resolve_device)
from .losses import build_ssl_loss

__all__ = ["SSLTrainer", "EarlyStopping", "build_optimizer",
           "build_scheduler", "load_checkpoint", "CHECKPOINT_VERSION"]

log = get_logger("ssl.trainer")

CHECKPOINT_VERSION = 1


class SSLTrainer:
    """Config-driven contrastive pretrainer for one encoder + head."""

    def __init__(self, cfg: dict | None = None,
                 checkpoint_dir: str = "models/self_supervised") -> None:
        cfg = cfg or {}
        train_cfg = cfg.get("training", {})
        self.max_epochs = int(train_cfg.get("epochs", 100))
        self.grad_clip = float(train_cfg.get("gradient_clip", 1.0))
        self.mixed_precision = bool(train_cfg.get("mixed_precision", True))
        self.log_every = int(train_cfg.get("log_every", 1))
        self.device = resolve_device(train_cfg.get("device", "auto"))
        self.loss_cfg = cfg.get("loss", {})
        self.optimizer_cfg = cfg.get("optimizer", {})
        self.scheduler_cfg = cfg.get("scheduler", {})
        es_cfg = cfg.get("early_stopping", {})
        self.es_kwargs = {"patience": int(es_cfg.get("patience", 15)),
                          "min_delta": float(es_cfg.get("min_delta", 1e-4)),
                          "enabled": bool(es_cfg.get("enabled", True))}
        ckpt_cfg = cfg.get("checkpoint", {})
        self.checkpoint_dir = ckpt_cfg.get("dir", checkpoint_dir)
        self.save_checkpoints = bool(ckpt_cfg.get("enabled", True))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ── main loop ─────────────────────────────────────────────────────────────
    def fit(self, name: str, encoder: nn.Module, head: nn.Module,
            train_loader, val_loader=None,
            resume_from: str | None = None) -> tuple[TrainingHistory, dict]:
        """Pretrain ``encoder``+``head`` in place; return
        (history, checkpoint paths). Early stopping tracks the validation
        contrastive loss (train loss when no val loader is given)."""
        encoder = encoder.to(self.device)
        head = head.to(self.device)
        criterion = build_ssl_loss(self.loss_cfg).to(self.device)
        joint = nn.ModuleDict({"encoder": encoder, "head": head})
        optimizer = build_optimizer(joint, self.optimizer_cfg)
        scheduler, plateau = build_scheduler(optimizer, self.scheduler_cfg,
                                             self.max_epochs)
        amp = (self.mixed_precision and self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        stopper = EarlyStopping(**self.es_kwargs)
        history = TrainingHistory()
        start_epoch = 1
        if resume_from:
            start_epoch = 1 + self._load_checkpoint(
                resume_from, joint, optimizer, scheduler)

        log.info("pretraining %s on %s: %d params (encoder %d), "
                 "epochs %d..%d, amp=%s", name, self.device,
                 count_parameters(joint), count_parameters(encoder),
                 start_epoch, self.max_epochs, amp)
        started = time.perf_counter()
        for epoch in range(start_epoch, self.max_epochs + 1):
            tick = time.perf_counter()
            train_loss = self._run_epoch(joint, criterion, train_loader,
                                         optimizer, scaler, amp)
            val_loss = (self._run_epoch(joint, criterion, val_loader)
                        if val_loader is not None else train_loss)
            if not (train_loss == train_loss and val_loss == val_loss):
                raise SSLError(f"{name}: NaN contrastive loss at epoch "
                               f"{epoch} (train={train_loss}, "
                               f"val={val_loss}) — lower the learning "
                               f"rate or soften the augmentations")
            if scheduler is not None:
                scheduler.step(val_loss) if plateau else scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            record = EpochRecord(epoch=epoch, train_loss=train_loss,
                                 val_loss=val_loss, lr=lr,
                                 seconds=time.perf_counter() - tick)
            history.add(record)
            if epoch % self.log_every == 0:
                log.info("%s epoch %3d/%d | train %.5f | val %.5f | "
                         "lr %.2e | %.2fs", name, epoch, self.max_epochs,
                         train_loss, val_loss, lr, record.seconds)

            should_stop = stopper.step(epoch, val_loss, joint)
            if stopper.best_epoch == epoch and self.save_checkpoints:
                self._save_checkpoint(name, "best", joint, optimizer,
                                      scheduler, epoch, history)
            if should_stop:
                history.stopped_early = True
                log.info("%s: early stopping at epoch %d (best epoch %d, "
                         "val_loss=%.6f)", name, epoch, stopper.best_epoch,
                         stopper.best_loss)
                break

        history.best_epoch = stopper.best_epoch
        history.best_val_loss = stopper.best_loss
        history.total_seconds = time.perf_counter() - started
        checkpoints = {}
        if self.save_checkpoints:
            checkpoints["last"] = self._save_checkpoint(
                name, "last", joint, optimizer, scheduler,
                history.epochs[-1].epoch if history.epochs else 0, history)
        stopper.restore(joint)                   # best weights for extraction
        if self.save_checkpoints:
            checkpoints["best"] = os.path.join(
                self.checkpoint_dir, f"{name}_best.pt")
        log.info("%s: pretrained %d epochs in %.2fs (best epoch %d)", name,
                 len(history.epochs), history.total_seconds,
                 history.best_epoch)
        return history, checkpoints

    # ── epoch helper ──────────────────────────────────────────────────────────
    def _run_epoch(self, joint: nn.ModuleDict, criterion, loader,
                   optimizer=None, scaler=None, amp: bool = False) -> float:
        """One pass over a two-view loader; trains when an optimizer is
        given, evaluates under no_grad otherwise."""
        training = optimizer is not None
        joint.train(training)
        total, n = 0.0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for v1, v2, _ in loader:
                v1, v2 = v1.to(self.device), v2.to(self.device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=amp):
                    p1 = joint["head"](joint["encoder"](v1))
                    p2 = joint["head"](joint["encoder"](v2))
                    loss = criterion(p1, p2)
                if training:
                    scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(joint.parameters(),
                                                 self.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                total += float(loss.detach()) * len(v1)
                n += len(v1)
        if n == 0:
            raise SSLError("empty contrastive loader — batch size larger "
                           "than the split with drop_last?")
        return total / n

    # ── checkpoints ───────────────────────────────────────────────────────────
    def _save_checkpoint(self, name: str, tag: str, joint: nn.ModuleDict,
                         optimizer, scheduler, epoch: int,
                         history: TrainingHistory) -> str:
        path = os.path.join(self.checkpoint_dir, f"{name}_{tag}.pt")
        torch.save({
            "version": CHECKPOINT_VERSION,
            "name": name,
            "epoch": epoch,
            "model_state": joint.state_dict(),        # encoder.* + head.*
            "encoder_state": joint["encoder"].state_dict(),
            "head_state": joint["head"].state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (scheduler.state_dict()
                                if scheduler is not None else None),
            "history": history.as_dict(),
        }, path)
        log.info("checkpoint saved: %s (epoch %d)", path, epoch)
        return path

    def _load_checkpoint(self, path: str, joint: nn.ModuleDict, optimizer,
                         scheduler) -> int:
        """Restore training state; return the epoch it was saved at."""
        state = load_checkpoint(path, self.device)   # validates + raises
        joint.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        log.info("resumed from %s (epoch %d)", path, state["epoch"])
        return int(state["epoch"])
