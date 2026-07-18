"""Training engine for the Transformer Models module.

Reuses :class:`pipeline.deep_learning.trainer.Trainer` unchanged — device
selection (CUDA → MPS → CPU), mixed precision on CUDA, gradient clipping,
configurable loss (BCE / BCEWithLogits / weighted BCE / focal), optimizer
(Adam / AdamW / SGD / RMSProp), LR scheduling (plateau / cosine / step /
exponential), early stopping with best-weight restoration, NaN detection,
and best/last checkpoint saving + resuming — pointed at the transformer
checkpoint directory. The loss/optimizer/scheduler factories and
:func:`load_checkpoint` are re-exported for tests and the predictor.
"""
from __future__ import annotations

from ingestion.logging_config import get_logger
from pipeline.deep_learning.trainer import (EarlyStopping,  # noqa: F401
                                            FocalLoss, Trainer, build_loss,
                                            build_optimizer,
                                            build_scheduler,
                                            load_checkpoint)

__all__ = ["TransformerTrainer", "EarlyStopping", "FocalLoss", "Trainer",
           "build_loss", "build_optimizer", "build_scheduler",
           "load_checkpoint"]

log = get_logger("transformers.trainer")


class TransformerTrainer(Trainer):
    """Config-driven mini-batch trainer for one transformer.

    Identical training loop to the deep-learning :class:`Trainer` — kept as
    a subclass for the transformer default checkpoint directory and log
    namespace, and as the seam for future transformer-specific behaviour
    (e.g. LR warmup).
    """

    def __init__(self, cfg: dict | None = None,
                 checkpoint_dir: str = "models/transformers",
                 evaluator=None) -> None:
        super().__init__(cfg, checkpoint_dir=checkpoint_dir,
                         evaluator=evaluator)
