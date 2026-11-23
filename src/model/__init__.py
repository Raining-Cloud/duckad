"""The DuckAD model.

There is a single model architecture. Its only configurable variant is the waypoint-head
implementation (see WAYPOINT_HEADS / model.head in the training config): "gru" (default) or "mlp".
Both training.train and eval.inference import DuckAD directly and pass `head` through from the
(training or checkpoint) config - there is no model registry / model-name selection anymore.
"""

from __future__ import annotations

from .duckad import (
    WAYPOINT_HEADS,
    DuckAD,
    GruWaypointHead,
    MLPWaypointHead,
    loss_duckad,
)

__all__ = [
    "WAYPOINT_HEADS",
    "DuckAD",
    "GruWaypointHead",
    "MLPWaypointHead",
    "loss_duckad",
]
