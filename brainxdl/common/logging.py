"""Logging and metric-formatting helpers shared across ST and MM modules."""

import math
from typing import Dict, Optional

# Maps each tracked metric name to whether higher ("max") or lower ("min") is better.
# Used by metric_is_better and EarlyStopping.
METRIC_DIRECTIONS: Dict[str, str] = {
    "loss": "min",
    "log_loss": "min",
    "acc": "max",
    "pre": "max",
    "sen": "max",
    "spe": "max",
    "f1": "max",
    "roc": "max",
}


def metric_is_better(
    metric_name: str,
    current: Optional[float],
    best: Optional[float],
) -> bool:
    """Return True when *current* outperforms *best* for the given metric.

    Handles None and NaN safely so callers do not need to guard against
    uninitialised best values.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    if current is None:
        return False
    try:
        if math.isnan(current):
            return False
    except TypeError:
        pass

    if best is None:
        return True
    try:
        if math.isnan(best):
            return True
    except TypeError:
        pass

    direction = METRIC_DIRECTIONS.get(metric_name, "max")
    return current < best if direction == "min" else current > best


def format_metric_value(metric_name: str, value: Optional[float]) -> str:
    """Format a metric value for console display.

    Loss metrics are shown as 4 decimal floats; all others as percentages.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    if value is None:
        return "nan"
    try:
        if math.isnan(value):
            return "nan"
    except TypeError:
        pass
    if metric_name == "loss":
        return f"{value:.4f}"
    return f"{value * 100:.2f}%"
