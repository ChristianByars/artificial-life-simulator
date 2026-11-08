"""Period measurement for the cell-cycle validation tests (SPEC §6.1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from alife.lifeforms.cell.cycle import MPF_THRESHOLD


def peak_times(t: ArrayLike, y: ArrayLike, min_height: float = MPF_THRESHOLD) -> list[float]:
    """Times of the local maxima of y above min_height, refined by parabolic interpolation.

    The default suits MPF: a peak is a spike that takes the cell into mitosis, so
    sub-threshold numerical ripples in an arrested cell don't count.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mid = y[1:-1]
    is_peak = (mid > y[:-2]) & (mid >= y[2:]) & (mid > min_height)
    return [_vertex(t[i - 1 : i + 2], y[i - 1 : i + 2]) for i in np.flatnonzero(is_peak) + 1]


def _vertex(t: np.ndarray, y: np.ndarray) -> float:
    """Time of the vertex of the parabola through three points (spacing may be uneven).

    At a detected peak y[1] > y[0] and y[1] >= y[2], so a >= 0 > b and the denominator is > 0.
    """
    a = (t[1] - t[0]) * (y[1] - y[2])
    b = (t[1] - t[2]) * (y[1] - y[0])
    return float(t[1] - 0.5 * ((t[1] - t[0]) * a - (t[1] - t[2]) * b) / (a - b))


def mean_period(t: ArrayLike, y: ArrayLike, skip: int = 2) -> float:
    """Mean interval between peaks of y, after dropping the first `skip` peaks (warm-up)."""
    peaks = peak_times(t, y)[skip:]
    if len(peaks) < 2:
        raise ValueError(f"need two peaks after skipping {skip}, found {len(peaks)}")
    return float(np.mean(np.diff(peaks)))
