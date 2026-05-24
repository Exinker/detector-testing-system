from typing import Callable, TypeAlias

import numpy as np
from scipy import stats

from vmk_spectrum3_wrapper.types import Array, MilliSecond, U

from detector_testing_system.data import Trace


INF = np.inf


def calculate_stats(
    __value: Array[float],
    confidence: float = .99
) -> tuple[float, float]:
    """Calculate mean and confidence interval"""
    __value = __value[~np.isnan(__value)]

    n = len(__value)

    mean = np.mean(__value)
    se = stats.sem(__value)
    ci = se * stats.t.ppf((1 + confidence) / 2., n - 1)

    return mean, ci


def calculate_outlier_bounds(
    __value: Array[float],
    q: tuple[int, int] = (25, 75),
    k: float = 1.5,
) -> tuple[float, float]:

    q1, q3 = np.nanpercentile(__value, sorted(q))
    iqr = q3 - q1

    lb = q1 - k * iqr
    ub = q3 + k * iqr
    return tuple([lb, ub])


def trunk_outliers(
    __value: Array[float],
    bounds: tuple[float, float],
) -> Array[float]:
    lb, ub = bounds

    mask = (__value >= lb) & (__value <= ub)
    return __value[mask]


def normalize(
    __value: Array[float],
) -> Array[float]:

    mean = np.mean(__value)
    std = np.std(__value, ddof=1)
    return (__value - mean) / std


def filter_factory(
    threshold: tuple[U , U] | None = None,
    interval: tuple[MilliSecond, MilliSecond] | None = None,
) -> Callable[[Trace], Array[bool]]:
    threshold = threshold or (-INF, +INF)
    interval = interval or (-INF, +INF)

    def create_mask(
        __value: Array[float],
        lb: float,
        ub: float,
    ) -> Array[bool]:
        return (__value >= lb) & (__value <= ub)

    def inner(
        trace: Trace,
    ) -> Array[bool]:
        return create_mask(trace.u, *threshold) & create_mask(trace.tau, *interval)

    return inner
