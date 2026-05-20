import numpy as np

from scipy import stats

from vmk_spectrum3_wrapper.types import Array


def calculate_stats(__values: Array[float], confidence: float = .99) -> tuple[float, float]:
    """Calculate mean and confidence interval"""
    __values = __values[~np.isnan(__values)]

    n = len(__values)

    mean = np.mean(__values)
    se = stats.sem(__values)
    ci = se * stats.t.ppf((1 + confidence) / 2., n - 1)

    return mean, ci


def calculate_bounds(
    __values: Array[float],
    q: tuple[int, int] = (25, 75),
    k: float = 1.5,
) -> tuple[float, float]:

    q1, q3 = np.nanpercentile(__values, sorted(q))
    iqr = q3 - q1

    lb = q1 - k * iqr
    ub = q3 + k * iqr
    return tuple([lb, ub])


def trunk_outliers(
    __values: Array[float],
    bounds: tuple[float, float],
) -> Array[float]:
    lb, ub = bounds

    mask = (__values >= lb) & (__values <= ub)
    return __values[mask]


def normalize_values(values: Array[float]) -> Array[float]:
    """Normalize values"""

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    return (values - mean) / std
