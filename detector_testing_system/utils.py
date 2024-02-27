import numpy as np

from scipy import stats

from vmk_spectrum2_wrapper.typing import Array


def calculate_stats(__values: Array[float], confidence: float = .99) -> tuple[float, float]:
    """Calculate mean and confidence interval."""
    __values = __values[~np.isnan(__values)]

    #
    n = len(__values)

    mean = np.mean(__values)
    se = stats.sem(__values)
    ci = se * stats.t.ppf((1 + confidence) / 2., n - 1)

    #
    return mean, ci


def treat_outliers(values: Array[float], coeff: float = 5) -> Array[float]:
    """Treat outliers by coeff."""
    values = values[~np.isnan(values)]

    #
    mean = np.mean(values)
    interval = coeff * np.std(values, ddof=1)

    mask = (values >= mean - interval) & (values <= mean + interval)
    return values[mask]


def normalize_values(values: Array[float]) -> Array[float]:
    """Normalize values."""

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    return (values - mean) / std