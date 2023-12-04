import numpy as np

from scipy import stats

from libspectrum2_wrapper.alias import Array


def calculate_stats(__values: Array[float], confidence: float = .99) -> tuple[float, float]:
    """Calculate mean and confidence interval."""
    n = len(__values)

    mean = np.mean(__values)
    se = stats.sem(__values)
    ci = se * stats.t.ppf((1 + confidence) / 2., n - 1)

    return mean, ci
