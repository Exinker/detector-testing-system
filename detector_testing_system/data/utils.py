from vmk_spectrum3_wrapper.typing import Array

from detector_testing_system.data import Data


def to_array(data: Data, n: int, threshold: float | None = None) -> tuple[Array[float], Array[float], Array[float]]:
    """Convert data to array."""
    u = data.mean[:, n]
    du = data.variance[:, n]

    threshold = data.units.value_max if threshold is None else threshold
    cond = u < threshold

    return u[cond], du[cond], data.exposure[cond]
