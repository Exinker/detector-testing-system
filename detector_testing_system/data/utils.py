from vmk_spectrum2_wrapper.typing import Array
from vmk_spectrum2_wrapper.units import get_units_clipping

from detector_testing_system.data import Data


def to_array(data: Data, n: int, threshold: float | None = None) -> tuple[Array[float], Array[float], Array[float]]:
    """Convert data to array."""
    u = data.mean[:, n]
    du = data.variance[:, n]

    threshold = get_units_clipping(units=data.units) if threshold is None else threshold
    cond = u < threshold

    return u[cond], du[cond], data.exposure[cond]
