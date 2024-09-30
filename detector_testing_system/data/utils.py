import numpy as np

from vmk_spectrum2_wrapper.typing import Array, MilliSecond
from vmk_spectrum2_wrapper.units import get_units_clipping

from detector_testing_system.data.data import Data
from detector_testing_system.data.exceptions import EmptyArrayError


def to_array(
    data: Data,
    n: int,
    threshold: float | None = None,
) -> tuple[Array[float], Array[float], Array[MilliSecond]]:
    """Convert data to arrays (mean, variance and exposure)."""

    u = data.mean[:, n]
    variance = data.variance[:, n]
    tau = data.exposure

    threshold = get_units_clipping(units=data.units) if threshold is None else threshold
    cond = u < threshold

    if not np.any(cond):
        raise EmptyArrayError(message=f'Data was converted to empty array (n: {n}).')

    return u[cond], variance[cond], tau[cond]
