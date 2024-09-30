from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from detector_testing_system.data.data import Data
from detector_testing_system.data.exceptions import EmptyArrayError
from vmk_spectrum3_wrapper.typing import Array, Electron, MilliSecond, Percent
from vmk_spectrum3_wrapper.units import get_units_clipping


T = TypeVar('T', Electron, Percent)


@dataclass
class Signal:
    value: Array[T]
    variance: Array[T]
    exposure: Array[MilliSecond]

    @classmethod
    def create(
        cls,
        data: Data,
        n: int,
        threshold: float | None = None,
    ) -> 'Signal':
        value = data.mean[:, n]
        variance = data.variance[:, n]
        exposure = data.exposure

        threshold = get_units_clipping(units=data.units) if threshold is None else threshold
        cond = value < threshold

        if not np.any(cond):
            raise EmptyArrayError(message=f'Data couldn\'t be converted! An array is empty for cell: {n}.')

        return Signal(
            value=value[cond],
            variance=variance[cond],
            exposure=exposure[cond],
        )
