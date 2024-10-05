from dataclasses import dataclass
import numpy as np

from vmk_spectrum3_wrapper.typing import Array, MilliSecond
from vmk_spectrum3_wrapper.units import Units

from detector_testing_system.data.data import Data
from detector_testing_system.data.exceptions import EmptyArrayError
from detector_testing_system.types import T


@dataclass
class Output:
    average: Array[T]
    variance: Array[T]
    exposure: Array[MilliSecond]
    n: int
    label: str
    units: Units

    @classmethod
    def create(
        cls,
        data: Data,
        n: int,
        threshold: float | None = None,
    ) -> 'Output':
        average = data.average[:, n]
        variance = data.variance[:, n]
        exposure = data.exposure

        threshold = threshold or data.units.value_max
        cond = average < threshold

        if not np.any(cond):
            raise EmptyArrayError(message=f'Data couldn\'t be converted! An array is empty for cell: {n}.')

        return cls(
            average=average[cond],
            variance=variance[cond],
            exposure=exposure[cond],
            n=n,
            label=data.label,
            units=data.units,
        )
