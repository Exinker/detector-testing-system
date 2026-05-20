from dataclasses import dataclass

import numpy as np

from vmk_spectrum3_wrapper.types import Array, MilliSecond
from vmk_spectrum3_wrapper.units import U, Units

from detector_testing_system.experiment import Data, EmptyArrayError


@dataclass
class Trace:

    u: Array[U]
    variance: Array[U]
    tau: Array[MilliSecond]
    n: int
    label: str
    units: Units

    @classmethod
    def create(
        cls,
        data: Data,
        n: int,
        threshold: float | None = None,
    ) -> 'Trace':

        u = data.u[:, n]
        variance = data.variance[:, n]
        tau = data.tau

        threshold = threshold or data.units.value_max
        cond = u < threshold

        if not np.any(cond):
            raise EmptyArrayError(
                message=f'Data couldn\'t be converted!  Calculation was failed in cell {n}.',
            )

        return cls(
            u=u[cond],
            variance=variance[cond],
            tau=tau[cond],
            n=n,
            label=data.label,
            units=data.units,
        )
