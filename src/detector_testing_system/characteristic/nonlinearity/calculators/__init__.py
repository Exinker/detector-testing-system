from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.characteristic.nonlinearity.calculators.calculate_nonlinearity_jnorm import (
    calculate_nonlinearity_jnorm,
)
from detector_testing_system.characteristic.nonlinearity.calculators.calculate_nonlinearity_fit import (
    calculate_nonlinearity_fit,
)
from detector_testing_system.output import Output


def calculate_nonlinearity(
    output: Output,
    method: str = 'fit',
    **kwargs,
) -> tuple[Array[float], float]:

    if method == 'fit':
        return calculate_nonlinearity_fit(output=output, **kwargs)
    if method == 'jnorm':
        return calculate_nonlinearity_jnorm(output=output, **kwargs)

    raise ValueError('method must be either "fit" or "jnorm"')


__all__ = [
    'calculate_nonlinearity',
]
