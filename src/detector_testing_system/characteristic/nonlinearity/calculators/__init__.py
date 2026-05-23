from detector_testing_system.characteristic.dark_current.models import (
    BaseDarkCurrentModel,
    DarkCurrentModelABC,
    JNormDarkCurrentModel,
)
from detector_testing_system.characteristic.nonlinearity.results import NonlinearityResultABC
from detector_testing_system.data import Trace

from .calculate_nonlinearity_jnorm import calculate_nonlinearity_jnorm
from .calculate_nonlinearity_base import calculate_nonlinearity_base


def calculate_nonlinearity(
    trace: Trace,
    model: DarkCurrentModelABC | None = None,
    **kwargs,
) -> NonlinearityResultABC:
    model = model or BaseDarkCurrentModel()

    if isinstance(model, BaseDarkCurrentModel):
        return calculate_nonlinearity_base(
            trace=trace,
            model=model,
            **kwargs,
        )

    if isinstance(model, JNormDarkCurrentModel):
        return calculate_nonlinearity_jnorm(
            trace=trace,
            model=model,
            **kwargs,
        )

    raise TypeError('`BaseDarkCurrentModel` and `JNormDarkCurrentModel` are supported only!')
