from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.characteristic.dark_current.models import (
    BaseDarkCurrentModel,
    DarkCurrentModelABC,
    JNormDarkCurrentModel,
)
from detector_testing_system.data.trace import Trace
from .calculate_nonlinearity_jnorm import calculate_nonlinearity_jnorm
from .calculate_nonlinearity_base import calculate_nonlinearity_base


def calculate_nonlinearity(
    trace: Trace,
    model: DarkCurrentModelABC | None = None,
    show: bool = False,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    **kwargs,
) -> tuple[Array[float], float]:
    model = model or BaseDarkCurrentModel()

    if isinstance(model, BaseDarkCurrentModel):
        return calculate_nonlinearity_base(
            trace=trace,
            model=model,
            show=show,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

    if isinstance(model, JNormDarkCurrentModel):
        return calculate_nonlinearity_jnorm(
            trace=trace,
            model=model,
            show=show,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

    raise TypeError('`BaseDarkCurrentModel` and `JNormDarkCurrentModel` are supported only!')
