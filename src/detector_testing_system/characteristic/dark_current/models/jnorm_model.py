import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.characteristic.gradient import (
    GradientResult,
    calculate_gradient,
)
from detector_testing_system.data import Trace, TraceFilter
from detector_testing_system.experiment import FitArrayError

from .base_model import DarkCurrentModelABC, DarkCurrentResult


class JNormDarkCurrentModel(DarkCurrentModelABC):

    name = 'jnorm'

    def __init__(
        self,
        epsilon: float = .10,
        min_points: int = 10,
        filter: TraceFilter | None = None,
    ) -> None:
        super().__init__(filter=filter)

        self.epsilon = epsilon
        self.min_points = min_points

    def fit(self, trace: Trace) -> DarkCurrentResult:

        gradient = calculate_gradient(trace=trace)

        mask = self.filter(trace)
        mask = self._build_mask(
            mask=mask,
            gradient=gradient,
        )
        if sum(mask) == 0:
            raise FitArrayError(
                message=f'Data don\'t enough to be fitted! Linear fit calculation was failed in cell {trace.n}.',
            )

        value = float(np.mean(gradient.value[mask]))
        bias = float(np.mean(trace.u[mask] - value * trace.tau[mask]))

        xi = self._calculate_xi(
            tau=trace.tau,
            u=trace.u,
            p=np.array([value, bias]),
        )

        return DarkCurrentResult(
            trace=trace,
            model=self,
            value=value,
            bias=bias,
            mask=mask,
            xi=xi,
        )

    def _build_mask(
        self,
        mask: Array[bool],
        gradient: GradientResult,
    ) -> Array[bool]:
        n_points = len(gradient.value)

        for n in range(n_points - self.min_points + 1):
            values = gradient.value[n:][mask[n:]]
            if len(values) < self.min_points:
                continue

            if self._relative_std(values) <= self.epsilon:
                return (np.arange(n_points) >= n) & mask

        return np.full(n_points, False)

    @staticmethod
    def _relative_std(values: Array[float]) -> float:

        mean = float(np.mean(values))
        if mean == 0 or not np.isfinite(mean):
            return float(np.inf)

        return float(np.std(values) / abs(mean))
