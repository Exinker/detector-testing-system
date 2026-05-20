import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.characteristic.gradient import calculate_gradient
from detector_testing_system.experiment import EmptyArrayError
from detector_testing_system.trace import Trace

from .base_model import DarkCurrentModelABC, DarkCurrentResult


class JNormDarkCurrentModel(DarkCurrentModelABC):

    name = 'jnorm'

    def __init__(
        self,
        epsilon: float = .10,
        min_points: int = 10,
    ) -> None:

        self.epsilon = epsilon
        self.min_points = min_points

    def fit(self, trace: Trace) -> DarkCurrentResult:

        u_grad = calculate_gradient(trace=trace)
        mask = self._select_mask(
            u_grad=u_grad,
        )
        if sum(mask) == 0:
            raise EmptyArrayError(
                message=f'Data don\'t enough to be fitted! Linear fit calculation was failed in cell {trace.n}.',
            )

        value = float(np.mean(np.asarray(u_grad)[mask]))
        bias = float(np.mean(trace.u[mask] - value * trace.tau[mask]))

        xi = self._calculate_xi(
            tau=trace.tau,
            u=trace.u,
            p=np.array([value, bias]),
        )

        return DarkCurrentResult(
            value=value,
            bias=bias,
            mask=mask,
            xi=xi,
        )

    def _select_mask(
        self,
        u_grad: Array[float],
    ) -> Array[bool]:
        n_points = len(u_grad)

        mask = np.full(n_points, False)
        for n in range(n_points - self.min_points + 1):

            if self._relative_std(u_grad[n:]) <= self.epsilon:
                mask[n:] = True
                return mask

        return mask

    @staticmethod
    def _relative_std(values: Array[float]) -> float:

        mean = float(np.mean(values))
        if mean == 0 or not np.isfinite(mean):
            return float(np.inf)

        return float(np.std(values) / abs(mean))
