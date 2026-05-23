from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from vmk_spectrum3_wrapper.types import Array, MilliSecond, U

from detector_testing_system.data import Trace
from detector_testing_system.experiment import EmptyArrayError


@dataclass
class DarkCurrentResult:

    value: float
    bias: float
    mask: Array[bool]
    xi: Array[U]

    @property
    def p(self) -> Array[float]:
        return [self.value, self.bias]

    def interpolate(self, __exposure: Array[MilliSecond]) -> Array[U]:
        return np.polyval(self.p, __exposure)


class DarkCurrentModelABC(ABC):

    def __init_subclass__(cls, *args, **kwargs):

        if 'name' not in cls.__dict__:
            raise TypeError(f'{cls.__name__} must have have "name" attribute')

        return super().__init_subclass__(*args, **kwargs)

    @abstractmethod
    def fit(self, trace: Trace) -> DarkCurrentResult:
        pass

    @staticmethod
    def _calculate_xi(
        tau: Array[float],
        u: Array[float],
        p: Array[float],
    ) -> Array[float]:
        """Calculate a residual of approximation"""
        u_hat = np.polyval(p, tau)

        # xi = 100*(u_hat - u) / u
        xi = 100*(u_hat - u) / np.polyval([p[0], 0], tau)

        return xi


class BaseDarkCurrentModel(DarkCurrentModelABC):

    name = 'base'
    degree = 1

    def __init__(
        self,
        weighted: bool = True,
        threshold: tuple[U, U] | None = None,
        span: tuple[MilliSecond, MilliSecond] = None,
    ) -> None:

        self.weighted = weighted
        self.threshold = threshold
        self.span = span

    def fit(
        self,
        trace: Trace,
    ) -> DarkCurrentResult:
        threshold = self.threshold or (0, trace.units.value_max)
        span = self.span or (min(trace.tau), max(trace.tau))

        mask = (trace.u >= threshold[0]) & (trace.u <= threshold[1]) & (trace.tau >= span[0]) & (trace.tau <= span[1])
        if sum(mask) < self.degree + 1:
            raise EmptyArrayError(
                message=f'Data don\'t enough to be fitted! Linear fit calculation was failed in cell {trace.n}.',
            )

        if self.weighted:
            p = self._optimize_weighted(
                tau=trace.tau[mask],
                u=trace.u[mask],
            )
        else:
            p = np.polyfit(trace.tau[mask], trace.u[mask], deg=self.degree)

        xi = self._calculate_xi(
            tau=trace.tau,
            u=trace.u,
            p=p,
        )

        return DarkCurrentResult(
            value=float(p[0]),
            bias=float(p[1]),
            mask=mask,
            xi=xi,
        )

    @staticmethod
    def _optimize_weighted(
        tau: Array[float],
        u: Array[float],
    ) -> Array[float]:
        """Optimize weighted approximation"""

        alpha = np.sum(u / tau)
        alpha2 = np.sum(u / tau**2)

        beta = np.sum(1 / tau)
        beta2 = np.sum(1 / tau**2)

        b = (alpha*alpha2 - beta*np.sum(u**2 / tau**2)) / (alpha*beta2 - beta*alpha2)
        a = (alpha2 - b * beta2) / beta

        return np.array([a, b])
