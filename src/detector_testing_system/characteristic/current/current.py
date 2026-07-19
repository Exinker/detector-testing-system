from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from vmk_spectrum3_wrapper.types import Array, MilliSecond, U

from detector_testing_system.characteristic.gradient import (
    Gradient,
    calculate_gradient,
)
from detector_testing_system import ROOT
from detector_testing_system.data import (
    Trace,
    TraceFilter,
    filter_trace_factory,
)
from detector_testing_system.experiment import FitError
from detector_testing_system.types import AxesView


@dataclass
class Current:

    trace: Trace
    model: 'CurrentModelABC'
    mask: Array[bool]
    value: float
    bias: float
    xi: Array[U]

    @property
    def p(self) -> Array[float]:
        return [self.value, self.bias]

    def interpolate(self, __exposure: Array[MilliSecond]) -> Array[U]:
        return np.polyval(self.p, __exposure)

    def show(
        self,
        views: Sequence[AxesView | None] | None = None,
        verbose: bool = True,
        note: str = '',
    ) -> None:
        view_left, view_right = views or [{}, {}]

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)

        self._show_left(ax_left, view_left, verbose=verbose)
        self._show_right(ax_right, view_right, verbose=verbose, note=note)

        filedir = ROOT / 'img' / self.trace.label
        filedir.mkdir(parents=True, exist_ok=True)
        filename = 'dark-current-{name} ({n}).png'.format(
            name=self.model.name,
            n=self.trace.n,
        )
        plt.savefig(filedir / filename)

        plt.show()

    def _show_left(
        self,
        ax: Axes,
        view: AxesView,
        verbose: bool = True,
        color: str = 'red',
    ) -> None:
        view = view or {}

        plt.sca(ax)
        plt.scatter(
            self.trace.tau, self.trace.u,
            c='grey', s=10,
        )
        plt.scatter(
            self.trace.tau[self.mask], self.trace.u[self.mask],
            c=color, s=10,
            label=r'$U$',
        )
        plt.plot(
            self.trace.tau, self.interpolate(self.trace.tau),
            color='black', linestyle='-', linewidth=1,
            label=r'$\hat{U}$',
        )
        if verbose:
            plt.text(
                0.975, 0.975,
                '\n'.join([
                    r'$i$: {value:.4f} [{units}]'.format(
                        value=1e+3*self.value,  # in %/s
                        units=f'{self.trace.units.label}/s',
                    ),
                ]),
                transform=ax.transAxes,
                ha='right', va='top',
            )
            plt.text(
                0.975, 0.025,
                '\n'.join([
                    r'$a = {{{:.4f}}}$'.format(self.value),
                    r'$b = {{{:.4f}}}$'.format(self.bias),
                ]),
                transform=ax.transAxes,
                ha='right', va='bottom',
            )

        plt.xlabel(r'$\tau$ [{units}]'.format(units=r'$ms$'))
        plt.ylabel(r'$U$ [{units}]'.format(units=self.trace.units.label))

        plt.grid(color='grey', linestyle=':')
        plt.legend(loc='upper left')

        ax.set(**view)

    def _show_right(
        self,
        ax: Axes,
        view: AxesView,
        verbose: bool = True,
        color: str = 'red',
        note: str = '',
    ) -> None:
        view = view or {}

        plt.sca(ax)
        plt.scatter(
            self.trace.u, self.xi,
            c='grey', s=10,
        )
        plt.scatter(
            self.trace.u[self.mask], self.xi[self.mask],
            c=color, s=10,
            label=rf'$U_{{{self.trace.n}}}$',
        )
        if verbose:
            plt.text(
                0.975, 0.975,
                '\n'.join([
                    self.trace.label.prefix,
                    fr'$n: {{{self.trace.n}}}$',
                    note,
                ]),
                transform=ax.transAxes,
                ha='right', va='top',
            )
            plt.text(
                0.975, 0.025,
                '\n'.join([
                    r'$\xi = 100\frac{\hat{U} - U}{a \tau}$',
                ]),
                transform=ax.transAxes,
                ha='right', va='bottom',
            )

        plt.xlabel(r'$\tau$ [{units}]'.format(units=r'$ms$'))
        plt.ylabel(r'$\xi$ [{units}]'.format(units=self.trace.units.label))

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


class CurrentModelABC(ABC):

    def __init__(
        self,
        filter: TraceFilter | None = None,
    ) -> None:

        self.filter = filter or filter_trace_factory()

    def __init_subclass__(cls, *args, **kwargs):

        if 'name' not in cls.__dict__:
            raise TypeError(f'{cls.__name__} must have have "name" attribute')

        return super().__init_subclass__(*args, **kwargs)

    @abstractmethod
    def fit(self, trace: Trace) -> Current:
        pass

    @staticmethod
    def _calculate_xi(
        tau: Array[float],
        u: Array[float],
        p: Array[float],
    ) -> Array[float]:
        """Calculate a residual of approximation."""
        u_hat = np.polyval(p, tau)

        # xi = 100*(u_hat - u) / u
        xi = 100*(u_hat - u) / np.polyval([p[0], 0], tau)

        return xi


class BaseCurrentModel(CurrentModelABC):

    name = 'base'
    degree = 1

    def __init__(
        self,
        weighted: bool = True,
        filter: TraceFilter | None = None,
    ) -> None:
        super().__init__(filter=filter)

        self.weighted = weighted

    def fit(
        self,
        trace: Trace,
    ) -> Current:

        mask = self.filter(trace)
        if sum(mask) < self.degree + 1:
            raise FitError(
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

        return Current(
            trace=trace,
            model=self,
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
        """Optimize weighted approximation."""

        alpha = np.sum(u / tau)
        alpha2 = np.sum(u / tau**2)

        beta = np.sum(1 / tau)
        beta2 = np.sum(1 / tau**2)

        b = (alpha*alpha2 - beta*np.sum(u**2 / tau**2)) / (alpha*beta2 - beta*alpha2)
        a = (alpha2 - b * beta2) / beta

        return np.array([a, b])


class JNormCurrentModel(CurrentModelABC):

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

    def fit(self, trace: Trace) -> Current:

        gradient = calculate_gradient(trace=trace)

        mask = self.filter(trace)
        mask = self._build_mask(
            mask=mask,
            gradient=gradient,
        )
        if sum(mask) == 0:
            raise FitError(
                message=f'Data don\'t enough to be fitted! Linear fit calculation was failed in cell {trace.n}.',
            )

        value = float(np.mean(gradient.value[mask]))
        bias = float(np.mean(trace.u[mask] - value * trace.tau[mask]))

        xi = self._calculate_xi(
            tau=trace.tau,
            u=trace.u,
            p=np.array([value, bias]),
        )

        return Current(
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
        gradient: Gradient,
    ) -> Array[bool]:
        n_points = len(gradient.value)

        for n in range(n_points - self.min_points + 1):
            value = gradient.value[n:][mask[n:]]
            if len(value) < self.min_points:
                continue

            if self._relative_std(value) <= self.epsilon:
                return (np.arange(n_points) >= n) & mask

        return np.full(n_points, False)

    @staticmethod
    def _relative_std(values: Array[float]) -> float:

        mean = float(np.mean(values))
        if mean == 0 or not np.isfinite(mean):
            return float(np.inf)

        return float(np.std(values) / abs(mean))


def calculate_current(
    trace: Trace,
    model: CurrentModelABC | None = None,
) -> Current:
    model = model or BaseCurrentModel()

    try:
        result = model.fit(trace=trace)
    except FitError:
        raise FitError(
            message=f'Data don\'t enough to be fitted! Dark current calculation was failed in cell {trace.n}.',
        )

    return result
