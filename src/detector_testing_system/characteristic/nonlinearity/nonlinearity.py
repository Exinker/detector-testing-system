import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from vmk_spectrum3_wrapper.types import Array, U

from detector_testing_system import ROOT
from detector_testing_system.characteristic.dark_current import (
    BaseDarkCurrentModel,
    DarkCurrentModelABC,
    DarkCurrent,
    JNormDarkCurrentModel,
)
from detector_testing_system.characteristic.gradient import Gradient, calculate_gradient
from detector_testing_system.data import Trace
from detector_testing_system.types import AxesView


LOGGER = logging.getLogger(__name__)


@dataclass
class NonlinearityABC(ABC):

    trace: Trace
    model: DarkCurrentModelABC
    dark_current: DarkCurrent
    value: float

    def show(
        self,
        views: Sequence[AxesView | None] | None = None,
        verbose: bool = False,
    ) -> None:
        view_left, view_right = views or [{}, {}]

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        self._plot_left(ax_left, view_left, verbose=verbose)
        self._plot_right(ax_right, view_right, verbose=verbose)

        filedir = ROOT / 'img' / self.trace.label
        filedir.mkdir(parents=True, exist_ok=True)
        filename = 'nonlinearity-{name} ({n}).png'.format(
            name=self.model.name,
            n=self.trace.n,
        )
        plt.savefig(filedir / filename)

        plt.show()

    @abstractmethod
    def _plot_left(
        self,
        ax: Axes,
        view: AxesView | None,
        color: str | None = None,
        verbose: bool = False,
        label: str | None = None,
        hat_label: str | None = r'$\hat{U}$',
    ) -> None:
        pass

    @abstractmethod
    def _plot_right(
        self,
        ax: Axes,
        view: AxesView | None,
        color: str | None = None,
        verbose: bool = False,
    ) -> None:
        pass


class BaseNonlinearity(NonlinearityABC):

    def _plot_left(
        self,
        ax: Axes,
        view: AxesView | None,
        color: str | None = None,
        verbose: bool = False,
        label: str | None = None,
        hat_label: str | None = r'$\hat{U}$',
    ) -> None:
        trace = self.dark_current.trace
        mask = self.dark_current.mask

        view = view or {}
        color = color or 'red'
        label = label or rf'$U_{{{trace.n}}}$'

        plt.sca(ax)
        plt.scatter(
            trace.tau, trace.u,
            c='grey', s=10,
        )
        plt.scatter(
            trace.tau[mask], trace.u[mask],
            c=color, s=10,
            label=label,
        )
        plt.plot(
            trace.tau, self.dark_current.interpolate(trace.tau),
            color='black', linestyle='solid', linewidth=1,
            label=hat_label,
        )
        if verbose:
            plt.text(
                0.95, 0.05/2,
                '\n'.join([
                    fr'$a = {{{self.dark_current.value:.4f}}}$',
                    fr'$b = {{{self.dark_current.bias:.4f}}}$',
                ]),
                transform=ax.transAxes,
                ha='right', va='bottom',
            )

        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ [{units}]'.format(units=trace.units.label))
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        ax.set(**view)

    def _plot_right(
        self,
        ax: Axes,
        view: AxesView | None,
        color: str | None = None,
        verbose: bool = False,
    ) -> None:
        view = view or {}
        color = color or 'red'

        trace = self.dark_current.trace
        mask = self.dark_current.mask

        plt.sca(ax)
        plt.scatter(
            trace.u, self.dark_current.xi,
            c='grey', s=10,
        )
        plt.scatter(
            trace.u[mask], self.dark_current.xi[mask],
            c=color, s=10,
            label=rf'$U_{{{trace.n}}}$',
        )
        if verbose:
            plt.text(
                0.95, 0.95,
                '\n'.join([
                    trace.label.prefix,
                    fr'$\alpha: {{{self.value:.2f}}}$ [%]',
                ]),
                transform=ax.transAxes,
                ha='right', va='top',
            )
            plt.text(
                0.95, 0.05/2,
                '\n'.join([
                    r'$error = 100\frac{\hat{U} - U_{i}}{a \tau}$',
                ]),
                transform=ax.transAxes,
                ha='right', va='bottom',
            )

        plt.xlabel(r'$U$ [{units}]'.format(units=trace.units.label))
        plt.ylabel(r'$error$ [%]')

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


@dataclass
class JNormNonlinearity(NonlinearityABC):

    trace: Trace
    model: DarkCurrentModelABC
    dark_current: DarkCurrent
    value: float
    k: float

    def _plot_left(
        self,
        ax: Axes,
        view: AxesView | None,
        color: str | None = None,
        verbose: bool = False,
        label: str | None = None,
        hat_label: str | None = r'$\hat{U}$',
    ) -> None:
        trace = self.dark_current.trace
        mask = self.dark_current.mask

        view = view or {}
        color = color or 'red'
        label = label or rf'$U_{{{trace.n}}}$'

        plt.sca(ax)
        plt.scatter(
            trace.tau, trace.u,
            c='grey', s=10,
        )
        plt.scatter(
            trace.tau[mask], trace.u[mask],
            c=color, s=10,
            label=label,
        )
        plt.plot(
            trace.tau, self.dark_current.interpolate(trace.tau),
            color='black', linestyle='solid', linewidth=1,
            label=hat_label,
        )
        if verbose:
            plt.text(
                0.95, 0.05/2,
                '\n'.join([
                    fr'$a = {{{self.dark_current.value:.4f}}}$',
                    fr'$b = {{{self.dark_current.bias:.4f}}}$',
                ]),
                transform=ax.transAxes,
                ha='right', va='bottom',
            )

        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ [{units}]'.format(units=trace.units.label))

        plt.grid(color='grey', linestyle=':')
        plt.legend()

        ax.set(**view)

    def _plot_right(
        self,
        ax: Axes,
        view: AxesView | None,
        color: str | None = None,
        verbose: bool = False,
    ) -> None:
        view = view or {}
        color = color or 'red'

        trace = self.dark_current.trace
        gradient = calculate_gradient(trace=trace)
        mask = self.dark_current.mask

        plt.sca(ax)
        plt.scatter(
            trace.u, gradient.value,
            c='grey', s=10,
        )
        plt.scatter(
            trace.u[mask], gradient.value[mask],
            c=color, s=10,
            label=rf'$U_{{{trace.n}}}$',
        )
        plt.axhline(
            self.dark_current.value,
            color='black', linestyle='solid', linewidth=1,
        )
        if verbose and self.value > 0:
            plt.axhline(
                self.k * self.dark_current.value,
                color='red', linestyle='--', linewidth=1,
            )
            plt.axvspan(
                trace.u[0],
                trace.u[0] + self.value,
                color='grey',
                alpha=.125,
            )
        if verbose:
            plt.text(
                0.95, 0.95,
                '\n'.join([
                    trace.label.prefix,
                    fr'$\Delta U$: {self.value:.2f} [%]',
                ]),
                transform=ax.transAxes,
                ha='right', va='top',
            )

        plt.xlabel(r'$U$ [{units}]'.format(units=trace.units.label))
        plt.ylabel(r'$dU / d\tau$')

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


def calculate_nonlinearity(
    trace: Trace,
    model: DarkCurrentModelABC | None = None,
    **kwargs,
) -> NonlinearityABC:
    model = model or BaseDarkCurrentModel()

    if isinstance(model, BaseDarkCurrentModel):
        return _calculate_nonlinearity_base(
            trace=trace,
            model=model,
            **kwargs,
        )

    if isinstance(model, JNormDarkCurrentModel):
        return _calculate_nonlinearity_jnorm(
            trace=trace,
            model=model,
            **kwargs,
        )

    raise TypeError('`BaseDarkCurrentModel` and `JNormDarkCurrentModel` are supported only!')


def _calculate_nonlinearity_base(
    trace: Trace,
    model: BaseDarkCurrentModel,
) -> BaseNonlinearity:

    if not model.weighted:
        raise ValueError('To calculate nonlinearity use weighted model only!')

    dark_current = model.fit(trace)
    alpha = _calculate_alpha(xi=dark_current.xi)

    return BaseNonlinearity(
        trace=trace,
        model=model,
        dark_current=dark_current,
        value=alpha,
    )


def _calculate_alpha(xi: Array[U]) -> float:
    """Calculate nonlinearity coefficient (alpha)"""
    return (np.max(xi) - np.min(xi)) / 2


def _calculate_nonlinearity_jnorm(
    trace: Trace,
    model: JNormDarkCurrentModel,
    k: float = 2,
) -> JNormNonlinearity:

    gradient = calculate_gradient(trace=trace)
    dark_current = model.fit(trace=trace)

    x_intersection = _calculate_intersection(
        u=trace.u,
        gradient=gradient,
        threshold=k * dark_current.value,
    )
    if x_intersection is None:
        span = 0
    else:
        span = x_intersection - trace.u[0]

    return JNormNonlinearity(
        trace=trace,
        model=model,
        dark_current=dark_current,
        value=span,
        k=k,
    )


def _calculate_intersection(
    u: Array[float],
    gradient: Gradient,
    threshold: float,
) -> float | None:

    diff = gradient.value - threshold
    exact_indexes = np.argwhere(diff == 0).ravel()
    if len(exact_indexes) > 0:
        return float(u[exact_indexes[-1]])

    crossing_indexes = np.argwhere(diff[:-1] * diff[1:] < 0).ravel()
    if len(crossing_indexes) == 0:
        return None

    i = int(crossing_indexes[-1])
    x0, x1 = u[i], u[i + 1]
    y0, y1 = gradient.value[i], gradient.value[i + 1]
    if y1 == y0:
        x_intersection = float(x0)
    else:
        x_intersection = float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))

    return x_intersection
