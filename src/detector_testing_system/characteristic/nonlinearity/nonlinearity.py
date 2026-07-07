import logging
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
class Nonlinearity:

    trace: Trace
    model: DarkCurrentModelABC
    dark_current: DarkCurrent
    value: float

    def show(
        self,
        views: Sequence[AxesView | None] | None = None,
        verbose: bool = True,
        note: str | None = None,
    ) -> None:
        view_left, view_right = views or [{}, {}]

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        self._plot_left(ax_left, view_left, verbose=verbose)
        self._plot_right(ax_right, view_right, verbose=verbose, note=note)

        filedir = ROOT / 'img' / self.trace.label
        filedir.mkdir(parents=True, exist_ok=True)
        filename = 'nonlinearity-{name} ({n}).png'.format(
            name=self.model.name,
            n=self.trace.n,
        )
        plt.savefig(filedir / filename)

        plt.show()

    def _plot_left(
        self,
        ax: Axes,
        view: AxesView | None,
        color: str | None = None,
        verbose: bool = True,
        label: str | None = None,
        hat_label: str | None = r'$\hat{U}$',
    ) -> None:
        trace = self.dark_current.trace
        mask = self.dark_current.mask

        view = view or {}
        color = color or 'red'
        label = label or r'$U$'

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
        verbose: bool = True,
        note: str | None = None,
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
            label=r'$U$',
        )
        if verbose:
            plt.text(
                0.95, 0.05/2,
                '\n'.join([
                    r'$error = 100\frac{\hat{U} - U_{i}}{a \tau}$',
                ]),
                transform=ax.transAxes,
                ha='right', va='bottom',
            )
            plt.text(
                0.95, 0.95,
                '\n'.join([
                    trace.label.prefix,
                    fr'$\alpha: {{{self.value:.2f}}}$ [%]',
                    fr'$n: {{{self.trace.n}}}$',
                    note if note else '',
                ]),
                transform=ax.transAxes,
                ha='right', va='top',
            )

        plt.xlabel(r'$U$ [{units}]'.format(units=trace.units.label))
        plt.ylabel(r'$error$ [%]')

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


def calculate_nonlinearity(
    trace: Trace,
    model: DarkCurrentModelABC | None = None,
    verbose: bool = True,
    **kwargs,
) -> Nonlinearity:
    model = model or BaseDarkCurrentModel()

    if isinstance(model, BaseDarkCurrentModel):
        return _calculate_nonlinearity_base(
            trace=trace,
            model=model,
            verbose=verbose,
            **kwargs,
        )

    if isinstance(model, JNormDarkCurrentModel):
        return _calculate_nonlinearity_jnorm(
            trace=trace,
            model=model,
            verbose=verbose,
            **kwargs,
        )

    raise TypeError('`BaseDarkCurrentModel` and `JNormDarkCurrentModel` are supported only!')


def _calculate_nonlinearity_base(
    trace: Trace,
    model: BaseDarkCurrentModel,
    verbose: bool = True,
) -> Nonlinearity:

    if not model.weighted:
        raise ValueError('To calculate nonlinearity use weighted model only!')

    dark_current = model.fit(trace)
    alpha = _calculate_alpha(xi=dark_current.xi)

    return Nonlinearity(
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
    verbose: bool = True,
) -> Nonlinearity:

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

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))

        plt.sca(ax)
        plt.scatter(
            trace.u, gradient.value,
            c='grey', s=10,
        )
        plt.scatter(
            trace.u[dark_current.mask], gradient.value[dark_current.mask],
            c='red', s=10,
            label=r'$U$',
        )
        plt.axhline(
            dark_current.value,
            color='black', linestyle='solid', linewidth=1,
        )
        plt.text(
            0.975, 0.95,
            '\n'.join([
                trace.label.prefix,
                r'$i$: {value:.4f} [{units}]'.format(
                    value=1e+3*np.nanmean(gradient.value[dark_current.mask]),  # in %/s
                    units=f'{trace.units.label}/s',
                ),
                fr'$\Delta U: {{{span:.2f}}}$ [%]',
                fr'$n: {{{trace.n}}}$',
            ]),
            transform=ax.transAxes,
            ha='right', va='top',
        )
        if span > 0:
            plt.axhline(
                k * dark_current.value,
                color='red', linestyle='--', linewidth=1,
            )
            plt.axvspan(
                trace.u[0],
                trace.u[0] + span,
                color='grey',
                alpha=.125,
            )

        plt.xlabel(r'$U$ [{units}]'.format(units=trace.units.label))
        plt.ylabel(r'$dU / d\tau$')

        plt.grid(color='grey', linestyle=':')
        plt.show()


    return Nonlinearity(
        trace=trace,
        model=model,
        dark_current=dark_current,
        value=span,
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
