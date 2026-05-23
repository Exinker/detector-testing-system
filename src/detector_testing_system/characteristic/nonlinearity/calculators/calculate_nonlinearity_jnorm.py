from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.characteristic.dark_current.models import (
    DarkCurrentModelABC,
    DarkCurrentResult,
    JNormDarkCurrentModel,
)
from detector_testing_system.characteristic.nonlinearity.results import NonlinearityResultABC
from detector_testing_system.characteristic.gradient import (
    GradientResult,
    calculate_gradient,
)
from detector_testing_system.data import Trace
from detector_testing_system.types import AxesView


@dataclass
class JNormNonlinearityResult(NonlinearityResultABC):

    trace: Trace
    model: DarkCurrentModelABC
    dark_current: DarkCurrentResult
    value: float
    k: int

    def _plot_left(
        self,
        ax: Axes,
        view: AxesView | None,
        color: str | None = None,
        verbose: bool = False,
        hat_label: str | None = r'$\hat{U}$',
    ) -> None:
        view = view or {}
        color = color or 'red'

        plt.sca(ax)
        plt.scatter(
            self.trace.tau, self.trace.u,
            c='grey', s=10,
        )
        plt.scatter(
            self.trace.tau[self.dark_current.mask], self.trace.u[self.dark_current.mask],
            c=color, s=10,
            label=rf'$U_{{{self.trace.n}}}$',
        )
        plt.plot(
            self.trace.tau, self.dark_current.interpolate(self.trace.tau),
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
        plt.ylabel(r'$U$ {units}'.format(units=self.trace.units.label))

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

        gradient = calculate_gradient(trace=self.trace)

        plt.sca(ax)
        plt.scatter(
            self.trace.u, gradient.value,
            c='grey', s=10,
        )
        plt.scatter(
            self.trace.u[self.dark_current.mask], gradient.value[self.dark_current.mask],
            c=color, s=10,
            label=rf'$U_{{{self.trace.n}}}$',
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
                self.trace.u[0],
                self.trace.u[0] + self.value,
                color='grey',
                alpha=.125,
            )
        if verbose:
            plt.text(
                0.95, 0.95,
                '\n'.join([
                    self.trace.label.prefix,
                    fr'$\Delta U$: {self.value:.2f} [%]',
                ]),
                transform=ax.transAxes,
                ha='right', va='top',
            )

        plt.xlabel(r'$U$ {units}'.format(units=self.trace.units.label))
        plt.ylabel(r'$dU / d\tau$')

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


def calculate_nonlinearity_jnorm(
    trace: Trace,
    model: JNormDarkCurrentModel,
    k: float = 2,
) -> JNormNonlinearityResult:

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

    return JNormNonlinearityResult(
        trace=trace,
        model=model,
        dark_current=dark_current,
        value=span,
        k=k,
    )


def _calculate_intersection(
    u: Array[float],
    gradient: GradientResult,
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
