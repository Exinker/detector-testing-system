from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system import ROOT
from detector_testing_system.data import Trace
from detector_testing_system.types import AxesView


@dataclass
class GradientResult:

    trace: Trace
    value: Array[float]

    def show(
        self,
        views: Sequence[AxesView | None] | None = None,
        verbose: bool = False,
    ) -> None:
        view_left, view_right = views or [None, None]


        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        self._plot_left(ax_left, view_left, verbose=verbose)
        self._plot_right(ax_right, view_right, verbose=verbose)

        filedir = ROOT / 'img' / self.trace.label
        filedir.mkdir(parents=True, exist_ok=True)
        filename = 'gradient ({n}).png'.format(
            n=self.trace.n,
        )
        plt.savefig(filedir / filename)

        plt.show()

    def _plot_left(
        self,
        ax: Axes,
        view: AxesView | None,
        verbose: bool = False,
    ) -> None:
        view = view or {}

        p = np.polyfit(self.trace.tau, self.trace.u, deg=1)

        plt.sca(ax)
        plt.scatter(
            self.trace.tau, self.trace.u,
            c='red', s=10,
            label=rf'$U_{{{self.trace.n}}}$',
        )
        plt.plot(
            self.trace.tau, np.polyval(p, self.trace.tau),
            color='black', linestyle='solid', linewidth=1,
            label=r'$\hat{U}$',
        )
        if verbose:
            ax.text(
                0.95, 0.05/2,
                '\n'.join([
                    fr'$a = {{{p[0]:.4f}}}$',
                    fr'$b = {{{p[1]:.4f}}}$',
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
        verbose: bool = False,
    ) -> None:
        view = view or {}

        plt.sca(ax)
        plt.scatter(
            self.trace.u, self.value,
            c='red', s=10,
        )

        if verbose:
            plt.text(
                0.95, 0.95,
                '\n'.join([
                    self.trace.label.prefix,
                ]),
                transform=ax.transAxes,
                ha='right', va='top',
            )

        plt.xlabel(r'$U$ {units}'.format(units=self.trace.units.label))
        plt.ylabel(r'$dU / d\tau$')

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


def calculate_gradient(
    trace: Trace,
) -> GradientResult:
    """Calculate gradient"""

    value = np.gradient(trace.u, trace.tau)

    return GradientResult(
        trace=trace,
        value=value,
    )
