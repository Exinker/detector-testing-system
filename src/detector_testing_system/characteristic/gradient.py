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
class Gradient:

    trace: Trace
    value: Array[float]

    def show(
        self,
        views: Sequence[AxesView | None] | None = None,
    ) -> None:
        view_left, view_right = views or [None, None]


        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        self._plot_left(ax_left, view_left)
        self._plot_right(ax_right, view_right)

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
    ) -> None:
        view = view or {}

        plt.sca(ax)
        plt.scatter(
            self.trace.tau, self.trace.u,
            c='red', s=10,
            label=r'$U$',
        )

        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ [{units}]'.format(units=self.trace.units.label))

        plt.grid(color='grey', linestyle=':')
        plt.legend()

        ax.set(**view)

    def _plot_right(
        self,
        ax: Axes,
        view: AxesView | None,
    ) -> None:
        view = view or {}

        plt.sca(ax)
        plt.scatter(
            self.trace.u, self.value,
            c='red', s=10,
        )
        plt.text(
            0.95, 0.95,
            '\n'.join([
                self.trace.label.prefix,
            ]),
            transform=ax.transAxes,
            ha='right', va='top',
        )

        plt.xlabel(r'$U$ [{units}]'.format(units=self.trace.units.label))
        plt.ylabel(r'$dU / d\tau$')

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


def calculate_gradient(
    trace: Trace,
) -> Gradient:
    """Calculate gradient"""

    value = np.gradient(trace.u, trace.tau)

    return Gradient(
        trace=trace,
        value=value,
    )
