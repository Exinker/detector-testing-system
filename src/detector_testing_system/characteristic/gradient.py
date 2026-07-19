from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system import ROOT
from detector_testing_system.data import Trace
from detector_testing_system.types import AxesView

CMAP = plt.get_cmap('tab10')


@dataclass
class Gradient:

    trace: Trace
    value: Array[float]

    def show(
        self,
        views: Sequence[AxesView | None] | None = None,
        note: str = '',
    ) -> None:
        view_left, view_right = views or [None, None]

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)

        self._show_left(ax_left, view_left)
        self._show_right(ax_right, view_right, note=note)

        filedir = ROOT / 'img' / self.trace.label
        filedir.mkdir(parents=True, exist_ok=True)
        filename = 'gradient ({n}).png'.format(
            n=self.trace.n,
        )
        plt.savefig(filedir / filename)

        plt.show()

    def _show_left(
        self,
        ax: Axes,
        view: AxesView | None,
        verbose: bool = True,
        color: str | None = 'red',
        label: str | None = r'$U$',
    ) -> None:
        view = view or {}

        plt.sca(ax)
        plt.scatter(
            self.trace.tau, self.trace.u,
            c=color, s=10,
            label=label,
        )

        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ [{units}]'.format(units=self.trace.units.label))
        plt.grid(color='grey', linestyle=':')

        plt.legend(loc='upper left')

        ax.set(**view)

    def _show_right(
        self,
        ax: Axes,
        view: AxesView | None,
        verbose: bool = True,
        color: str | None = 'red',
        note: str = '',
    ) -> None:
        view = view or {}

        plt.sca(ax)
        plt.scatter(
            self.trace.u, 1e+3*self.value,
            c=color, s=10,
        )
        if verbose:
            plt.text(
                0.975, 0.975,
                '\n'.join([
                    self.trace.label.prefix,
                    r'n: {}'.format(self.trace.n),
                    note,
                ]),
                transform=ax.transAxes,
                ha='right', va='top',
            )

        plt.xlabel(r'$U$ [{units}]'.format(units=self.trace.units.label))
        plt.ylabel(r'$dU / d\tau$ [%/s]')

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


def calculate_gradient(trace: Trace) -> Gradient:
    """Calculate gradient."""

    value = np.gradient(trace.u, trace.tau)

    return Gradient(
        trace=trace,
        value=value,
    )


def compare_gradient(
    __traces: Sequence[tuple[Trace, str]],
    views: Sequence[AxesView | None] | None = None,
) -> None:
    view_left, view_right = views or [None, None]

    fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)
    for i, (trace, label) in enumerate(__traces):
        color = CMAP(i % 10)

        gradient = calculate_gradient(trace=trace)
        gradient._show_left(
            ax_left,
            view_left,
            color=color,
            verbose=False,
            label=label,
        )
        gradient._show_right(
            ax_right,
            view_right,
            color=color,
            verbose=False,
        )

    filedir = ROOT / 'img'
    filedir.mkdir(parents=True, exist_ok=True)
    filepath = filedir / 'gradient {n}.png'.format(
        n='x'.join([str(trace.n) for trace, _ in __traces]),
    )
    plt.savefig(filepath)

    plt.show()
