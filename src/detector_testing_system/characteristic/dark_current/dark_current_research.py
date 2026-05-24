from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system import ROOT
from detector_testing_system.characteristic.dark_current.dark_current import (
    BaseDarkCurrentModel,
    DarkCurrentModelABC,
    calculate_dark_current,
)
from detector_testing_system.data import Data
from detector_testing_system.experiment import FitError
from detector_testing_system.types import AxesView
from detector_testing_system.utils import (
    calculate_outlier_bounds,
    calculate_stats,
    trunk_outliers,
)


@dataclass
class DarkCurrentResearch:

    data: Data
    model: DarkCurrentModelABC
    value: Array[float]

    def show(
        self,
        confidence: float = .95,
        bins: int = 40,
        views: Sequence[AxesView | None] | None = None,
        color: str | None = None,
        verbose: bool = False,
    ) -> None:
        view_left, view_right = views or [{}, {}]

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        self._plot_left(
            ax_left,
            view_left,
            confidence=confidence,
            color=color,
            verbose=verbose,
        )
        self._plot_right(
            ax_right,
            view_right,
            bins=bins,
            color=color,
            verbose=verbose,
        )

        filedir = ROOT / 'img' / self.data.label
        filedir.mkdir(parents=True, exist_ok=True)
        filename = 'dark-current-{name}.png'.format(
            name=self.model.name,
        )
        plt.savefig(filedir / filename)

        plt.show()

    def _plot_left(
        self,
        ax: Axes,
        view: AxesView,
        confidence: float = .95,
        color: str | None = None,
        verbose: bool = False,
    ) -> None:
        view = view or {}
        color = color or 'red'

        number = np.arange(self.data.n_numbers)
        lb, ub = calculate_outlier_bounds(self.value, k=3)
        dark_current_trunked = trunk_outliers(self.value, (lb, ub))
        mean, ci = calculate_stats(dark_current_trunked, confidence=confidence)

        plt.sca(ax)
        plt.scatter(
            number, self.value,
            c='black', s=2,
        )
        if verbose:
            plt.text(
                0.05/2, 0.95,
                '\n'.join([
                    r'$i_{{d}}$: {mean:.4f} $\pm$ {ci:.4f} [%/s]'.format(
                        mean=1e+3*mean,
                        ci=1e+3*ci,
                    ),
                ]),
                transform=ax.transAxes,
                ha='left', va='top',
            )

        plt.xlabel(r'$number$')
        plt.ylabel(r'$i_{{d}}$ [{units}]'.format(
            units=f'{self.data.units.label}/s',
        ))

        ylim_max = min(1.5 * np.nanmax(self.value), 5*ub)
        ylim_min = np.nanmin(self.value) - .025*(ylim_max - np.nanmin(self.value))
        plt.ylim([
            ylim_min,
            ylim_max,
        ])

        plt.grid(color='grey', linestyle=':')
        plt.legend()

        ax.set(**view)

    def _plot_right(
        self,
        ax: Axes,
        view: AxesView,
        bins: int,
        color: str | None = None,
        verbose: bool = False,
    ) -> None:
        view = view or {}
        color = color or 'red'

        lb, ub = calculate_outlier_bounds(self.value, k=3)
        dark_current_trunked = trunk_outliers(self.value, (lb, ub))

        plt.sca(ax)
        plt.hist(
            1e+3*dark_current_trunked,
            bins=bins,
            edgecolor='black', facecolor='white',
            # fill=False,
        )

        plt.xlabel(r'$i_{{d}}$ [{units}]'.format(
            units=f'{self.data.units.label}/s',
        ))

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


def research_dark_current(
    data: Data,
    model: DarkCurrentModelABC | None = None,
    mask: Array[bool] | None = None,
) -> DarkCurrentResearch:
    """Calculate a dark current of the cells"""
    model = model or BaseDarkCurrentModel()
    mask = np.full(data.n_numbers, True) if mask is None else mask

    value = np.full(data.n_numbers, np.nan)
    for n, *_ in np.argwhere(mask):
        try:
            dark_current = calculate_dark_current(
                trace=data.trace(n),
                model=model,
            )
            value[n] = dark_current.value
        except FitError:
            value[n] = float(np.nan)

    return DarkCurrentResearch(
        data=data,
        model=model,
        value=value,
    )
