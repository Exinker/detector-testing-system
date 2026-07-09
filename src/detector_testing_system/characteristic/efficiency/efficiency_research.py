import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from distfit import distfit
from matplotlib.axes import Axes
from tqdm.notebook import tqdm

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system import ROOT
from detector_testing_system.characteristic.efficiency import calculate_efficiency
from detector_testing_system.data import (
    Data,
    Trace,
)
from detector_testing_system.experiment import FitError
from detector_testing_system.utils import (
    calculate_outlier_bounds,
    calculate_stats,
    normalize,
    trunk_outliers,
)
from detector_testing_system.types import AxesView

LOGGER = logging.getLogger(__name__)
CMAP = plt.get_cmap('tab10')


@dataclass
class EfficiencyResearch:

    data: Data
    value: Array[float]

    def show(
        self,
        bins: int | Sequence = 40,
        views: Sequence[AxesView | None] | None = None,
        verbose: bool = True,
        note: str = '',
    ) -> None:
        view_left, view_right = views or [None, None]

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        self._show_left(ax_left, view_left, verbose=verbose, note=note)
        self._show_right(ax_right, view_right, bins=bins, verbose=verbose)

        plt.show()

    def _show_left(
        self,
        ax: Axes,
        view: AxesView | None,
        verbose: bool = True,
        note: str = '',
        color: str | None = 'red',
        label: str | None = r'$U$',
        hat_label: str | None = r'$\hat{U}$',
    ) -> None:
        view = view or {}

        number = np.arange(self.data.n_numbers)
        lb, ub = calculate_outlier_bounds(self.value, k=3)
        efficiency_trunked = trunk_outliers(self.value, (lb, ub))
        mean, ci = calculate_stats(efficiency_trunked)

        plt.sca(ax)

        plt.scatter(
            number, self.value,
            label=label,
            c=color, s=2,
        )
        plt.axhline(
            mean,
            label=hat_label,
            color='black', linestyle='-', linewidth=1,
        )
        if verbose:
            plt.text(
                0.05/2, 0.95,
                '\n'.join([
                    self.data.label.prefix,
                    r'$k: {mean:.0f} \pm {ci:.0f} \text{{ [e}}^{{-}}\text{{/\%]}}$'.format(
                        mean=mean,
                        ci=ci,
                    ),
                    note,
                ]),
                transform=ax.transAxes,
                ha='left', va='top',
            )

        ylim_max = min(1.5 * np.nanmax(efficiency_trunked), 5*ub)
        ylim_min = np.nanmin(efficiency_trunked) - .025*(ylim_max - np.nanmin(efficiency_trunked))
        plt.ylim([
            ylim_min,
            ylim_max,
        ])
        plt.xlabel(r'$number$')
        plt.ylabel(r'k [$e^{-}/\%$]')
        plt.grid(color='grey', linestyle=':')

        ax.set(**view)

    def _show_right(
        self,
        ax: Axes,
        view: AxesView | None,
        bins: int,
        verbose: bool = True,
    ) -> None:
        view = view or {}

        lb, ub = calculate_outlier_bounds(self.value, k=3)
        efficiency_trunked = trunk_outliers(self.value, (lb, ub))

        plt.sca(ax)

        plt.hist(
            efficiency_trunked[~np.isnan(efficiency_trunked)],
            bins=bins,
            edgecolor='black', facecolor='white',
        )

        plt.xlabel(r'k [$e^{-}/\%$]')
        plt.ylabel(r'count')
        plt.grid(color='grey', linestyle=':')

        ax.set(**view)

    def stat(self) -> None:

        lb, ub = calculate_outlier_bounds(self.value, k=3)
        value_trunked = trunk_outliers(self.value, (lb, ub))
        value_normalized = normalize(value_trunked)

        dfit = distfit(
            distr='norm',
        )

        dfit.fit_transform(
            value_normalized,
            verbose=False,
        )

        #
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

        dfit.plot(
            chart='pdf',
            ax=ax_left,
        )
        dfit.qqplot(
            value_normalized,
            ax=ax_right,
        )


def research_efficiency(
    data: Data,
    filter: Callable[[Trace], Array[bool]] | None = None,
    mask: Array[bool] | None = None,
) -> EfficiencyResearch:
    mask = np.full(data.n_numbers, True) if mask is None else mask

    value = np.full(data.n_numbers, np.nan)
    for n, *_ in np.argwhere(mask):
        try:
            result = calculate_efficiency(
                trace=data.trace(n),
                filter=filter
            )
            value[n] = result.value
        except FitError as error:
            # LOGGER.error(
            #     'Calculate nonlinearity (n: %d): %s',
            #     n,
            #     error,
            # )
            value[n] = float(np.nan)

        except Exception as error:
            print(error)

    return EfficiencyResearch(
        data=data,
        value=value,
    )
