from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tqdm.notebook import tqdm

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system import ROOT
from detector_testing_system.characteristic.current.current import (
    BaseCurrentModel,
    CurrentModelABC,
    calculate_current,
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
class CurrentResearch:

    data: Data
    model: CurrentModelABC
    value: Array[float]

    def show(
        self,
        confidence: float = .95,
        bins: int = 40,
        views: Sequence[AxesView | None] | None = None,
        color: str | None = None,
        verbose: bool = True,
        note: str = '',
    ) -> None:
        view_left, view_right = views or [{}, {}]

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)

        self._show_left(
            ax_left,
            view_left,
            confidence=confidence,
            color=color,
            verbose=verbose,
            note=note,
        )
        self._show_right(
            ax_right,
            view_right,
            confidence=confidence,
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

    def _show_left(
        self,
        ax: Axes,
        view: AxesView,
        confidence: float,
        color: str | None = None,
        verbose: bool = True,
        note: str = '',
    ) -> None:
        view = view or {}
        color = color or 'red'

        number = np.arange(self.data.n_numbers)
        lb, ub = calculate_outlier_bounds(1e+3*self.value, k=3)
        dark_current_trunked = trunk_outliers(1e+3*self.value, (lb, ub))
        mean, ci = calculate_stats(dark_current_trunked, confidence=confidence)

        plt.sca(ax)
        plt.scatter(
            number, 1e+3*self.value,
            c='black', s=2,
            label='$i_{{d}}$',
        )
        plt.axhline(
            mean,
            color='red', linestyle='--', linewidth=1,
        )
        if verbose:
            plt.text(
                0.025, 0.975,
                '\n'.join([
                    self.data.label.prefix,
                    'method: {method}'.format(
                        method=getattr(self.model, 'name', 'base'),
                    ),
                    r'$i_{{d}}$: {mean:.4f} $\pm$ {ci:.4f} [%/s]'.format(
                        mean=mean,
                        ci=ci,
                    ),
                    note,
                ]),
                transform=ax.transAxes,
                ha='left', va='top',
            )

        ylim_max = min(1.5 * np.nanmax(1e+3*self.value), 5*ub)
        ylim_min = np.nanmin(1e+3*self.value) - .025*(ylim_max - np.nanmin(1e+3*self.value))
        plt.ylim([
            ylim_min,
            ylim_max,
        ])
        plt.xlabel(r'$number$')
        plt.ylabel(r'$i_{{d}}$ [{units}]'.format(
            units=f'{self.data.units.label}/s',
        ))
        plt.grid(color='grey', linestyle=':')

        ax.set(**view)

    def _show_right(
        self,
        ax: Axes,
        view: AxesView,
        confidence: float,
        bins: int,
        color: str | None = None,
        verbose: bool = True,
    ) -> None:
        view = view or {}
        color = color or 'red'

        lb, ub = calculate_outlier_bounds(self.value, k=3)
        dark_current_trunked = trunk_outliers(self.value, (lb, ub))
        mean, ci = calculate_stats(dark_current_trunked, confidence=confidence)

        plt.sca(ax)
        plt.hist(
            1e+3*dark_current_trunked,
            bins=bins,
            edgecolor='black', facecolor='white',
            # fill=False,
        )
        plt.axvline(
            1e+3*mean,
            color='red', linestyle='--', linewidth=1,
        )

        plt.xlabel(r'$i_{{d}}$ [{units}]'.format(
            units=f'{self.data.units.label}/s',
        ))
        plt.ylabel('count')

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


def research_current(
    data: Data,
    model: CurrentModelABC | None = None,
    mask: Array[bool] | None = None,
) -> CurrentResearch:
    """Calculate a dark current of the cells."""
    model = model or BaseCurrentModel()
    mask = np.full(data.n_numbers, True) if mask is None else mask

    value = np.full(data.n_numbers, np.nan)
    for n, *_ in tqdm(np.argwhere(mask)):
        try:
            dark_current = calculate_current(
                trace=data.trace(n),
                model=model,
            )
            value[n] = dark_current.value
        except FitError:
            value[n] = float(np.nan)

    return CurrentResearch(
        data=data,
        model=model,
        value=value,
    )
