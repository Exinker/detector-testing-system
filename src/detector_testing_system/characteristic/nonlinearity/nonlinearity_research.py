import logging
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tqdm.notebook import tqdm

from vmk_spectrum3_wrapper.types import Array, Number

from detector_testing_system import ROOT
from detector_testing_system.characteristic.current import (
    BaseCurrentModel,
    CurrentModelABC,
)
from detector_testing_system.characteristic.nonlinearity.nonlinearity import (
    calculate_nonlinearity,
)
from detector_testing_system.data import Data, Trace
from detector_testing_system.experiment import FitError
from detector_testing_system.types import AxesView

LOGGER = logging.getLogger(__name__)
CMAP = plt.get_cmap('tab10')


@dataclass
class NonlinearityResearch:

    data: Data
    model: CurrentModelABC
    value: Array[float]

    @property
    def number(self) -> Array[Number]:
        return np.arange(self.data.n_numbers)

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
    ) -> None:
        view = view or {}

        plt.sca(ax)

        plt.scatter(
            self.number, self.value,
            c='red', s=10,
            label=r'$U$',
        )
        if verbose:
            plt.text(
                0.05/2, 0.95,
                '\n'.join([
                    self.data.label.prefix,
                    'method: {method}'.format(
                        method=getattr(self.model, 'name', 'base'),
                    ),
                    {
                        'base': fr'$\alpha: {{{np.nanmean(self.value):.2f}}}$ [%]',
                        'jnorm': fr'$\Delta U: {{{np.nanmean(self.value):.2f}}}$ [%]',
                    }[getattr(self.model, 'name', 'base')],
                    note,
                ]),
                transform=ax.transAxes,
                ha='left', va='top',
            )

        plt.xlabel(r'number')
        plt.ylabel({
            'base': r'$\alpha$ [%]',
            'jnorm': r'$\Delta U$ [%]',
        }[getattr(self.model, 'name', 'base')])
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

        plt.sca(ax)
        plt.hist(
            self.value[~np.isnan(self.value)],
            bins=bins,
            edgecolor='black', facecolor='white',
            # fill=False,
        )

        plt.xlabel({
            'base': r'$\alpha$ [%]',
            'jnorm': r'$\Delta U$ [%]',
        }[getattr(self.model, 'name', 'base')])
        plt.ylabel(r'count')

        ax.set(**view)


def research_nonlinearity(
    data: Data,
    model: CurrentModelABC | None = None,
    mask: Array[bool] | None = None,
    **kwargs,
) -> NonlinearityResearch:
    model = model or BaseCurrentModel()
    mask = np.full(data.n_numbers, True) if mask is None else mask

    value = np.full(data.n_numbers, np.nan)
    for n, *_ in tqdm(np.argwhere(mask)):
        try:
            result = calculate_nonlinearity(
                trace=data.trace(n),
                model=model,
                verbose=False,
                **kwargs,
            )
            value[n] = result.value
        except FitError as error:
            # LOGGER.error(
            #     'Calculate nonlinearity (n: %d): %s',
            #     n,
            #     error,
            # )
            value[n] = float(np.nan)

    return NonlinearityResearch(
        data=data,
        model=model,
        value=value,
    )


def compare_nonlinearity(
    __traces: Sequence[tuple[Trace, str]],
    model: CurrentModelABC | None = None,
    views: Sequence[AxesView | None] | None = None,
    **kwargs,
) -> None:
    model = model or BaseCurrentModel()
    view_left, view_right = views or [None, None]

    fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for i, (trace, label) in enumerate(__traces):
        color = CMAP(i % 10)

        nonlinearity = calculate_nonlinearity(
            trace=trace,
            model=model,
            verbose=False,
            **kwargs,
        )
        nonlinearity._show_left(
            ax_left,
            view_left,
            color=color,
            verbose=False,
            label=label,
            hat_label=None,
        )
        nonlinearity._show_right(
            ax_right,
            view_right,
            color=color,
            verbose=False,
        )

    filedir = ROOT / 'img'
    filedir.mkdir(parents=True, exist_ok=True)
    filepath = filedir / 'nonlinearity-{method} {n}.png'.format(
        method=model.name,
        n='x'.join([str(trace.n) for trace, _ in __traces]),
    )
    plt.savefig(filepath)

    plt.show()
