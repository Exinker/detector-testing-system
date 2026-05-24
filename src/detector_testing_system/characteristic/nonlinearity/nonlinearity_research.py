import logging
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tqdm.notebook import tqdm

from vmk_spectrum3_wrapper.types import Array, Number

from detector_testing_system import ROOT
from detector_testing_system.characteristic.dark_current import (
    BaseDarkCurrentModel,
    DarkCurrentModelABC,
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
class NonlinearityResearchResult:

    data: Data
    model: DarkCurrentModelABC
    value: Array[float]

    @property
    def number(self) -> Array[Number]:
        return np.arange(self.data.n_numbers)

    def show(
        self,
        bins: int | Sequence = 40,
        views: Sequence[AxesView | None] | None = None,
        verbose: bool = False,
    ) -> None:
        view_left, view_right = views or [None, None]

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        self._plot_left(ax_left, view_left, verbose=verbose)
        self._plot_right(ax_right, view_right, bins=bins, verbose=verbose)

        plt.show()

    def _plot_left(
        self,
        ax: Axes,
        view: AxesView | None,
        verbose: bool = False,
    ) -> None:
        view = view or {}

        plt.sca(ax)
        if verbose:
            plt.text(
                0.05/2, 0.95,
                '\n'.join([
                    self.data.label.prefix,
                    'method: {method}'.format(
                        method=getattr(self.model, 'name', 'base'),
                    ),
                ]),
                transform=ax.transAxes,
                ha='left', va='top',
            )
        plt.scatter(
            self.number, self.value,
            c='red', s=10,
            label=r'$U$',
        )
        plt.xlabel(r'number')
        plt.ylabel({
            'base': r'$\alpha$ [%]',
            'jnorm': r'$\Delta U$ [%]',
        }[getattr(self.model, 'name', 'base')])
        plt.grid(color='grey', linestyle=':')

        ax.set(**view)

    def _plot_right(
        self,
        ax: Axes,
        view: AxesView | None,
        bins: int,
        verbose: bool = False,
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

        ax.set(**view)


def research_nonlinearity(
    data: Data,
    model: DarkCurrentModelABC | None = None,
    mask: Array[bool] | None = None,
    verbose: bool = False,
    **kwargs,
) -> NonlinearityResearchResult:
    model = model or BaseDarkCurrentModel()
    mask = np.full(data.n_numbers, True) if mask is None else mask

    value = np.full(data.n_numbers, np.nan)
    for n, *_ in tqdm(np.argwhere(mask)):
        try:
            result = calculate_nonlinearity(
                trace=data.trace(n),
                model=model,
                **kwargs,
            )
            value[n] = result.value
        except FitError as error:
            if verbose:
                LOGGER.error(
                    'Calculate nonlinearity (n: %d): %s',
                    n,
                    error,
                )
            value[n] = float(np.nan)

    return NonlinearityResearchResult(
        data=data,
        model=model,
        value=value,
    )


def compare_nonlinearity(
    __traces: Sequence[tuple[Trace, str]],
    model: DarkCurrentModelABC | None = None,
    views: Sequence[AxesView | None] | None = None,
    verbose: bool = False,
    **kwargs,
) -> None:
    model = model or BaseDarkCurrentModel()
    view_left, view_right = views or [None, None]

    fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for i, (trace, label) in enumerate(__traces):
        color = CMAP(i % 10)

        result = calculate_nonlinearity(
            trace=trace,
            model=model,
            **kwargs,
        )
        result._plot_left(
            ax_left,
            view_left,
            color=color,
            verbose=verbose,
            label=label,
            hat_label=None,
        )
        result._plot_right(
            ax_right,
            view_right,
            color=color,
            verbose=verbose,
        )

    filedir = ROOT / 'img'
    filedir.mkdir(parents=True, exist_ok=True)
    filepath = filedir / 'nonlinearity-{method} {n}.png'.format(
        method=model.name,
        n='x'.join([str(trace.n) for trace, _ in __traces]),
    )
    plt.savefig(filepath)

    plt.show()
