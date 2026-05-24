import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system import ROOT
from detector_testing_system.characteristic.dark_current.models import (
    BaseDarkCurrentModel,
    DarkCurrentModelABC,
)
from detector_testing_system.characteristic.nonlinearity.calculators import calculate_nonlinearity
from detector_testing_system.characteristic.nonlinearity.results import NonlinearityResearchResult
from detector_testing_system.data import Data, Trace
from detector_testing_system.experiment import FitError
from detector_testing_system.types import AxesView

LOGGER = logging.getLogger(__name__)
CMAP = plt.get_cmap('tab10')


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
    for n, *_ in np.argwhere(mask):
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
    traces: Sequence[Trace],
    model: DarkCurrentModelABC | None = None,
    views: Sequence[AxesView | None] | None = None,
    verbose: bool = False,
    **kwargs,
) -> None:
    model = model or BaseDarkCurrentModel()
    view_left, view_right = views or [None, None]

    fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for i, trace in enumerate(traces):
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
        n='x'.join([str(trace.n) for trace in traces]),
    )
    plt.savefig(filepath)

    plt.show()
