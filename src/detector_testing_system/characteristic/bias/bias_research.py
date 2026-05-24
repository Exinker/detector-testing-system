from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array, U

from detector_testing_system.characteristic.bias.bias import calculate_bias
from detector_testing_system.data import Data, Trace
from detector_testing_system.experiment import FitError
from detector_testing_system.types import AxesView
from detector_testing_system.utils import calculate_stats


DEGREE = 1


@dataclass
class BiasResearch:

    data: Data
    value: Array[U]

    def show(
        self,
        confidence: float = .95,
        view: AxesView | None = None,
        color: str | None = None,
        verbose: bool = False,
    ) -> None:
        view = view or {}
        color = color or 'red'

        number = range(self.data.n_numbers)
        mean, ci = calculate_stats(self.value, confidence=confidence)

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.scatter(
            number, self.value,
            c='black', s=2,
        )
        if verbose:
            plt.text(
                0.05/2, 0.95,
                '\n'.join([
                    r'$U_{{b}}$: {mean:.4f} $\pm$ {ci:.4f}'.format(
                        mean=mean,
                        ci=ci,
                    ),
                ]),
                transform=ax.transAxes,
                ha='left', va='top',
            )

        plt.xlabel(r'$number$')
        plt.ylabel(r'$U_{{b}}$ [{units}]'.format(units=self.data.units.label))

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)

        plt.show()


def research_bias(
    data: Data,
    filter: Callable[[Trace], Array[bool]] | None = None,
) -> BiasResearch:
    """Calculate a bias of the cells"""

    value = np.zeros(data.n_numbers)
    for n in range(data.n_numbers):
        try:
            bias = calculate_bias(
                trace=data.trace(n),
                filter=filter,
            )
            value[n] = bias.value
        except FitError:
            value[n] = float(np.nan)

    return BiasResearch(
        data=data,
        value=value,
    )
