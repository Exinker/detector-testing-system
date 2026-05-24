from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array, U

from detector_testing_system.data import Data, Trace, filter_trace_factory
from detector_testing_system.experiment import FitError
from detector_testing_system.types import AxesView
from detector_testing_system.utils import calculate_stats


DEGREE = 1


@dataclass
class Bias:

    trace: Trace
    mask: Array[bool]
    p: tuple[float, float]

    @property
    def value(self) -> U:
        return self.p[1]

    def show(
        self,
        view: AxesView | None = None,
        color: str | None = None,
        verbose: bool = False,
    ) -> None:
        view = view or {}
        color = color or 'red'

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        u_hat = np.polyval(self.p, self.trace.tau)

        plt.scatter(
            self.trace.tau, self.trace.u,
            c='grey', s=10,
        )
        plt.scatter(
            self.trace.tau[self.mask], self.trace.u[self.mask],
            c=color, s=10,
            label=rf'$U_{{{self.trace.n}}}$',
        )
        plt.plot(
            self.trace.tau, u_hat,
            color='black', linestyle='-', linewidth=1,
        )
        plt.scatter(
            0, self.value,
            s=40,
            marker='*', facecolors='none', edgecolors='red',
        )
        if verbose:
            plt.text(
                0.95, 0.05/2,
                '\n'.join([
                    r'$U_{{b}}$: {bias:.4f} {units}'.format(
                        bias=self.value,
                        units=self.trace.units.label,
                    ),
                ]),
                transform=ax.transAxes,
                ha='right', va='bottom',
            )

        plt.xlabel(r'$\tau$ {units}'.format(units=r'[$ms$]'))
        plt.ylabel(r'$U$ {units}'.format(units=self.trace.units.label))

        plt.grid(color='grey', linestyle=':')
        plt.legend()

        ax.set(**view)

        plt.show()


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


def calculate_bias(
    trace: Trace,
    filter: Callable[[Trace], Array[bool]] | None = None,
) -> Bias:
    """Calculate a bias of the cell"""
    filter = filter or filter_trace_factory()

    mask = filter(trace)
    if len(np.argwhere(mask)) < DEGREE + 1:
        raise FitError(
            message=f'Data don\'t enough to be fitted! Bias calculation was failed in cell {trace.n}.',
        )

    p = np.polyfit(trace.tau[mask], trace.u[mask], deg=DEGREE)

    return Bias(
        trace=trace,
        mask=mask,
        p=p,
    )


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
