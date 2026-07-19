from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array, U

from detector_testing_system.data import Trace, filter_trace_factory
from detector_testing_system.experiment import FitError
from detector_testing_system.types import AxesView


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
        verbose: bool = True,
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
                    r'$U_{{b}}$: {bias:.4f} [{units}]'.format(
                        bias=self.value,
                        units=self.trace.units.label,
                    ),
                ]),
                transform=ax.transAxes,
                ha='right', va='bottom',
            )

        plt.xlabel(r'$\tau$ [{units}]'.format(units=r'$ms$'))
        plt.ylabel(r'$U$ [{units}]'.format(units=self.trace.units.label))

        plt.grid(color='grey', linestyle=':')
        plt.legend()

        ax.set(**view)

        plt.show()


def calculate_bias(
    trace: Trace,
    filter: Callable[[Trace], Array[bool]] | None = None,
) -> Bias:
    """Calculate a bias of the cell."""
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
