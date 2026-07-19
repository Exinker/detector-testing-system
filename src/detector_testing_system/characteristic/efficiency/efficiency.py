from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system import ROOT
from detector_testing_system.characteristic.bias import (
    Bias,
    calculate_bias,
)
from detector_testing_system.data import Trace, filter_trace_factory
from detector_testing_system.experiment import FitError
from detector_testing_system.types import AxesView

DEGREE = 1


@dataclass
class Efficiency:

    trace: Trace
    mask: Array[bool]
    value: float
    bias: Bias

    def show(
        self,
        view: AxesView | None = None,
        verbose: bool = True,
        note: str = '',
    ) -> None:
        view = view or {}

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        trace = self.trace
        mask = self.mask

        p = np.polyfit(trace.u[mask], trace.variance[mask], deg=DEGREE)
        variance_hat = np.polyval(p, trace.u)

        plt.scatter(
            trace.u, trace.variance,
            c='grey', s=10,
        )
        plt.scatter(
            trace.u[mask], trace.variance[mask],
            c='red', s=10,
        )
        plt.plot(
            trace.u, variance_hat,
            color='black', linestyle='solid', linewidth=1,
        )
        if verbose:
            plt.text(
                0.025, 0.975,
                '\n'.join([
                    trace.label.prefix,
                    r'$k$: {:.0f} [$e^-/\%$]'.format(np.round(self.value, 0)),
                    # r'$c$: {:.0f} [$e^-$]'.format(np.round(self.value, 0) * trace.units.value_max),
                    r'$U_{{b}}$: {bias:.4f} [{units}]'.format(
                        bias=self.bias.value,
                        units=trace.units.label,
                    ),
                    r'n: {}'.format(trace.n),
                    note,
                ]),
                transform=ax.transAxes,
                ha='left', va='top',
            )
        plt.xlabel(r'$U$ [{units}]'.format(units=trace.units.label))
        plt.ylabel(r'$\sigma^{2}$ ' + r'$[\%^{2}]$')
        plt.grid(color='grey', linestyle=':')

        ax.set(**view)

        filedir = ROOT / 'img' / self.trace.label
        filedir.mkdir(parents=True, exist_ok=True)
        filename = 'efficiency ({n}).png'.format(
            n=self.trace.n,
        )
        plt.savefig(filedir / filename)

        plt.show()


def calculate_efficiency(
    trace: Trace,
    filter: Callable[[Trace], Array[bool]] | None = None,
) -> Efficiency:
    filter = filter or filter_trace_factory()

    mask = filter(trace)
    if len(np.argwhere(mask)) < DEGREE + 1:
        raise FitError(
            message=f'Data don\'t enough to be fitted! Efficiency calculation was failed in cell {trace.n}.',
        )

    bias = calculate_bias(
        trace=trace,
        filter=filter,
    )

    p = np.polyfit(trace.u[mask], trace.variance[mask], deg=DEGREE)

    angle = float(p[0])
    if angle < 0:
        value = np.nan
    else:
        value = 1/angle

    return Efficiency(
        trace=trace,
        mask=mask,
        value=value,
        bias=bias,
    )
