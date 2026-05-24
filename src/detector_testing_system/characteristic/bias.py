from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.data import Data, Trace
from detector_testing_system.experiment import FitArrayError
from detector_testing_system.utils import calculate_stats


DEGREE = 1


def calculate_bias(
    trace: Trace,
    filter: Callable[[Trace], Array[bool]],
    show: bool = False,
) -> float:
    """Calculate a bias of the cell"""

    mask = filter(trace)
    if len(np.argwhere(mask)) < DEGREE + 1:
        raise FitArrayError(
            message=f'Data don\'t enough to be fitted! Bias calculation was failed in cell {trace.n}.',
        )

    p = np.polyfit(trace.tau[mask], trace.u[mask], deg=DEGREE)
    bias = p[1]

    u_hat = np.polyval(p, trace.tau)

    if show:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.scatter(
            trace.tau, trace.u,
            c='grey', s=10,
            label=rf'$U_{{{trace.n}}}$',
        )
        plt.scatter(
            trace.tau[mask], trace.u[mask],
            c='red', s=10,
        )
        plt.plot(
            trace.tau, u_hat,
            color='black', linestyle='-', linewidth=1,
        )
        plt.scatter(
            0, bias,
            s=40,
            marker='*', facecolors='none', edgecolors='red',
        )
        plt.text(
            0.05/2, 0.95,
            '\n'.join([
                r'$U_{{b}}$: {bias:.4f} {units}'.format(
                    bias=bias,
                    units=trace.units.label,
                ),
            ]),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$\tau$ {units}'.format(units=r'[$ms$]'))
        plt.ylabel(r'$U$ {units}'.format(units=trace.units.label))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return bias


def research_bias(
    data: Data,
    filter: Callable[[Trace], Array[bool]],
    confidence: float = .95,
    verbose: bool = False,
    show: bool = False,
) -> Array[float]:
    """Calculate a bias of the cells"""

    bias = np.zeros(data.n_numbers)
    for n in range(data.n_numbers):
        try:
            value = calculate_bias(
                trace=data.trace(n),
                filter=filter,
            )
        except FitArrayError as error:
            value = float(np.nan)

            if verbose:
                print(error)
        finally:
            bias[n] = value

    if show:
        mean, ci = calculate_stats(bias, confidence=confidence)

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.scatter(
            range(data.n_numbers), bias,
            c='black', s=2,
        )
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
        plt.ylabel(r'$U_{{b}}$ {units}'.format(units=data.units.label))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return bias
