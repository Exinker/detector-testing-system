import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.experiment import Data, EmptyArrayError
from detector_testing_system.trace import Trace
from detector_testing_system.utils import calculate_stats


DEGREE = 1


def calculate_bias(
    trace: Trace,
    threshold: tuple[float, float],
    show: bool = False,
) -> float:
    """Calculate a bias of the cell"""

    lb, ub = threshold
    mask = (lb < trace.u) & (trace.u < ub)
    if len(np.argwhere(mask)) < DEGREE + 1:
        raise EmptyArrayError(
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
                r'$n$: {n:.0f}'.format(
                    n=trace.n,
                ),
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
    threshold: tuple[float, float] | None = None,
    confidence: float = .95,
    verbose: bool = False,
    show: bool = False,
) -> Array[float]:
    """Calculate a bias of the cells"""
    threshold = threshold or (0, data.units.value_max)

    bias = np.zeros(data.n_numbers)
    for n in range(data.n_numbers):
        try:
            value = calculate_bias(
                trace=Trace.create(data=data, n=n),
                threshold=threshold,
            )
        except EmptyArrayError as error:
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
