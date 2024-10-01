import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.typing import Array

from detector_testing_system.data import Data
from detector_testing_system.data.exceptions import EmptyArrayError
from detector_testing_system.signal import Signal
from detector_testing_system.utils import calculate_stats


def calculate_bias(
    signal: Signal,
    threshold: float,
    show: bool = False,
) -> float:
    """Calculate a bias of the cell."""

    mask = signal.value < threshold
    p = np.polyfit(signal.exposure[mask], signal.value[mask], deg=1)
    bias = p[1]

    u_hat = np.polyval(p, signal.exposure)

    if show:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.scatter(
            signal.exposure, signal.value,
            c='grey', s=10,
        )
        plt.scatter(
            signal.exposure[mask], signal.value[mask],
            c='red', s=10,
        )
        plt.plot(
            signal.exposure, u_hat,
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
                    n=signal.n,
                ),
                r'$U_{{b}}$: {value:.4f} {units}'.format(
                    value=bias,
                    units=signal.units.label,
                ),
            ]),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$\tau$ {units}'.format(units=r'[$ms$]'))
        plt.ylabel(r'$U$ {units}'.format(units=signal.units.label))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return bias


def research_bias(
    data: Data,
    threshold: float | None = None,
    confidence: float = .95,
    verbose: bool = False,
    show: bool = False,
) -> Array[float]:
    """Calculate a bias of the cells."""
    threshold = threshold or data.units.value_max

    bias = np.zeros(data.n_numbers)
    for n in range(data.n_numbers):
        try:
            value = calculate_bias(
                signal=Signal.create(data=data, n=n),
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
