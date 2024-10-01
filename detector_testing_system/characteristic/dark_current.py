import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.typing import Array

from detector_testing_system.data import Data
from detector_testing_system.data.exceptions import EmptyArrayError
from detector_testing_system.signal import Signal
from detector_testing_system.utils import calculate_stats


def calculate_dark_current(
    signal: Signal,
    threshold: float,
    show: bool = False,
) -> float:
    """Calculate a dark current of the cell."""

    mask = signal.value < threshold
    p = np.polyfit(signal.exposure[mask], signal.value[mask], deg=1)
    current = 1e+3*p[0]  # in %/s

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
        plt.text(
            0.05/2, 0.95,
            '\n'.join([
                r'$n$: {n:.0f}'.format(
                    n=signal.n,
                ),
                r'$i$: {value:.4f} {units}'.format(
                    value=current,
                    units=f'[{signal.units.get_label(is_enclosed=False)}/s]',
                ),
            ]),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$\tau$ {units}'.format(units=r'[$ms$]'))
        plt.ylabel(r'$U$ {units}'.format(units=f'[{signal.units.get_label(is_enclosed=False)}/s]'))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return current


def research_dark_current(
    data: Data,
    threshold: float | None = None,
    confidence: float = .95,
    verbose: bool = False,
    show: bool = False,
) -> Array[float]:
    """Calculate a dark current of the cells."""
    threshold = threshold or data.units.value_max

    current = np.zeros(data.n_numbers)
    for n in range(data.n_numbers):
        try:
            value = calculate_dark_current(
                signal=Signal.create(data=data, n=n),
                threshold=threshold,
            )

        except EmptyArrayError as error:
            value = float(np.nan)

            if verbose:
                print(error)

        finally:
            current[n] = value

    if show:
        mean, ci = calculate_stats(current, confidence=confidence)

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.scatter(
            range(data.n_numbers), current,
            c='black', s=2,
        )
        plt.text(
            0.05/2, 0.95,
            '\n'.join([
                r'$i_{{d}}$: {mean:.4f} $\pm$ {ci:.4f}'.format(
                    mean=mean,
                    ci=ci,
                ),
            ]),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$number$')
        plt.ylabel(r'$i_{{d}}$ {units}'.format(
            units=f'[{data.units.get_label(is_enclosed=False)}/s]',
        ))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return current
