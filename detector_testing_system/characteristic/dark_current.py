import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.typing import Array

from detector_testing_system.data import Data
from detector_testing_system.data.exceptions import EmptyArrayError
from detector_testing_system.output import Output
from detector_testing_system.utils import calculate_stats


def calculate_dark_current(
    output: Output,
    threshold: float,
    show: bool = False,
) -> float:
    """Calculate a dark current of the cell."""

    mask = output.average < threshold
    p = np.polyfit(output.exposure[mask], output.average[mask], deg=1)
    current = 1e+3*p[0]  # in %/s

    u_hat = np.polyval(p, output.exposure)

    if show:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.scatter(
            output.exposure, output.average,
            c='grey', s=10,
        )
        plt.scatter(
            output.exposure[mask], output.average[mask],
            c='red', s=10,
        )
        plt.plot(
            output.exposure, u_hat,
            color='black', linestyle='-', linewidth=1,
        )
        plt.text(
            0.05/2, 0.95,
            '\n'.join([
                r'$n$: {n:.0f}'.format(
                    n=output.n,
                ),
                r'$i$: {value:.4f} {units}'.format(
                    value=current,
                    units=f'[{output.units.label}/s]',
                ),
            ]),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$\tau$ {units}'.format(units=r'[$ms$]'))
        plt.ylabel(r'$U$ {units}'.format(units=f'[{output.units.label}/s]'))
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
                output=Output.create(data=data, n=n),
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
            units=f'[{data.units.label}/s]',
        ))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return current
