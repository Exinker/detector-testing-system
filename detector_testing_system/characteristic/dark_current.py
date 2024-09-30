import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.typing import Array
from vmk_spectrum3_wrapper.units import get_units_clipping, get_units_label

from detector_testing_system.data import Data, to_array
from detector_testing_system.utils import calculate_stats


def calculate_dark_current(data: Data, n: int, threshold: float = np.inf, show: bool = False) -> float:
    """Calculate a dark current of the cell."""
    u, variance, tau = to_array(data=data, n=n)

    # mask
    mask = u < threshold

    #
    p = np.polyfit(tau[mask], u[mask], deg=1)
    current = 1e+3*p[0]  # in %/s

    u_hat = np.polyval(p, tau)

    # show
    if show:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.scatter(
            tau, u,
            c='grey', s=10,
        )
        plt.scatter(
            tau[mask], u[mask],
            c='red', s=10,
        )
        plt.plot(
            tau, u_hat,
            color='black', linestyle='-', linewidth=1,
        )
        content = [
            r'$n$: {n:.0f}'.format(n=n),
            r'$i$: {value:.4f} {units}'.format(
                value=current,
                units=f'[{get_units_label(data.units, is_enclosed=False)}/s]',
            ),
        ]
        plt.text(
            0.05/2, 0.95,
            '\n'.join(content),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$\tau$ {units}'.format(units=r'[$ms$]'))
        plt.ylabel(r'$U$ {units}'.format(units=f'[{get_units_label(data.units, is_enclosed=False)}/s]'))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    #
    return current


def research_dark_current(data: Data, threshold: float | None = None, confidence: float = .95, show: bool = False) -> Array[float]:
    """Calculate a dark current of the cells."""
    _, n_numbers = data.mean.shape

    clipping_value = get_units_clipping(units=data.units)
    threshold = threshold or clipping_value

    current = np.zeros(n_numbers)
    for n in range(n_numbers):
        current[n] = calculate_dark_current(
            data=data, n=n,
            threshold=threshold,
        )

    # show
    if show:
        mean, ci = calculate_stats(current, confidence=confidence)

        #
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.scatter(
            range(n_numbers), current,
            c='black', s=2,
        )
        content = [
            r'$i_{{d}}$: {mean:.4f} $\pm$ {ci:.4f}'.format(
                mean=mean,
                ci=ci,
            ),
        ]
        plt.text(
            0.05/2, 0.95,
            '\n'.join(content),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$number$')
        plt.ylabel(r'$i_{{d}}$ {units}'.format(
            units=f'[{get_units_label(data.units, is_enclosed=False)}/s]',
        ))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    #
    return current
