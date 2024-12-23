import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.experiment import Data, EmptyArrayError
from detector_testing_system.output import Output
from detector_testing_system.utils import calculate_stats


DEGREE = 1


def calculate_bias(
    output: Output,
    threshold: float,
    show: bool = False,
) -> float:
    """Calculate a bias of the cell."""

    mask = output.average < threshold
    if len(np.argwhere(mask)) < DEGREE + 1:
        raise EmptyArrayError(
            message=f'Data don\'t enough to be fitted! Bias calculation was failed in cell {output.n}.',
        )

    p = np.polyfit(output.exposure[mask], output.average[mask], deg=DEGREE)
    bias = p[1]

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
        plt.scatter(
            0, bias,
            s=40,
            marker='*', facecolors='none', edgecolors='red',
        )
        plt.text(
            0.05/2, 0.95,
            '\n'.join([
                r'$n$: {n:.0f}'.format(
                    n=output.n,
                ),
                r'$U_{{b}}$: {value:.4f} {units}'.format(
                    value=bias,
                    units=output.units.label,
                ),
            ]),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$\tau$ {units}'.format(units=r'[$ms$]'))
        plt.ylabel(r'$U$ {units}'.format(units=output.units.label))
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
                output=Output.create(data=data, n=n),
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
