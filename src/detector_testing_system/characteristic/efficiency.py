"""Calculate an efficiency (factor to convert percents to electrons)."""

from collections.abc import Sequence

from distfit import distfit
import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.characteristic.bias import calculate_bias
from detector_testing_system.experiment import Data, EmptyArrayError
from detector_testing_system.output import Output
from detector_testing_system.utils import (
    calculate_stats,
    normalize_values,
    treat_outliers,
)


DEGREE = 1


def calculate_efficiency(
    output: Output,
    threshold: float,
    verbose: bool = False,
    show: bool = False,
) -> float:

    mask = output.average < threshold
    if len(np.argwhere(mask)) < DEGREE + 1:
        raise EmptyArrayError(
            message=f'Data don\'t enough to be fitted! Efficiency calculation was failed in cell {output.n}.',
        )

    p = np.polyfit(output.average[mask], output.variance[mask], deg=DEGREE)
    variance_hat = np.polyval(p, output.average)

    angle = p[0]
    if angle < 0:
        efficiency = np.nan
    else:
        efficiency = 1/angle

    if verbose:
        print(f'efficiency: {efficiency:.0f}, e\%')

    if show:
        fig, ax = plt.subplots(figsize=(6, 4))

        bias = calculate_bias(
            output=output,
            threshold=threshold,
        )

        plt.scatter(
            output.average, output.variance,
            c='grey', s=10,
        )
        plt.scatter(
            output.average[mask], output.variance[mask],
            c='red', s=10,
        )
        plt.plot(
            output.average, variance_hat,
            linestyle='solid', c='grey',
        )
        plt.text(
            0.05/2, 0.95,
            '\n'.join([
                r'$U_{{b}}$: {value:.4f} {units}'.format(
                    value=bias,
                    units=output.units.label,
                ),
                r'$k$: {value:.0f} [$e^-/\%$]'.format(
                    value=np.round(efficiency, 0),
                ),
                r'$c$: {value:.0f} [$e^-$]'.format(
                    value=np.round(efficiency, 0) * output.units.value_max,
                ),
            ]),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$U$ {units}'.format(units=output.units.label))
        plt.ylabel(r'$\sigma^{2}$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return efficiency


def research_efficiency(
    data: Data,
    threshold: float | None = None,
    verbose: bool = False,
    show: bool = False,
    bins: int | Sequence = 40,
) -> Array[float]:
    threshold = threshold or data.units.value_max

    efficiency = np.zeros(data.n_numbers)
    for n in range(data.n_numbers):
        try:
            value = calculate_efficiency(
                output=Output.create(data=data, n=n),
                threshold=threshold,
            )

        except EmptyArrayError as error:
            value = float(np.nan)

            if verbose:
                print(error)

        finally:
            efficiency[n] = value

    if show:
        mean, ci = calculate_stats(efficiency)

        #
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        plt.scatter(
            range(data.n_numbers), efficiency,
            c='black', s=2,
        )
        plt.axhline(
            mean,
            color='red', linestyle='-', linewidth=2,
        )
        plt.xlabel(r'$number$')
        plt.ylabel(r'k [$e^{-}/\%$]')
        plt.grid(color='grey', linestyle=':')
        # plt.legend()

        plt.sca(ax_right)
        plt.text(
            0.05/2, 0.95,
            '\n'.join([
                fr'$k: {np.round(mean, 0):.0f} \pm {np.round(ci, 0):.0f}$',
            ]),
            transform=ax_right.transAxes,
            ha='left', va='top',
        )
        plt.hist(
            efficiency,
            bins=bins,
            edgecolor='black', facecolor='white',
            # fill=False,
        )

        plt.xlabel(r'k [$e^{-}/\%$]')
        plt.ylabel(r'freq')
        plt.grid(color='grey', linestyle=':')
        # plt.legend()

        plt.show()

    if show and False:  # deprecated functionality
        values = efficiency.copy()
        values = treat_outliers(values)
        values = normalize_values(values)

        # show
        dfit = distfit(
            distr='norm',
        )

        dfit.fit_transform(
            values,
            verbose=False,
        )

        #
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

        dfit.plot(
            chart='pdf',
            ax=ax_left,
        )
        dfit.qqplot(
            values,
            ax=ax_right,
        )

    return efficiency
