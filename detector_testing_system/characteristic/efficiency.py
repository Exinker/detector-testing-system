"""Calculate an efficiency (factor to convert percents to electrons)."""

from distfit import distfit
import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.typing import Array
# from vmk_spectrum3_wrapper.units import get_units_clipping

from detector_testing_system.characteristic.bias import calculate_bias
from detector_testing_system.data import Data
from detector_testing_system.data.exceptions import EmptyArrayError
from detector_testing_system.signal import Signal
from detector_testing_system.utils import calculate_stats, normalize_values, treat_outliers


def calculate_efficiency(
    data: Data,
    n: int,
    threshold: float | None = None,
    verbose: bool = False,
    show: bool = False,
) -> float:
    threshold = threshold or get_units_clipping(units=data.units)

    try:
        signal = Signal.create(data=data, n=n)

    except EmptyArrayError as error:
        if verbose:
            print(error)

        return float(np.nan)

    else:
        mask = signal.value < threshold

        p = np.polyfit(signal.value[mask], signal.variance[mask], deg=1)
        variance_hat = np.polyval(p, signal.value)

        bias = calculate_bias(
            data=data,
            n=n,
            threshold=threshold,
        )

        angle = p[0]
        if angle < 0:
            efficiency = np.nan
        else:
            efficiency = 1/angle

    if verbose:
        print(f'efficiency: {efficiency:.0f}, e/%')

    if show:
        fig, ax = plt.subplots(figsize=(6, 4))

        plt.scatter(
            signal.value, signal.variance,
            c='grey', s=10,
        )
        plt.scatter(
            signal.value[mask], signal.variance[mask],
            c='red', s=10,
        )
        plt.plot(
            signal.value, variance_hat,
            linestyle='solid', c='grey',
        )
        plt.text(
            0.05/2, 0.95,
            '\n'.join([
                r'$U_{{b}}$: {value:.4f} {units}'.format(
                    value=bias,
                    units=data.units_label,
                ),
                r'$k$: {value:.0f} [$e^-/%$]'.format(
                    value=np.round(efficiency, 0),
                ),
                r'$c$: {value:.0f} [$e^-$]'.format(
                    value=np.round(efficiency, 0) * get_units_clipping(units=data.units),
                ),
            ]),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$U$ {units}'.format(units=data.units_label))
        plt.ylabel(r'$\sigma^{2}$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return efficiency


def research_efficiency(
    data: Data,
    threshold: float | None = None,
    show: bool = False,
) -> Array[float]:
    threshold = threshold or get_units_clipping(units=data.units)

    efficiency = np.zeros(data.n_numbers)
    for n in range(data.n_numbers):
        efficiency[n] = calculate_efficiency(
            data=data,
            n=n,
            threshold=threshold,
        )

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
            bins=40,
            edgecolor='black', facecolor='white',
            # fill=False,
        )

        plt.xlabel(r'k [$e^{-}/\%$]')
        plt.ylabel(r'freq')
        plt.grid(color='grey', linestyle=':')
        # plt.legend()

        plt.show()

    if show:
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
