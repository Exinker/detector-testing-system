from distfit import distfit
import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum2_wrapper.typing import Array
from vmk_spectrum2_wrapper.units import get_units_clipping

from detector_testing_system.characteristic.bias import calculate_bias
from detector_testing_system.data import Data, to_array
from detector_testing_system.utils import calculate_stats, normalize_values, treat_outliers


def calculate_capacity(data: Data, n: int, threshold: float | None = None, verbose: bool = False, show: bool = False) -> float:
    """Calculate capacity."""
    clipping_value = get_units_clipping(units=data.units)
    threshold = clipping_value if threshold is None else threshold
    u, du, tau = to_array(data=data, n=n)

    # mask
    mask = u < threshold

    #
    p = np.polyfit(u[mask], du[mask], deg=1)
    du_hat = np.polyval(p, u)

    u_bias = calculate_bias(
        data=data, n=n,
        threshold=threshold,
    )

    angle = p[0]
    capacity = np.nan if angle < 0 else clipping_value/angle

    #  verbose
    if verbose:
        print(f'capacity: {capacity:.0f}, e')

    # show
    if show:
        fig, ax = plt.subplots(figsize=(6, 4))

        plt.scatter(
            u, du,
            c='grey', s=10,
        )
        plt.scatter(
            u[mask], du[mask],
            c='red', s=10,
        )
        plt.plot(
            u, du_hat,
            linestyle='solid', c='grey',
        )
        content = [
            r'$U_{{b}}$: {value:.4f} {units}'.format(
                value=u_bias,
                units=data.units_label,
            ),
            r'$c$: {value:.0f} [$e^-$]'.format(
                value=np.round(capacity, 0),
            ),
        ]
        plt.text(
            0.05/2, 0.95,
            '\n'.join(content),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$U$ {units}'.format(units=data.units_label))
        plt.ylabel(r'$\sigma^{2}$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    #
    return capacity


def research_capacity(data: Data, threshold: float | None = None, show: bool = False) -> Array[float]:
    """Research capacity."""
    _, n_numbers = data.mean.shape

    clipping_value = get_units_clipping(units=data.units)
    threshold = threshold or clipping_value

    capacity = np.zeros(n_numbers)
    for n in range(n_numbers):
        capacity[n] = calculate_capacity(
            data=data,
            n=n,
            threshold=threshold,
        )

    # show
    if show:
        mean, ci = calculate_stats(capacity)

        #
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        plt.scatter(
            range(n_numbers), capacity,
            c='black', s=2,
        )
        plt.axhline(
            mean,
            color='red', linestyle='-', linewidth=2,
        )
        plt.xlabel(r'$number$')
        plt.ylabel(r'capacity [$e^-$]')
        plt.grid(color='grey', linestyle=':')
        # plt.legend()

        plt.sca(ax_right)
        content = [
            fr'$\hat{{c}}: {np.round(mean, 0):.0f} \pm {np.round(ci, 0):.0f}$',
        ]
        plt.text(
            0.05/2, 0.95,
            '\n'.join(content),
            transform=ax_right.transAxes,
            ha='left', va='top',
        )
        plt.hist(
            capacity,
            bins=40,
            edgecolor='black', facecolor='white',
            # fill=False,
        )

        plt.xlabel(r'capacity [$e^-$]')
        plt.ylabel(r'freq')
        plt.grid(color='grey', linestyle=':')
        # plt.legend()

        plt.show()

    # show distribution
    if show:
        values = capacity.copy()
        values = treat_outliers(values)
        values = normalize_values(values)

        # show
        dfit = distfit(
            distr='norm',
        )

        result = dfit.fit_transform(
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

    #
    return capacity
