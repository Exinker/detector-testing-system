import numpy as np
import matplotlib.pyplot as plt
from distfit import distfit

from libspectrum2_wrapper.alias import Array
from libspectrum2_wrapper.units import get_units_clipping, to_electron

from core.data import Data, to_array

from .bias import calculate_bias
from .utils import calculate_stats, normalize_values, treat_outliers


def calculate_read_noise(data: Data, n: int, threshold: float | None = None, verbose: bool = False, show: bool = False) -> float:
    """Calculate noise read."""
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

    read_noise = np.sqrt(np.polyval(p, u_bias))

    # show
    if show:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

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
            r'$\sigma_{{rd}}$: {value:.4f} {units}'.format(
                value=to_electron(
                    read_noise,
                    units=data.units,
                    capacity=capacity,
                ),
                units=r'$[e^{-}]$',
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
    return read_noise


def research_read_noise(data: Data, threshold: float | None = None, show: bool = False) -> Array[float]:
    """Research read_noise."""
    _, n_numbers = data.mean.shape

    clipping_value = get_units_clipping(units=data.units)
    threshold = threshold or clipping_value

    read_noise = np.zeros(n_numbers,)
    for n in range(n_numbers):
        read_noise[n] = calculate_read_noise(
            data=data,
            n=n,
            threshold=threshold,
        )

    # show
    if show:
        mean, ci = calculate_stats(read_noise)

        #
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        plt.scatter(
            range(n_numbers), read_noise,
            c='black', s=2,
        )
        plt.axhline(
            mean,
            color='red', linestyle='-', linewidth=2,
        )
        plt.xlabel(r'$number$')
        plt.ylabel(r'read-noise {units}'.format(units=data.units_label))
        plt.grid(color='grey', linestyle=':')
        # plt.legend()

        plt.sca(ax_right)
        content = [
            fr'$\hat{{c}}: {mean:.4f} \pm {ci:.4f}$',
        ]
        plt.text(
            0.05/2, 0.95,
            '\n'.join(content),
            transform=ax_right.transAxes,
            ha='left', va='top',
        )
        plt.hist(
            read_noise,
            bins=40,
            edgecolor='black', facecolor='white',
            # fill=False,
        )

        plt.xlabel(r'read-noise {units}'.format(units=data.units_label))
        plt.ylabel(r'freq')
        plt.grid(color='grey', linestyle=':')
        # plt.legend()

        plt.show()

    # show distribution
    if show:
        values = read_noise.copy()
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
    return read_noise


def compare_noise(read_noise: Array[float], data: Data) -> None:
    """"""
    _, n_numbers = data.mean.shape

    noise = np.mean(np.sqrt(data.variance), axis=0)

    #
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    # plt.scatter(
    #     range(n_numbers), read_noise,
    #     c='black', s=2,
    # )
    plt.axhline(
        np.mean(read_noise),
        color='black', linestyle=':',
    )
    plt.scatter(
        range(n_numbers), noise,
        c='red', s=2,
    )
    plt.xlabel(r'$number$')
    plt.ylabel(r'noise-read {units}'.format(units=data.units_label))
    plt.grid(color='grey', linestyle=':')

    plt.show()
