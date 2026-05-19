import os
import reprlib

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.characteristic.gradient import calculate_gradient
from detector_testing_system.experiment.utils import create_directory
from detector_testing_system.output import Output


def calculate_nonlinearity_jnorm(
    output: Output,
    epsilon: float = .10,
    min_points: int = 10,
    k: float = 2,
    show: bool = False,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
) -> tuple[Array[float], float]:

    u_grad = calculate_gradient(output=output)
    mask = _select_mask(
        u_grad=u_grad,
        epsilon=epsilon,
        min_points=min_points,
    )

    if not np.any(mask):
        xi = np.full(len(output.average), np.nan)
        span = float(np.nan)
        a = float(np.nan)
        b = float(np.nan)
        u_hat = xi

    else:
        a = float(np.mean(np.asarray(u_grad)[mask]))
        b = float(np.mean(output.average[mask] - a * output.exposure[mask]))
        u_hat = a * output.exposure + b
        xi = _calculate_xi(
            tau=output.exposure,
            u=output.average,
            u_hat=u_hat,
            jnorm=a,
        )
        x_intersection = _calculate_intersection(
            u=output.average,
            u_grad=u_grad,
            threshold=k * a,
        )
        if x_intersection is not None:
            span = x_intersection - output.average[0]
        else:
            span = 0

    if show:
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        plt.scatter(
            output.exposure, output.average,
            c='grey', s=10,
        )
        plt.scatter(
            output.exposure[mask], output.average[mask],
            c='red', s=10,
            label=r'$U$',
        )
        plt.plot(
            output.exposure, u_hat,
            color='black', linestyle='solid', linewidth=1,
            label=r'$\hat{U}$',
        )
        ax_left.text(
            0.95, 0.05/2,
            '\n'.join([
                fr'$a = {{{a:.4f}}}$',
                fr'$b = {{{b:.4f}}}$',
            ]),
            transform=plt.gca().transAxes,
            ha='right', va='bottom',
        )
        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ {units}'.format(units=output.units.label))
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.sca(ax_right)
        plt.scatter(
            output.average, u_grad,
            c='grey', s=10,
        )
        plt.scatter(
            output.average[mask], u_grad[mask],
            c='red', s=10,
            label=r'$U$',
        )
        plt.axhline(
            a,
            color='black', linestyle='solid', linewidth=1,
        )
        plt.axhline(
            k * a,
            color='red', linestyle='--', linewidth=1,
        )
        ax_right.axvspan(
            output.average[0],
            x_intersection,
            color='grey',
            alpha=.125,
        )
        ax_right.text(
            0.95, 0.95,
            '\n'.join([
                fr'{str(reprlib.repr(output.label))}',
                fr'n: {output.n}',
                fr'$\Delta U$: {span:.2f} [%]',
            ]),
            transform=ax_right.transAxes,
            ha='right', va='top',
        )
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(r'$U$ {units}'.format(units=output.units.label))
        plt.ylabel(r'$dU / d\tau$')
        plt.grid(color='grey', linestyle=':')

        filedir = create_directory(os.path.join('.', 'img'), label=output.label)
        filepath = os.path.join(filedir, f'nonlinearity-jnorm ({output.n}).png')
        plt.savefig(filepath)

        plt.show()

    return xi, span


def _select_mask(
    u_grad: Array[float],
    epsilon: float,
    min_points: int,
) -> Array[bool]:
    n_points = len(u_grad)

    mask = np.full(n_points, False)
    for n in range(n_points - min_points + 1):

        if _relative_std(u_grad[n:]) <= epsilon:
            mask[n:] = True
            return mask

    return mask


def _calculate_xi(
    tau: Array[float],
    u: Array[float],
    u_hat: Array[float],
    jnorm: float,
) -> Array[float]:
    return 100 * (u_hat - u) / (jnorm * tau)


def _calculate_intersection(
    u: Array[float],
    u_grad: Array[float],
    threshold: float,
) -> float | None:

    diff = u_grad - threshold
    exact_indexes = np.argwhere(diff == 0).ravel()
    if len(exact_indexes) > 0:
        return float(u[exact_indexes[-1]])

    crossing_indexes = np.argwhere(diff[:-1] * diff[1:] < 0).ravel()
    if len(crossing_indexes) == 0:
        return None

    i = int(crossing_indexes[-1])
    x0, x1 = u[i], u[i + 1]
    y0, y1 = u_grad[i], u_grad[i + 1]
    if y1 == y0:
        x_intersection = float(x0)
    else:
        x_intersection = float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))

    return x_intersection


def _relative_std(values: Array[float]) -> float:

    mean = float(np.mean(values))
    if mean == 0 or not np.isfinite(mean):
        return float(np.inf)

    return float(np.std(values) / abs(mean))
