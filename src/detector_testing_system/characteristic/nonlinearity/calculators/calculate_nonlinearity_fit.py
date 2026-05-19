import os
import reprlib

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array, MilliSecond

from detector_testing_system.experiment.utils import create_directory
from detector_testing_system.output import Output


def calculate_nonlinearity_fit(
    output: Output,
    span: tuple[MilliSecond, MilliSecond] = None,
    show: bool = False,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
) -> tuple[Array[float], float]:
    span = span or (min(output.exposure), max(output.exposure))

    mask = (output.exposure >= span[0]) & (output.exposure <= span[1])
    p = _optimize(output.exposure[mask], output.average[mask])
    u_hat = np.polyval(p, output.exposure)

    xi = _calculate_xi(output.exposure, output.average, p)
    alpha = _calculate_alpha(output.exposure[mask], output.average[mask], p)

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
                fr'$a = {{{p[0]:.4f}}}$',
                fr'$b = {{{p[1]:.4f}}}$',
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
            output.average, xi,
            c='grey', s=10,
        )
        plt.scatter(
            output.average[mask], xi[mask],
            c='red', s=10,
            label=r'$U$',
        )
        ax_right.text(
            0.95, 0.95,
            '\n'.join([
                fr'{str(reprlib.repr(output.label))}',
                fr'n: {output.n}',
                fr'$\alpha: {{{alpha:.2f}}}$ [%]',
            ]),
            transform=ax_right.transAxes,
            ha='right', va='top',
        )
        ax_right.text(
            0.95, 0.05/2,
            '\n'.join([
                r'$error = 100\frac{\hat{U} - U_{i}}{a \tau}$',
            ]),
            transform=plt.gca().transAxes,
            ha='right', va='bottom',
        )
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(r'$U$ {units}'.format(units=output.units.label))
        plt.ylabel(r'$error$ [%]')
        plt.grid(color='grey', linestyle=':')

        filedir = create_directory(os.path.join('.', 'img'), label=output.label)
        filepath = os.path.join(filedir, f'nonlinearity-fit ({output.n}).png')
        plt.savefig(filepath)

        plt.show()

    return xi, alpha


def _calculate_xi(tau: Array[float], u: Array[float], p: Array[float]) -> Array[float]:
    """Calculate a residual of approximation."""
    u_hat = np.polyval(p, tau)

    # xi = 100*(u_hat - u) / u
    xi = 100*(u_hat - u) / np.polyval([p[0], 0], tau)

    return xi


def _calculate_alpha(tau: Array[float], u: Array[float], p: Array[float]) -> float:
    """Calculate nonlinearity coefficient."""
    xi = _calculate_xi(tau, u, p)

    return (np.max(xi) - np.min(xi)) / 2


def _optimize(tau: Array[float], u: Array[float]) -> Array[float]:

    alpha = np.sum(u / tau)
    alpha2 = np.sum(u / tau**2)

    beta = np.sum(1 / tau)
    beta2 = np.sum(1 / tau**2)

    b = (alpha*alpha2 - beta*np.sum(u**2 / tau**2)) / (alpha*beta2 - beta*alpha2)
    a = (alpha2 - b * beta2) / beta

    return np.array([a, b])
