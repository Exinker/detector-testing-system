import reprlib

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array, U

from detector_testing_system import ROOT
from detector_testing_system.characteristic.dark_current.models import BaseDarkCurrentModel
from detector_testing_system.data import Trace
from detector_testing_system.experiment.utils import create_directory


def calculate_nonlinearity_base(
    trace: Trace,
    model: BaseDarkCurrentModel,
    show: bool = False,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
) -> tuple[Array[float], float]:

    if not model.weighted:
        raise ValueError('To calculate nonlinearity use weighted model only!')

    result = model.fit(trace)

    alpha = _calculate_alpha(xi=result.xi)

    if show:
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        plt.scatter(
            trace.tau, trace.u,
            c='grey', s=10,
        )
        plt.scatter(
            trace.tau[result.mask], trace.u[result.mask],
            c='red', s=10,
            label=r'$U$',
        )
        plt.plot(
            trace.tau, result.interpolate(trace.tau),
            color='black', linestyle='solid', linewidth=1,
            label=r'$\hat{U}$',
        )
        ax_left.text(
            0.95, 0.05/2,
            '\n'.join([
                fr'$a = {{{result.p[0]:.4f}}}$',
                fr'$b = {{{result.p[1]:.4f}}}$',
            ]),
            transform=plt.gca().transAxes,
            ha='right', va='bottom',
        )
        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ {units}'.format(units=trace.units.label))
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.sca(ax_right)
        plt.scatter(
            trace.u, result.xi,
            c='grey', s=10,
        )
        plt.scatter(
            trace.u[result.mask], result.xi[result.mask],
            c='red', s=10,
            label=r'$U$',
        )
        ax_right.text(
            0.95, 0.95,
            '\n'.join([
                fr'{str(reprlib.repr(trace.label))}',
                fr'n: {trace.n}',
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
        plt.xlabel(r'$U$ {units}'.format(units=trace.units.label))
        plt.ylabel(r'$error$ [%]')
        plt.grid(color='grey', linestyle=':')

        filedir = create_directory(ROOT / 'img', label=trace.label)
        filepath = filedir / f'nonlinearity-fit ({trace.n}).png'
        plt.savefig(filepath)

        plt.show()

    return result.xi, alpha


def _calculate_alpha(xi: Array[U]) -> float:
    """Calculate nonlinearity coefficient (alpha)"""
    return (np.max(xi) - np.min(xi)) / 2
