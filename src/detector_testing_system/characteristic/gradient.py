import reprlib

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system import ROOT
from detector_testing_system.data import Trace
from detector_testing_system.experiment.utils import create_directory


def calculate_gradient(
    trace: Trace,
    show: bool = False,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
) -> Array[float]:
    """Calculate gradient"""

    u_grad = np.gradient(trace.u, trace.tau)

    if show:
        p = np.polyfit(trace.tau, trace.u, deg=1)

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        plt.scatter(
            trace.tau, trace.u,
            c='red', s=10,
            label=r'$U$',
        )
        plt.plot(
            trace.tau, np.polyval(p, trace.tau),
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
        plt.ylabel(r'$U$ {units}'.format(units=trace.units.label))
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.sca(ax_right)
        ax_right.text(
            0.95, 0.95,
            '\n'.join([
                reprlib.repr(trace.label),
                fr'n: {trace.n}',
            ]),
            transform=ax_right.transAxes,
            ha='right', va='top',
        )
        plt.scatter(
            trace.u, u_grad,
            c='red', s=10,
        )
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(r'$U$ {units}'.format(units=trace.units.label))
        plt.ylabel(r'$dU / d\tau$')
        plt.grid(color='grey', linestyle=':')

        filedir = create_directory(ROOT / 'img', label=trace.label)
        filepath = filedir / f'gradient ({trace.n}).png'
        plt.savefig(filepath)

        plt.show()

    return u_grad
