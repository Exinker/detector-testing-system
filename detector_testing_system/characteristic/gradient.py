import os

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum2_wrapper.typing import Array

from detector_testing_system.data import Data, to_array


def calculate_gradient(data: Data, n: int, show: bool = False, xlim: tuple[float, float] = None, ylim: tuple[float, float] = None) -> Array[float]:
    """Calculate gradient."""
    u, du, tau = to_array(data=data, n=n)
    u_grad = np.gradient(u, tau)

    # show
    if show:
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        plt.scatter(
            tau, u,
            c='red', s=10,
            label=r'$U$',
        )
        plt.plot(
            tau, np.polyval(np.polyfit(tau, u, deg=1), tau),
            color='black', linestyle='solid', linewidth=1,
            label=r'$\hat{U}$',
        )
        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ {units}'.format(units=data.units_label))
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.sca(ax_right)
        content = [
            fr'{data.label}',
            fr'n: {n}',
        ]
        ax_right.text(
            0.95, 0.95,
            '\n'.join(content),
            transform=ax_right.transAxes,
            ha='right', va='top',
        )
        plt.scatter(
            u, u_grad,
            c='red', s=10,
        )
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        plt.xlabel(r'$U$ {units}'.format(units=data.units_label))
        plt.ylabel(r'$dU / d\tau$')
        plt.grid(color='grey', linestyle=':')

        #
        filedir = os.path.join('.', 'img', data.label)
        if not os.path.isdir(filedir):
            os.mkdir(filedir)

        filepath = os.path.join(filedir, f'gradient ({n}).png')
        plt.savefig(filepath)

        #
        plt.show()

    #
    return u_grad
