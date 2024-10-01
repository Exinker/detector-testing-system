import os

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.typing import Array

from detector_testing_system.data import Data
from detector_testing_system.signal import Signal


def calculate_gradient(
    signal: Signal,
    show: bool = False,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
) -> Array[float]:
    """Calculate gradient."""
    u_grad = np.gradient(signal.value, signal.exposure)

    if show:
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        plt.scatter(
            signal.exposure, signal.value,
            c='red', s=10,
            label=r'$U$',
        )
        plt.plot(
            signal.exposure, np.polyval(np.polyfit(signal.exposure, signal.value, deg=1), signal.exposure),
            color='black', linestyle='solid', linewidth=1,
            label=r'$\hat{U}$',
        )
        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ {units}'.format(units=signal.units.label))
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.sca(ax_right)
        ax_right.text(
            0.95, 0.95,
            '\n'.join([
                fr'{signal.label}',
                fr'n: {signal.n}',
            ]),
            transform=ax_right.transAxes,
            ha='right', va='top',
        )
        plt.scatter(
            signal.value, u_grad,
            c='red', s=10,
        )
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(r'$U$ {units}'.format(units=signal.units.label))
        plt.ylabel(r'$dU / d\tau$')
        plt.grid(color='grey', linestyle=':')

        filedir = os.path.join('.', 'img', signal.label)
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        filepath = os.path.join(filedir, f'gradient ({signal.n}).png')
        plt.savefig(filepath)

        plt.show()

    return u_grad
