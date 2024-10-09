import os
import reprlib

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.output import Output


def calculate_gradient(
    output: Output,
    show: bool = False,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
) -> Array[float]:
    """Calculate gradient."""
    u_grad = np.gradient(output.average, output.exposure)

    if show:
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        plt.scatter(
            output.exposure, output.average,
            c='red', s=10,
            label=r'$U$',
        )
        plt.plot(
            output.exposure, np.polyval(np.polyfit(output.exposure, output.average, deg=1), output.exposure),
            color='black', linestyle='solid', linewidth=1,
            label=r'$\hat{U}$',
        )
        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ {units}'.format(units=output.units.label))
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.sca(ax_right)
        ax_right.text(
            0.95, 0.95,
            '\n'.join([
                reprlib.repr(output.label),
                fr'n: {output.n}',
            ]),
            transform=ax_right.transAxes,
            ha='right', va='top',
        )
        plt.scatter(
            output.average, u_grad,
            c='red', s=10,
        )
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(r'$U$ {units}'.format(units=output.units.label))
        plt.ylabel(r'$dU / d\tau$')
        plt.grid(color='grey', linestyle=':')

        filedir = os.path.join('.', 'img', output.label)
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        filepath = os.path.join(filedir, f'gradient ({output.n}).png')
        plt.savefig(filepath)

        plt.show()

    return u_grad
