import os
from collections.abc import Sequence
import reprlib

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.characteristic.nonlinearity.calculators import calculate_nonlinearity
from detector_testing_system.experiment import Data, EmptyArrayError, load_data
from detector_testing_system.experiment.utils import create_directory
from detector_testing_system.output import Output
from detector_testing_system.utils import calculate_stats


def research_nonlinearity(
    data: Data,
    mask: Array[bool] | None = None,
    method: str = 'fit',
    verbose: bool = False,
    show: bool = False,
    bins: int | Sequence = 40,
    **kwargs,
) -> Array[float]:
    mask = np.full(data.n_numbers, True) if mask is None else mask

    nonlinearity = np.full(data.n_numbers, np.nan)
    for n, *_ in np.argwhere(mask):
        try:
            _, value = calculate_nonlinearity(
                output=Output.create(data=data, n=n),
                method=method,
                **kwargs,
            )

        except EmptyArrayError as error:
            value = float(np.nan)

            if verbose:
                print(error)

        finally:
            nonlinearity[n] = value

    if show:
        mean, ci = calculate_stats(nonlinearity)

        #
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        ax_left.text(
            0.05/2, 0.95,
            '\n'.join([
                reprlib.repr(data.label),
                fr'method: {method}',
            ]),
            transform=ax_left.transAxes,
            ha='left', va='top',
        )
        plt.scatter(
            range(data.n_numbers), nonlinearity,
            c='red', s=10,
            label=r'$U$',
        )
        plt.xlabel(r'number')
        plt.ylabel({
            'fit': r'$\alpha$ [%]',
            'jnorm': r'span [%]',
        }[method])
        plt.grid(color='grey', linestyle=':')

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
            nonlinearity[~np.isnan(nonlinearity)],
            bins=bins,
            edgecolor='black', facecolor='white',
            # fill=False,
        )
        plt.xlabel({
            'fit': r'$\alpha$ [%]',
            'jnorm': r'span [%]',
        }[method])

        plt.show()

    return nonlinearity


def compare_nonlinearity(
    labels: Sequence[str],
    n: int,
    method: str = 'fit',
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    **kwargs,
) -> None:

    fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for label in labels:
        data = load_data(
            label=label,
        )

        output = Output.create(data=data, n=n)
        xi, _ = calculate_nonlinearity(
            output=output,
            method=method,
            **kwargs,
        )

        plt.sca(ax_left)
        plt.scatter(
            output.exposure, output.average,
            s=10,
            label=label.split(' ')[0],
        )
        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ {units}'.format(units=data.units.label))
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.sca(ax_right)
        plt.scatter(
            output.average, xi,
            s=10,
            label=label.split(' ')[0],
        )
        ax_right.text(
            0.95, 0.95,
            fr'method: {method}',
            transform=ax_right.transAxes,
            ha='right', va='top',
        )
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(r'$U$ {units}'.format(units=data.units.label))
        plt.ylabel(r'$error$ [%]')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

    filedir = create_directory(os.path.join('.', 'img'), label=output.label)
    filepath = os.path.join(filedir, f'nonlinearities ({method}, {n}).png')
    plt.savefig(filepath)

    plt.show()
