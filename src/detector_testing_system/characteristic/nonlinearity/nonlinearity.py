import reprlib
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system import ROOT
from detector_testing_system.characteristic.dark_current.models import (
    BaseDarkCurrentModel,
    DarkCurrentModelABC,
)
from detector_testing_system.characteristic.nonlinearity.calculators import calculate_nonlinearity
from detector_testing_system.data import Data, load_data
from detector_testing_system.experiment import EmptyArrayError
from detector_testing_system.experiment.utils import create_directory


def research_nonlinearity(
    data: Data,
    model: DarkCurrentModelABC | None = None,
    mask: Array[bool] | None = None,
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
                trace=data.trace(n),
                model=model,
                **kwargs,
            )
        except EmptyArrayError as error:
            value = float(np.nan)

            if verbose:
                print(error)
        finally:
            nonlinearity[n] = value

    if show:
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        ax_left.text(
            0.05/2, 0.95,
            '\n'.join([
                reprlib.repr(data.label),
                'method: {method}'.format(
                    method=getattr(model, 'name', 'base'),
                ),
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
            'base': r'$\alpha$ [%]',
            'jnorm': r'$\Delta U$ [%]',
        }[getattr(model, 'name', 'base')])
        plt.grid(color='grey', linestyle=':')

        plt.sca(ax_right)
        plt.hist(
            nonlinearity[~np.isnan(nonlinearity)],
            bins=bins,
            edgecolor='black', facecolor='white',
            # fill=False,
        )
        plt.xlabel({
            'base': r'$\alpha$ [%]',
            'jnorm': r'$\Delta U$ [%]',
        }[getattr(model, 'name', 'base')])

        plt.show()

    return nonlinearity


def compare_nonlinearity(
    labels: Sequence[str],
    n: int,
    model: DarkCurrentModelABC | None = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    **kwargs,
) -> None:
    model = model or BaseDarkCurrentModel()

    fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for label in labels:
        data = load_data(
            label=label,
        )

        trace = data.trace(n)
        xi, _ = calculate_nonlinearity(
            trace=trace,
            model=model,
            **kwargs,
        )

        plt.sca(ax_left)
        plt.scatter(
            trace.tau, trace.u,
            s=10,
            label=label.split(' ')[0],
        )
        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ {units}'.format(units=data.units.label))
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.sca(ax_right)
        plt.scatter(
            trace.u, xi,
            s=10,
            label=label.split(' ')[0],
        )
        ax_right.text(
            0.95, 0.95,
            'method: {method}'.format(
                method=getattr(model, 'name', 'base'),
            ),
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

    filedir = create_directory(ROOT / 'img', label=trace.label)
    filepath = filedir / 'nonlinearities ({method}), {n}).png'.format(
        method=getattr(model, 'name', 'base'),
        n=n,
    )
    plt.savefig(filepath)

    plt.show()
