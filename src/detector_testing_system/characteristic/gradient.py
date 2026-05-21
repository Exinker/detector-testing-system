import os
from collections.abc import Sequence
import reprlib

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array, MilliSecond

from detector_testing_system.experiment import Data, EmptyArrayError, load_data
from detector_testing_system.experiment.utils import create_directory
from detector_testing_system.output import Output


def calculate_gradient(
    output: Output,
    show: bool = False,
    span: tuple[MilliSecond, MilliSecond] = None,
    threshold: tuple[float, float] = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    txlim: tuple[float, float] = None,
    tylim: tuple[float, float] = None,
) -> Array[float]:
    """Calculate gradient."""
    if (threshold != None):   
        lb, ub = threshold
        mask = (lb < output.average) & (output.average < ub)
    else: 
        span = span or (min(output.exposure), max(output.exposure))
        mask = (output.exposure >= span[0]) & (output.exposure <= span[1])
    u_grad = np.gradient(output.average, output.exposure)

    p = np.polyfit(output.exposure[mask], output.average[mask], deg=1)
    

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
            output.exposure, np.polyval(p, output.exposure),
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
        if txlim:
            plt.xlim(txlim)
        if tylim:
            plt.ylim(tylim)
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
            c='grey', s=10,
        )
        plt.scatter(
            output.average[mask], u_grad[mask],
            c='red', s=10,
        )
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(r'$U$ {units}'.format(units=output.units.label))
        plt.ylabel(r'$dU / d\tau$')
        plt.grid(color='grey', linestyle=':')

        filedir = create_directory(os.path.join('.', 'img'), label=output.label)
        filepath = os.path.join(filedir, f'gradient ({output.n}).png')
        plt.savefig(filepath)

        plt.show()

    return u_grad
    
def compare_gradient(
    labels: Sequence[str],
    names: Sequence[str],
    n: int,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    txlim: tuple[float, float] = None,
    tylim: tuple[float, float] = None,   
) -> None:
    j=0
    fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for label in labels:
        data = load_data(
            label=label,
        )

        output = Output.create(data=data, n=n)
        xi = calculate_gradient(
            output=output,
        )

        plt.sca(ax_left)
        plt.scatter(
            output.exposure, output.average,
            s=10,
            label=names[j]
        )
        if txlim:
            plt.xlim(txlim)
        if tylim:
            plt.ylim(tylim)
        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ {units}'.format(units=data.units.label))
        plt.text(
            0.75, 0.10,
            '\n'.join([
                r'$pixel$: {n:d}'.format(
                    n=n,
                ),
            ]),
            transform=ax_left.transAxes,
            ha='left', va='top',
        )
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.sca(ax_right)
        plt.scatter(
            output.average, xi,
            s=10,
            label=names[j]
        )
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(r'$U$ {units}'.format(units=data.units.label))
        plt.ylabel(r'$dU / d\tau$')
        plt.text(
            0.75, 0.1,
            '\n'.join([
                r'$pixel$: {n:d}'.format(
                    n=n,
                ),
            ]),
            transform=ax_right.transAxes,
            ha='left', va='top',
        )
        plt.grid(color='grey', linestyle=':')
        plt.legend()
        j=j+1

    filedir = create_directory(os.path.join('.', 'img'), label=output.label)
    filepath = os.path.join(filedir, f'nonlinearities ({n}).png')
    plt.savefig(filepath)

    plt.show()    

def compare_gradientes(
    datas: Sequence[Data],
    names: Sequence[str],
    n: int,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    txlim: tuple[float, float] = None,
    tylim: tuple[float, float] = None,   
) -> None:
    j=0
    fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for data in datas:

        output = Output.create(data=data, n=n)
        xi = calculate_gradient(
            output=output,
        )

        plt.sca(ax_left)
        plt.scatter(
            output.exposure, output.average,
            s=10,
            label=names[j]
        )
        if txlim:
            plt.xlim(txlim)
        if tylim:
            plt.ylim(tylim)
        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ {units}'.format(units=data.units.label))
        plt.text(
            0.75, 0.10,
            '\n'.join([
                r'$pixel$: {n:d}'.format(
                    n=n,
                ),
            ]),
            transform=ax_left.transAxes,
            ha='left', va='top',
        )
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.sca(ax_right)
        plt.scatter(
            output.average, xi,
            s=10,
            label=names[j]
        )
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(r'$U$ {units}'.format(units=data.units.label))
        plt.ylabel(r'$dU / d\tau$')
        plt.text(
            0.75, 0.1,
            '\n'.join([
                r'$pixel$: {n:d}'.format(
                    n=n,
                ),
            ]),
            transform=ax_right.transAxes,
            ha='left', va='top',
        )
        plt.grid(color='grey', linestyle=':')
        plt.legend()
        j=j+1

    filedir = create_directory(os.path.join('.', 'img'), label=output.label)
    filepath = os.path.join(filedir, f'nonlinearities ({n}).png')
    plt.savefig(filepath)

    plt.show()    
