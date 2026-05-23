import reprlib

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system import ROOT
from detector_testing_system.characteristic.dark_current.models import JNormDarkCurrentModel
from detector_testing_system.characteristic.gradient import calculate_gradient
from detector_testing_system.data import Trace
from detector_testing_system.experiment.utils import create_directory


def calculate_nonlinearity_jnorm(
    trace: Trace,
    model: JNormDarkCurrentModel,
    k: float = 2,
    show: bool = False,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
) -> tuple[Array[float], float]:

    u_grad = calculate_gradient(trace=trace)
    result = model.fit(trace=trace)

    x_intersection = _calculate_intersection(
        u=trace.u,
        u_grad=u_grad,
        threshold=k * result.value,
    )
    if x_intersection is not None:
        span = x_intersection - trace.u[0]
    else:
        span = 0

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
                fr'$a = {{{result.value:.4f}}}$',
                fr'$b = {{{result.bias:.4f}}}$',
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
            trace.u, u_grad,
            c='grey', s=10,
        )
        plt.scatter(
            trace.u[result.mask], u_grad[result.mask],
            c='red', s=10,
            label=r'$U$',
        )
        plt.axhline(
            result.value,
            color='black', linestyle='solid', linewidth=1,
        )
        plt.axhline(
            k * result.value,
            color='red', linestyle='--', linewidth=1,
        )
        if x_intersection is not None:
            ax_right.axvspan(
                trace.u[0],
                x_intersection,
                color='grey',
                alpha=.125,
            )
        ax_right.text(
            0.95, 0.95,
            '\n'.join([
                fr'{str(reprlib.repr(trace.label))}',
                fr'n: {trace.n}',
                fr'$\Delta U$: {span:.2f} [%]',
            ]),
            transform=ax_right.transAxes,
            ha='right', va='top',
        )
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(r'$U$ {units}'.format(units=trace.units.label))
        plt.ylabel(r'$dU / d\tau$')
        plt.grid(color='grey', linestyle=':')

        filedir = create_directory(ROOT / 'img', label=trace.label)
        filepath = filedir / f'nonlinearity-jnorm ({trace.n}).png'
        plt.savefig(filepath)

        plt.show()

    return result.xi, span


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
