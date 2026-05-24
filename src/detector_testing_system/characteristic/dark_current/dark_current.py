import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.characteristic.dark_current.models import (
    BaseDarkCurrentModel,
    DarkCurrentModelABC,
)
from detector_testing_system.data import Data, Trace
from detector_testing_system.experiment import EmptyArrayError
from detector_testing_system.utils import (
    calculate_outlier_bounds,
    calculate_stats,
    trunk_outliers,
)


def calculate_dark_current(
    trace: Trace,
    model: DarkCurrentModelABC | None = None,
    show: bool = False,
) -> float:
    model = model or BaseDarkCurrentModel()

    try:
        result = model.fit(trace=trace)
    except EmptyArrayError:
        raise EmptyArrayError(
            message=f'Data don\'t enough to be fitted! Dark current calculation was failed in cell {trace.n}.',
        )

    if show:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.scatter(
            trace.tau, trace.u,
            c='grey', s=10,
        )
        plt.scatter(
            trace.tau[result.mask], trace.u[result.mask],
            c='red', s=10,
        )
        plt.plot(
            trace.tau, result.interpolate(trace.tau),
            color='black', linestyle='-', linewidth=1,
        )
        plt.text(
            0.05/2, 0.95,
            '\n'.join([
                r'$n$: {n:.0f}'.format(
                    n=trace.n,
                ),
                r'$i$: {value:.4f} {units}'.format(
                    value=1e+3*result.value,  # in %/s
                    units=f'[{trace.units.label}/s]',
                ),
            ]),
            transform=ax.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$\tau$ {units}'.format(units=r'[$ms$]'))
        plt.ylabel(r'$U$ {units}'.format(units=trace.units.label))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return result.value


def research_dark_current(
    data: Data,
    model: DarkCurrentModelABC | None = None,
    mask: Array[bool] | None = None,
    confidence: float = .95,
    verbose: bool = False,
    show: bool = False,
    bins: int = 40,
) -> Array[float]:
    """Calculate a dark current of the cells"""
    mask = np.full(data.n_numbers, True) if mask is None else mask

    dark_current = np.full(data.n_numbers, np.nan)
    for n, *_ in np.argwhere(mask):
        try:
            value = calculate_dark_current(
                trace=data.trace(n),
                model=model,
            )
        except EmptyArrayError as error:
            value = float(np.nan)

            if verbose:
                print(error)
        finally:
            dark_current[n] = value

    if show:
        lb, ub = calculate_outlier_bounds(dark_current, k=3)
        dark_current_trunked = trunk_outliers(dark_current, (lb, ub))
        mean, ci = calculate_stats(dark_current_trunked, confidence=confidence)

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        plt.sca(ax_left)
        plt.scatter(
            np.arange(data.n_numbers), dark_current,
            c='black', s=2,
        )
        plt.text(
            0.05/2, 0.95,
            '\n'.join([
                r'$i_{{d}}$: {mean:.4f} $\pm$ {ci:.4f} [%/s]'.format(
                    mean=1e+3*mean,
                    ci=1e+3*ci,
                ),
            ]),
            transform=ax_left.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'$number$')
        plt.ylabel(r'$i_{{d}}$ {units}'.format(
            units=f'[{data.units.label}/s]',
        ))

        ylim_max = min(1.5 * np.nanmax(dark_current), 5*ub)
        ylim_min = np.nanmin(dark_current) - .025*(ylim_max - np.nanmin(dark_current))
        plt.ylim([
            ylim_min,
            ylim_max,
        ])
        plt.grid(color='grey', linestyle=':')

        plt.sca(ax_right)
        plt.hist(
            1e+3*dark_current_trunked,
            bins=bins,
            edgecolor='black', facecolor='white',
            # fill=False,
        )
        plt.xlabel(r'$i_{{d}}$ {units}'.format(
            units=f'[{data.units.label}/s]',
        ))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return dark_current
