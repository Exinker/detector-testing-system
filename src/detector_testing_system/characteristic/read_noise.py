from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from vmk_spectrum3_wrapper.types import Array, Number

from detector_testing_system.data import Data
from detector_testing_system.types import AxesView
from detector_testing_system.utils import calculate_stats


@dataclass
class ReadNoiseResearch:

    data: Data
    value: Array[float]
    confidence: float = .95
    is_relative: bool = False

    @property
    def number(self) -> Array[Number]:
        return np.arange(self.data.n_numbers)

    def show(
        self,
        bins: int | Sequence = 40,
        views: Sequence[AxesView | None] | None = None,
        verbose: bool = True,
        confidence: float = .95,
        note: str = '',
    ) -> None:
        view_left, view_right = views or [None, None]

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)

        self._show_left(ax_left, view_left, verbose=verbose, note=note)
        self._show_right(ax_right, view_right, bins=bins)

        plt.show()

    def _show_left(
        self,
        ax: Axes,
        view: AxesView | None,
        verbose: bool = True,
        note: str = '',
    ) -> None:
        view = view or {}

        mean, ci = calculate_stats(self.value, confidence=self.confidence)

        plt.sca(ax)

        plt.plot(
            self.value,
            linestyle='none', marker='.', markersize=2,
            color='black',
        )
        plt.axhline(
            mean,
            color='red', linestyle='--', linewidth=1,
        )
        if verbose:
            plt.text(
                0.025, 0.975,
                '\n'.join([
                    self.data.label.prefix,
                    r'$\sigma$: {mean:.4f} $\pm$ {ci:.4f} [%]'.format(
                        mean=mean,
                        ci=ci,
                    ),
                    note,
                ]),
                transform=ax.transAxes,
                ha='left', va='top',
            )

        plt.xlabel('number')
        plt.ylabel({
            False: r'$\sigma$ [{}]'.format(self.data.units.label),
            True: r'$\Delta\sigma / \sigma$ [{}]'.format(self.data.units.label),
        }[self.is_relative])
        plt.grid(color='grey', linestyle=':')

        ax.set(**view)

    def _show_right(
        self,
        ax: Axes,
        view: AxesView | None,
        bins: int,
    ) -> None:
        view = view or {}

        mean, ci = calculate_stats(self.value, confidence=self.confidence)

        plt.sca(ax)

        plt.hist(
            self.value[~np.isnan(self.value)],
            bins=bins,
            edgecolor='black', facecolor='white',
            # fill=False,
        )
        plt.axvline(
            mean,
            color='red', linestyle='--', linewidth=1,
        )

        plt.xlabel({
            False: r'$\sigma$ [{}]'.format(self.data.units.label),
            True: r'$\Delta\sigma / \sigma$ [{}]'.format(self.data.units.label),
        }[self.is_relative])
        plt.ylabel(r'count')
        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


def research_read_noise(data: Data) -> ReadNoiseResearch:
    """Calculate a read noise of the cells."""

    value = np.mean(np.sqrt(data.variance), axis=0)

    return ReadNoiseResearch(
        data=data,
        value=value,
        is_relative=False,
    )


def research_relative_read_noise(data: Data) -> ReadNoiseResearch:
    """Calculate a read noise of the cells."""

    value = 100 * np.std(np.sqrt(data.variance), ddof=1, axis=0) / np.mean(np.sqrt(data.variance), axis=0)

    return ReadNoiseResearch(
        data=data,
        value=value,
        is_relative=True,
    )
