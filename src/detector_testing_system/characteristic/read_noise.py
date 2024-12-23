import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.experiment import Data


def show_intensity(data: Data, n: int, show: bool = False, bins: int = 100) -> Array[float]:
    """Show a intensity of the cell."""

    intensity = data.concatenate(n=n)

    if show:
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)

        plt.sca(ax_left)
        plt.plot(
            intensity,
            linestyle='none', marker='.', markersize=1,
            color='black',
        )
        plt.xlabel('time')
        plt.ylabel(fr'$U$ {data.units.label}')
        plt.grid(color='grey', linestyle=':')

        plt.sca(ax_right)
        plt.hist(
            intensity,
            bins=bins,
            facecolor='white', edgecolor='black',
        )
        plt.xlabel(fr'$U$ {data.units.label}')
        plt.ylabel('freq')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return intensity


def research_read_noise(data: Data, show: bool = False) -> Array[float]:
    """Calculate a read noise of the cells."""

    read_noise = np.mean(np.sqrt(data.variance), axis=0)

    if show:
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)

        plt.sca(ax_left)
        plt.plot(
            read_noise,
            linestyle='none', marker='.', markersize=2,
            color='black',
        )
        plt.xlabel('number')
        plt.ylabel(fr'$\sigma$ {data.units.label}')
        plt.grid(color='grey', linestyle=':')

        plt.sca(ax_right)
        plt.hist(
            read_noise,
            bins=100,
            facecolor='white', edgecolor='black',
        )
        plt.xlabel(fr'$\sigma$ {data.units.label}')
        plt.ylabel('count')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return read_noise


def research_relative_read_noise(data: Data, show: bool = False) -> Array[float]:
    """Calculate a read noise of the cells."""

    read_noise = 100 * np.std(np.sqrt(data.variance), ddof=1, axis=0) / np.mean(np.sqrt(data.variance), axis=0)

    if show:
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)

        plt.sca(ax_left)
        plt.plot(
            read_noise,
            linestyle='none', marker='.', markersize=2,
            color='black',
        )
        plt.xlabel('number')
        plt.ylabel(fr'$\sigma / \Delta\sigma$ {data.units.label}')
        plt.grid(color='grey', linestyle=':')

        plt.sca(ax_right)
        plt.hist(
            read_noise,
            bins=100,
            facecolor='white', edgecolor='black',
        )
        plt.xlabel(fr'$\sigma / \Delta\sigma$ {data.units.label}')
        plt.ylabel('freq')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return read_noise
