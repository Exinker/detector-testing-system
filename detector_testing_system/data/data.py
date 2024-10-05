import os
from collections.abc import Sequence
import pickle
import reprlib
from time import time
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from vmk_spectrum3_wrapper import VERSION
from vmk_spectrum3_wrapper.device import Device
from vmk_spectrum3_wrapper.typing import Array, MilliSecond
from vmk_spectrum3_wrapper.units import Units

from detector_testing_system.types import T


class Datum:

    def __init__(self, intensity: Array[T], exposure: MilliSecond, n_frames: int, started_at: float, units: Units):
        self.intensity = intensity
        self.exposure = exposure
        self.n_frames = n_frames
        self.started_at = started_at
        self.units = units

        self._average = None
        self._variance = None

    @property
    def average(self) -> Array[T]:
        if self._average is None:
            self._average = np.mean(self.intensity, axis=0)

        return self._average

    @property
    def variance(self) -> Array[T]:
        if self._variance is None:
            self._variance = np.std(self.intensity, axis=0, ddof=1) ** 2

        return self._variance

    @property
    def label(self) -> str:
        return f'{self.exposure}'

    @property
    def n_times(self) -> int:
        return self.intensity.shape[0]

    @property
    def n_numbers(self) -> int:
        return self.intensity.shape[1]

    def show(self) -> None:

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.plot(
            self.average,
            label=f'{self.label}',
        )

        plt.xlabel(r'$number$')
        plt.ylabel(r'$U$ {units}'.format(units=self.units.label))

        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.show()

    def dump(self) -> Mapping[str, Any]:
        return {
            'intensity': pickle.dumps(self.intensity),
            'exposure': pickle.dumps(self.exposure),
            'n_frames': self.n_frames,
            'started_at': self.started_at,
        }

    def __str__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({self.label})'


class Data:

    def __init__(self, __data: Sequence[Datum], units: Units, label: str = ''):
        self.data = list(__data)
        self.units = units
        self.label = label

        self._average = None
        self._variance = None
        self._exposure = None

    @property
    def average(self) -> Array[T]:
        if self._average is None:
            self._average = np.array([datum.average for datum in self.data])

        return self._average

    @property
    def variance(self) -> Array[T]:
        if self._variance is None:
            self._variance = np.array([datum.variance for datum in self.data])

        return self._variance

    @property
    def exposure(self) -> Array[MilliSecond]:
        if self._exposure is None:
            self._exposure = np.array([datum.exposure for datum in self.data])

        return self._exposure

    @property
    def started_at(self) -> float:
        return min(
            datum.started_at
            for datum in self.data
        )

    @property
    def finished_at(self) -> float:
        return max(
            datum.started_at
            for datum in self.data
        )

    @property
    def n_times(self) -> int:
        if not self.data:
            return 0

        return self.data[0].n_times

    @property
    def n_numbers(self) -> int:
        if not self.data:
            return 0

        return self.data[0].n_numbers

    def concatenate(self, n: int | None = None) -> Array[T]:
        if n:
            return np.concatenate([datum.intensity[:, n] for datum in self])

        return np.concatenate([datum.intensity for datum in self])

    def add(self, __data: Sequence[Datum]):
        # TODO: check input data shape!

        self.data.extend(__data)

    def show(self, legend: bool = False, save: bool = False) -> None:
        """Show data."""

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.plot(
            self.average.T,
            label=[reprlib.repr(datum.label) for datum in self.data],
        )
        ax.text(
            0.95, 0.95,
            '\n'.join([
                fr'{str(reprlib.repr(self.label))}',
            ]),
            transform=ax.transAxes,
            ha='right', va='top',
        )
        plt.xlabel(r'$number$')
        plt.ylabel(r'$U$ {units}'.format(units=self.units.label))
        plt.grid(color='grey', linestyle=':')
        plt.legend().set_visible(legend)

        if save:
            filedir = os.path.join('.', 'img', self.label)
            if not os.path.isdir(filedir):
                os.mkdir(filedir)

            filepath = os.path.join(filedir, 'data.png')
            plt.savefig(filepath)

        plt.show()

    def dump(self) -> Mapping[str, Any]:
        return {
            'version': VERSION,
            'data': tuple([datum.dump() for datum in self.data]),
            'units': str(self.units),
            'label': str(self.label),
        }

    def save(self) -> None:
        """Save data to `./data//<label>/data.pkl` file."""

        filedir = os.path.join('.', 'data', self.label)
        if not os.path.isdir(filedir):
            os.mkdir(filedir)

        filepath = os.path.join(filedir, 'data.pkl')
        with open(filepath, 'wb') as file:
            pickle.dump(self.dump(), file)

    @classmethod
    def load(cls, label: str) -> 'Data':
        """Load data from filepath."""

        filedir = os.path.join('.', 'data', label)
        filepath = os.path.join(filedir, 'data.pkl')
        with open(filepath, 'rb') as file:
            data_serilized = pickle.load(file)

        units = {
            'Units.digit': Units.digit,
            'Units.percent': Units.percent,
            'Units.electron': Units.electron,
        }.get(data_serilized.get('units'), Units.percent)
        data = Data(
            [
                Datum(
                    intensity=pickle.loads(datum_serilized.get('intensity')),
                    exposure=pickle.loads(datum_serilized.get('exposure')),
                    n_frames=datum_serilized.get('n_frames'),
                    started_at=datum_serilized.get('started_at'),
                    units=units,
                )
                for datum_serilized in data_serilized.get('data', [])
            ],
            units=units,
            label=data_serilized.get('label', label),
        )

        return data

    def __getitem__(self, index: int) -> Datum:
        return self.data[index]

    def __str__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({self.label})'


# --------        handlers        --------
def read_datum(device: Device, exposure: MilliSecond, n_frames: int) -> Datum:
    """Read datum with a given `tau` and `n_frames`."""

    device.set_exposure(exposure)

    started_at = time()
    intensity = device.await_read(
        n_frames=n_frames+1,  # записать один лишний кадр (нередко первый кадр приходит с старым временем экспозиции из-за ошибки в устройстве)
    )
    intensity = intensity.reshape(intensity.shape[0], intensity.shape[2])

    return Datum(
        intensity=intensity[1:],  # выкинуть первый кадр (лишний)
        exposure=exposure,
        n_frames=n_frames,
        started_at=started_at,
        units=device.storage.units,
    )


def read_data(device: Device, exposure: Sequence[MilliSecond], n_frames: int, verbose: bool = True) -> Data:
    """Read data with a given sequence of `exposure` and `n_frames`."""

    data = []
    for tau in tqdm(exposure, disable=not verbose):
        datum = read_datum(device, exposure=tau, n_frames=n_frames)

        data.append(datum)

    #
    return Data(
        data,
        units=device.storage.units,
    )


def load_data(label: str, show: bool = False) -> Data:
    """Load data from `./data//<label>/data.pkl` file."""

    data = Data.load(label=label)

    # show
    if show:
        data.show()

    #
    return data
