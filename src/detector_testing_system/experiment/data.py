import os
import pickle
import reprlib
from collections.abc import Sequence
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from vmk_spectrum3_wrapper import VERSION
from vmk_spectrum3_wrapper.config import DEFAULT_DETECTOR
from vmk_spectrum3_wrapper.data import Data as RawData
from vmk_spectrum3_wrapper.detector import Detector
from vmk_spectrum3_wrapper.device import Device
from vmk_spectrum3_wrapper.measurement_manager.filters import (
    ClipFilter,
    PipeFilter,
    ScaleFilter,
)
from vmk_spectrum3_wrapper.types import Array, MilliSecond
from vmk_spectrum3_wrapper.units import U, Units

from detector_testing_system.experiment.utils import create_directory


class Datum:

    def __init__(
        self,
        intensity: Array[U],
        exposure: MilliSecond,
        n_frames: int,
        started_at: float,
        units: Units,
    ) -> None:
        self.intensity = intensity
        self.exposure = exposure
        self.n_frames = n_frames
        self.started_at = started_at
        self.units = units

        self._average = None
        self._variance = None

    @property
    def average(self) -> Array[U]:
        if self._average is None:
            self._average = np.mean(self.intensity, axis=0)

        return self._average

    @property
    def variance(self) -> Array[U]:
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
            label=reprlib.repr(self.label),
        )

        plt.xlabel(r'$number$')
        plt.ylabel(r'$U$ {units}'.format(units=self.units.label))

        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.show()

    def dumps(self) -> Mapping[str, Any]:
        return {
            'intensity': pickle.dumps(self.intensity),
            'exposure': pickle.dumps(self.exposure),
            'n_frames': self.n_frames,
            'started_at': self.started_at,
            'units': str(self.units),
        }

    @classmethod
    def read(cls, device: Device) -> 'Datum':

        raw = device.read()
        return cls(
            intensity=raw.intensity,
            exposure=raw.meta.exposure,
            n_frames=raw.meta.capacity,
            started_at=raw.meta.started_at,
            units=raw.units,
        )

    @classmethod
    def loads(cls, dat: Mapping[str, Any]) -> 'Datum':

        units = {
            'Units.digit': Units.digit,
            'Units.percent': Units.percent,
            'Units.electron': Units.electron,
        }.get(dat.get('units'), Units.percent)

        datum = cls(
            intensity=pickle.loads(dat.get('intensity')),
            exposure=pickle.loads(dat.get('exposure')),
            n_frames=dat.get('n_frames'),
            started_at=dat.get('started_at'),
            units=units,
        )
        return datum

    def __str__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({self.label})'


class Data:

    def __init__(
        self,
        __data: Sequence[Datum],
        label: str = '',
    ) -> None:
        self.data = tuple(__data)
        self.label = label

        self._average = None
        self._variance = None
        self._exposure = None

    @property
    def average(self) -> Array[U]:
        if self._average is None:
            self._average = np.array([datum.average for datum in self.data])

        return self._average

    @property
    def variance(self) -> Array[U]:
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
        return min(datum.started_at for datum in self.data)

    @property
    def finished_at(self) -> float:
        return max(datum.started_at for datum in self.data)

    @property
    def n_numbers(self) -> int:
        if not self.data:
            raise ValueError

        return self.data[0].n_numbers

    @property
    def units(self) -> Units:
        if not self.data:
            raise ValueError

        return self.data[0].units

    def concatenate(self, n: int) -> Array[U]:
        return np.concatenate([datum.intensity[:, n] for datum in self])

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
            filedir = create_directory(os.path.join('.', 'img'), label=self.label)
            filepath = os.path.join(filedir, 'data.png')
            plt.savefig(filepath)

        plt.show()

    def save(self) -> None:
        """Save data to `./data/<label>/data.pkl` file."""

        filedir = create_directory(os.path.join('.', 'data'), label=self.label)
        filepath = os.path.join(filedir, 'data.pkl')
        with open(filepath, 'wb') as file:
            pickle.dump(self.dumps(), file)

    def dumps(self) -> Mapping[str, Any]:

        dat = {
            'version': VERSION,
            'data': tuple([datum.dumps() for datum in self.data]),
            'units': str(self.units),
            'label': str(self.label),
        }
        return dat

    @classmethod
    def create(cls, __data: Sequence[Datum], label: str = '') -> 'Data':
        assert len(set(datum.units for datum in __data)) == 1, 'Data units have to be the same!'
        assert len(set(datum.n_numbers for datum in __data)) == 1, 'Data shapes have to be the same!'

        return Data(__data, label=label)

    @classmethod
    def load(cls, label: str) -> 'Data':
        """Load data from filepath."""

        filedir = os.path.join('.', 'data', label)
        filepath = os.path.join(filedir, 'data.pkl')
        with open(filepath, 'rb') as file:
            dat = pickle.load(file)

        data = cls.loads(dat, label=label)
        return data

    @classmethod
    def loads(cls, dat: Mapping[str, Any], label: str) -> 'Data':

        data = cls(
            map(Datum.loads, dat.get('data', [])),
            label=dat.get('label', label),
        )
        return data

    def __getitem__(self, index: int) -> Datum:
        return self.data[index]

    def __str__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({self.label})'


def read_data(
    device: Device,
    exposure: Sequence[MilliSecond],
    n_frames: int,
    verbose: bool = True,
) -> Data:
    """Read data with a given sequence of `exposure` and `n_frames`."""

    data = []
    for tau in tqdm(exposure, disable=not verbose):
        device.setup(
            n_times=1,
            exposure=float(tau),
            capacity=n_frames,
            filter=PipeFilter(filters=[
                ClipFilter(),
                ScaleFilter(units=Units.percent),
            ]),
        )

        datum = Datum.read(
            device=device,
        )
        data.append(datum)

    return Data.create(data)


def load_data(
    label: str,
    show: bool = False,
) -> Data:
    """Load data from `./data//<label>/data.pkl` file."""

    data = Data.load(label=label)

    if show:
        data.show()

    return data


def split_data_by_detector(
    __data: Data,
    detector: Detector = DEFAULT_DETECTOR,
) -> tuple[Data]:
    n_pixels = detector.config.n_pixels
    assert __data.n_numbers % n_pixels == 0, 'Invalid detector is selected!'

    def select(datum: Datum, index: slice) -> Datum:
        return Datum(
            intensity=datum.intensity[:, index],
            exposure=datum.exposure,
            n_frames=datum.n_frames,
            started_at=datum.started_at,
            units=datum.units,
        )

    return tuple([
        Data([select(datum, slice(n_pixels*(n), n_pixels*(n+1))) for datum in __data.data], label=__data.label)
        for n in range(__data.n_numbers // n_pixels)
    ])
