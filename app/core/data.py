import os
import pickle
from collections.abc import Sequence
from time import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from libspectrum2_wrapper.alias import Array, MilliSecond
from libspectrum2_wrapper.device import Device
from libspectrum2_wrapper.units import Units, get_units_clipping, get_units_label

from config.config import DATA_DIRECTORY


class Datum:

    def __init__(self, intensity: Array[float], exposure: MilliSecond, n_frames: int, started_at: float, units: Units):
        self.intensity = intensity
        self.exposure = exposure
        self.n_frames = n_frames
        self.started_at = started_at
        self.units = units

        self._mean = None
        self._variance = None

    @property
    def mean(self) -> Array[float]:
        if self._mean is None:
            self._mean = np.mean(self.intensity, axis=0)

        return self._mean

    @property
    def variance(self) -> Array[float]:
        if self._variance is None:
            self._variance = np.std(self.intensity, axis=0, ddof=1) ** 2

        return self._variance

    @property
    def label(self) -> str:
        return f'{self.exposure}'

    @property
    def units_label(self) -> str:
        return get_units_label(self.units)

    def show(self) -> None:

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)

        plt.plot(
            self.mean,
            label=f'{self.label}',
        )

        plt.xlabel(r'$number$')
        plt.ylabel(r'$U$ {units}'.format(units=self.units_label))

        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.show()


    def __str__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({self.label})'


class Data:

    def __init__(self, __data: Sequence[Datum], units: Units, label: str = ''):
        self.data = list(__data)
        self.units = units
        self.label = label

        self._mean = None
        self._variance = None
        self._exposure = None

    @property
    def mean(self) -> Array[float]:
        if self._mean is None:
            self._mean = np.array([datum.mean for datum in self.data])

        return self._mean

    @property
    def variance(self) -> Array[float]:
        if self._variance is None:
            self._variance = np.array([datum.variance for datum in self.data])

        return self._variance

    @property
    def exposure(self) -> Array[float]:
        if self._exposure is None:
            self._exposure = np.array([datum.exposure for datum in self.data])

        return self._exposure

    @property
    def started_at(self) -> float:
        return self.data[0].started_at

    @property
    def finished_at(self) -> float:
        return self.data[-1].started_at

    @property
    def units_label(self) -> str:
        return get_units_label(self.units)

    def add(self, __data: Sequence[Datum]):
        self.data.extend(__data)

    def show(self, legend: bool = False, save: bool = False) -> None:
        """Show data."""

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)

        plt.plot(
            self.mean.T,
            label=[datum.label for datum in self.data],
        )

        plt.xlabel(r'$number$')
        plt.ylabel(r'$U$ {units}'.format(units=self.units_label))

        plt.grid(color='grey', linestyle=':')
        plt.legend().set_visible(legend)

        if save:
            filedir = os.path.join('.', 'img', self.label)
            if not os.path.isdir(filedir):
                os.mkdir(filedir)

            filepath = os.path.join(filedir, 'data.png')
            plt.savefig(filepath)

        plt.show()

    def save(self) -> None:
        """Save data to `<DATA_DIRECTORY>/<label>/data.pkl` file."""

        filedir = os.path.join(DATA_DIRECTORY, self.label)
        if not os.path.isdir(filedir):
            os.mkdir(filedir)

        filepath = os.path.join(filedir, 'data.pkl')
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, label: str) -> 'Data':
        """Load data from filepath."""

        filedir = os.path.join(DATA_DIRECTORY, label)
        filepath = os.path.join(filedir, 'data.pkl')
        with open(filepath, 'rb') as file:
            tmp = pickle.load(file)

        #
        return Data(
            tmp.data,
            units=tmp.units if hasattr(tmp, 'units') else Units.percent,
            label=tmp.label if hasattr(tmp, 'label') else label,
        )

    def __getitem__(self, index: int) -> Datum:
        return self.data[index]

    def __str__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({self.label})'


def read_datum(device: Device, exposure: MilliSecond, n_frames: int) -> Datum:
    """Read datum with a given `tau` and `n_frames`."""

    device.set_exposure(exposure)

    started_at = time()
    intensity = device.await_read(
        n_frames=n_frames+1,  # FIXME: +1 frame
    )
    intensity = intensity.reshape(intensity.shape[0], intensity.shape[2])

    return Datum(
        intensity=intensity[1:],  # FIXME: +1 frame
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

    return Data(
        data,
        units=device.storage.units,
    )


def load_data(label: str, show: bool = False) -> Data:
    """Load data from `<DATA_DIRECTORY>/<label>/data.pkl` file."""

    # load
    data = Data.load(label=label)

    # show
    if show:
        data.show(legend=False)

    #
    return data


def to_array(data: Data, n: int, threshold: float | None = None) -> tuple[Array[float], Array[float], Array[float]]:
    """Convert data to array."""
    u = data.mean[:,n]
    du = data.variance[:,n]

    threshold = get_units_clipping(units=data.units) if threshold is None else threshold
    cond = u < threshold

    return u[cond], du[cond], data.exposure[cond]
