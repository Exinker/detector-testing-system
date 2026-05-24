import os
import pickle
from collections.abc import Sequence
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from vmk_spectrum3_wrapper.device import Device
from vmk_spectrum3_wrapper.measurement_manager.filters import (
    ClipFilter,
    PipeFilter,
    ScaleFilter,
)
from vmk_spectrum3_wrapper.types import Array, MilliSecond, Number, U
from vmk_spectrum3_wrapper.units import Units

from detector_testing_system import ROOT
from detector_testing_system.data.datum import Datum
from detector_testing_system.data.label import Label
from detector_testing_system.data.trace import Trace


def get_data_version(dat: Mapping[str, Any]) -> int:

    try:
        return int(dat.get('version', 1))
    except (TypeError, ValueError):
        return 1


class Data:

    DATA_VERSION = 2

    def __init__(
        self,
        __data: Sequence[Datum],
        label: str | Label,
    ) -> None:

        self.data = tuple(__data)
        self.label = label if isinstance(label, Label) else Label(label)

        self._u = None
        self._variance = None
        self._tau = None

    @property
    def u(self) -> Array[U]:
        if self._u is None:
            self._u = np.array([datum.u for datum in self.data])

        return self._u

    @property
    def variance(self) -> Array[U]:
        if self._variance is None:
            self._variance = np.array([datum.variance for datum in self.data])

        return self._variance

    @property
    def tau(self) -> Array[MilliSecond]:
        if self._tau is None:
            self._tau = np.array([datum.tau for datum in self.data])

        return self._tau

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

    def trace(
        self,
        __n: Number,
    ) -> Trace:

        return Trace(
            u=self.u[:, __n],
            variance=self.variance[:, __n],
            tau=self.tau,
            n=__n,
            label=self.label,
            units=self.units,
        )

    def show(self, legend: bool = False, save: bool = False) -> None:
        """Show data"""

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.plot(
            self.u.T,
            label=[datum.label for datum in self.data],
        )
        plt.text(
            0.95, 0.95,
            '\n'.join([
                self.label.prefix,
            ]),
            transform=ax.transAxes,
            ha='right', va='top',
        )
        plt.xlabel(r'$number$')
        plt.ylabel(r'$U$ [{units}]'.format(units=self.units.label))
        plt.grid(color='grey', linestyle=':')
        plt.legend().set_visible(legend)

        if save:
            filedir = ROOT / 'img' / self.label
            filedir.mkdir(parents=True, exist_ok=True)

            plt.savefig(filedir / 'data.png')

        plt.show()

    def save(self) -> None:
        """Save data to `./data/<label>/data.pkl` file"""

        filedir = ROOT / 'data' / self.label
        filedir.mkdir(parents=True, exist_ok=True)

        with open(filedir / 'data.pkl', 'wb') as file:
            pickle.dump(self.dumps(), file)

    def dumps(self) -> Mapping[str, Any]:

        dat = {
            'version': self.DATA_VERSION,
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
        """Load data from filepath"""

        filedir = ROOT / 'data' / label
        filepath = filedir / 'data.pkl'
        with open(filepath, 'rb') as file:
            dat = pickle.load(file)

        data = cls.loads(dat, label=label)
        return data

    @classmethod
    def loads(cls, dat: Mapping[str, Any], label: str) -> 'Data':
        if get_data_version(dat) != cls.DATA_VERSION:
            raise ValueError(
                'Unsupported data format version. '
                'Run migrate_data(label) first.',
            )

        data = cls(
            map(Datum.loads, dat.get('data', [])),
            label=dat.get('label', label),
        )
        return data

    def __getitem__(self, index: int) -> Datum:
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({self.label})'


def read_data(
    device: Device,
    tau: Sequence[MilliSecond],
    n_frames: int,
    verbose: bool = True,
) -> Data:
    """Read data with a given sequence of `tau` and `n_frames`"""

    data = []
    for exposure in tqdm(tau, disable=not verbose):
        device.setup(
            n_times=1,
            exposure=float(exposure),
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
