from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.device import Device
from vmk_spectrum3_wrapper.types import Array, MilliSecond, U
from vmk_spectrum3_wrapper.units import Units


class Datum:

    def __init__(
        self,
        u: Array[U],
        variance: Array[U],
        tau: MilliSecond,
        n_frames: int,
        started_at: float,
        units: Units,
    ) -> None:

        self.tau = tau
        self.n_frames = n_frames
        self.started_at = started_at
        self.units = units

        self._u = u
        self._variance = variance

    @property
    def u(self) -> Array[U]:
        return self._u

    @property
    def variance(self) -> Array[U]:
        return self._variance

    @property
    def label(self) -> str:
        return str(self.tau)

    @property
    def n_times(self) -> int:
        return self.n_frames

    @property
    def n_numbers(self) -> int:
        return self.u.shape[0]

    def show(self) -> None:

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.plot(
            self.u,
            label=self.label,
        )

        plt.xlabel(r'$number$')
        plt.ylabel(r'$U$ [{units}]'.format(units=self.units.label))

        plt.grid(color='grey', linestyle=':')
        plt.legend(loc='upper left')

        plt.show()

    def dumps(self) -> Mapping[str, Any]:
        return {
            'u': self.u,
            'variance': self.variance,
            'tau': self.tau,
            'n_frames': self.n_frames,
            'started_at': self.started_at,
            'units': str(self.units),
        }

    @classmethod
    def read(cls, device: Device) -> 'Datum':

        raw = device.read()
        return cls.create(
            intensity=raw.intensity,
            tau=raw.meta.exposure,
            n_frames=raw.meta.capacity,
            started_at=raw.meta.started_at,
            units=raw.units,
        )

    @classmethod
    def create(
        cls,
        intensity: Array[U],
        tau: MilliSecond,
        n_frames: int,
        started_at: float,
        units: Units,
    ) -> 'Datum':

        return cls(
            u=np.mean(intensity, axis=0),
            variance=np.std(intensity, axis=0, ddof=1) ** 2,
            tau=tau,
            n_frames=n_frames,
            started_at=started_at,
            units=units,
        )

    @classmethod
    def loads(cls, dat: Mapping[str, Any]) -> 'Datum':

        units = {
            'Units.digit': Units.digit,
            'Units.percent': Units.percent,
            'Units.electron': Units.electron,
        }.get(dat.get('units'), Units.percent)

        if 'intensity' in dat:
            raise ValueError(
                'Data format v1 is not supported by Data.load(). '
                'Run migrate_data(label) first.',
            )

        return cls(
            u=dat.get('u'),
            variance=dat.get('variance'),
            tau=dat.get('tau'),
            n_frames=dat.get('n_frames'),
            started_at=dat.get('started_at'),
            units=units,
        )

    def __str__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({self.label})'
