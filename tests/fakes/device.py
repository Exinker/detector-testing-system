from collections.abc import Sequence
from typing import Callable, Mapping

import numpy as np

from vmk_spectrum3_wrapper.data import Data, DataMeta
from vmk_spectrum3_wrapper.device.config import DeviceConfig, DeviceConfigAuto, DeviceConfigManual
from vmk_spectrum3_wrapper.filter import F
from vmk_spectrum3_wrapper.measurement import fetch_measurement
from vmk_spectrum3_wrapper.typing import Array, Digit, IP, MilliSecond


class FakeDevice:

    def __init__(
        self,
        config: DeviceConfig | None = None,
        verbose: bool = False,
    ) -> None:

        self._config = config or DeviceConfigAuto()
        # self._device = FakeDeviceFactory(
        #     on_context=self._on_context,
        #     on_status=self._on_status,
        #     on_error=self._on_error,
        # ).create(
        #     config=self.config,
        # )
        self._device = None
        self._status = None
        self._is_connected = False

        self.verbose = verbose

    def connect(self) -> 'FakeDevice':
        self._is_connected = True

        return self

    def disconnect(self) -> 'FakeDevice':
        self._is_connected = False

        return self

    def setup(
        self,
        n_times: int,
        exposure: MilliSecond | tuple[MilliSecond, MilliSecond],
        capacity: int | tuple[int, int] = 1,
        handler: F | None = None,
    ) -> 'FakeDevice':

        self._measurement = fetch_measurement(
            n_times=n_times,
            exposure=exposure,
            capacity=capacity,
            handler=handler,
        )

        return self

    def read(
        self,
        blocking: bool = True,
        timeout: MilliSecond = 100,
    ) -> Data | None:
        assert blocking is True, 'Non blicking mode is not implemented!'

        if blocking:
            while self._measurement.progress < 1:
                self._wait(timeout)

            storage = self._measurement.storage

            return Data.squeeze(
                storage.pull(),
                DataMeta(
                    exposure=storage.exposure,
                    capacity=storage.capacity,
                    started_at=storage.started_at,
                    finished_at=storage.finished_at,
                ),
            )

