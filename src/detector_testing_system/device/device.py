from vmk_spectrum3_wrapper.device import Device
from vmk_spectrum3_wrapper.device.device_config import DeviceConfigAuto, CHANGE_EXPOSURE_TIMEOUT
from vmk_spectrum3_wrapper.types import MilliSecond

from detector_testing_system.experiment import ExperimentConfig


def run_device(
    config: ExperimentConfig,
    change_exposure_timeout: MilliSecond = CHANGE_EXPOSURE_TIMEOUT,
    is_connected: bool = True,
) -> Device:

    device = Device(
        config=DeviceConfigAuto(
            change_exposure_timeout=change_exposure_timeout,
        ),
    )

    if is_connected:
        device.connect()

    return device
