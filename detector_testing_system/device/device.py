from vmk_spectrum3_wrapper.device import Device
from vmk_spectrum3_wrapper.device.config import DeviceConfigManual

from detector_testing_system.experiment import ExperimentConfig


def run_device(config: ExperimentConfig) -> Device:

    device = Device(
        config=DeviceConfigManual(
            ip=config.device_ip,
        )
    )
    device = device.connect()

    return device
