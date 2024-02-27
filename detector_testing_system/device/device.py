from vmk_spectrum2_wrapper.device import Device, DeviceEthernetConfig
from vmk_spectrum2_wrapper.storage import BufferDeviceStorage

from detector_testing_system.experiment import ExperimentConfig


def run_device(config: ExperimentConfig, is_connected: bool = True) -> Device:

    device = Device(
        storage=BufferDeviceStorage(
            buffer_size=1,
        ),
    )
    device.create(
        config=DeviceEthernetConfig(
            ip=config.device_ip,
        ),
    )

    if is_connected:
        device.connect()

    return device
