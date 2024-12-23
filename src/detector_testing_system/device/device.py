from vmk_spectrum3_wrapper.device import Device

from detector_testing_system.experiment import ExperimentConfig


def run_device(config: ExperimentConfig, is_connected: bool = True) -> Device:

    device = Device()

    if is_connected:
        device.connect()

    return device
