import os

from libspectrum2_wrapper.device import Device, DeviceEthernetConfig
from libspectrum2_wrapper.storage import BufferDeviceStorage

from .config import DEVICE_IP


# --------        device        --------
device =  Device(
    storage=BufferDeviceStorage(
        buffer_size=1,
    ),
)
device.create(
    config=DeviceEthernetConfig(
        ip=DEVICE_IP,
    ),
)
device.connect()


# --------        directories        --------
for folder in ['data', 'img']:
    filedir = os.path.join('.', folder)
    if not os.path.isdir(filedir):
        os.mkdir(filedir)
