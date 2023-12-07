import os
from configparser import ConfigParser

from libspectrum2_wrapper.device import Device, DeviceEthernetConfig
from libspectrum2_wrapper.storage import BufferDeviceStorage

from .config import Config, INI_DIRECTORY, DATA_DIRECTORY, IMG_DIRECTORY


# --------        config        --------
parser = ConfigParser(inline_comment_prefixes='#')
flag = parser.read(os.path.join('.', 'config.ini'))
assert flag, 'File `config.ini` not found!'

config = Config(id=parser.get('device', 'id'))

# --------        device        --------
device =  Device(
    storage=BufferDeviceStorage(
        buffer_size=1,
    ),
)
device.create(
    config=DeviceEthernetConfig(
        ip=config.device_ip,
    ),
)
device.connect()


# --------        directories        --------
for folder in [INI_DIRECTORY, DATA_DIRECTORY, IMG_DIRECTORY]:

    filedir = os.path.join('.', folder)
    if not os.path.isdir(filedir):
        os.mkdir(filedir)
