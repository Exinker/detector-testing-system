from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from detector_testing_system.loggers import setdefault_logger


setdefault_logger()

ROOT = Path(__file__).parents[2]


__name__ = 'Detector Testing System'
try:
    __version__ = version('detector-testing-system')
except PackageNotFoundError:
    __version__ = '0.0.0'
__author__ = 'Pavel Vaschenko'
__email__ = 'vaschenko@vmk.ru'
__organization__ = 'VMK-Optoelektronika'
__license__ = 'MIT'
__copyright__ = 'Copyright {}, {}'.format(datetime.now().year, __organization__)
