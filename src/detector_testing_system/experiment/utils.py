import logging
import os
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def create_directory(__root: Path, label: str) -> Path:

    filedir = __root
    for suffix in ('', *os.path.split(label)):
        filedir = os.path.join(filedir, suffix)

        if not os.path.isdir(filedir):
            LOGGER.debug(
                'Create directory: %s',
                filedir,
            )
            os.mkdir(filedir)

    return Path(filedir)
