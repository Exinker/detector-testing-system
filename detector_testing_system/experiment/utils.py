import logging
import os


LOGGER = logging.getLogger(__name__)


def create_directory(__root: str, label: str) -> str:

    filedir = __root
    for suffix in ('', *os.path.split(label)):
        filedir = os.path.join(filedir, suffix)

        if not os.path.isdir(filedir):
            LOGGER.debug(
                'Create directory: %s',
                filedir,
            )
            os.mkdir(filedir)

    return filedir
