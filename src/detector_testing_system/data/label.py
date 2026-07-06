from collections.abc import Sequence
from pathlib import (
    PurePosixPath,
    PureWindowsPath,
)

def split(__path: str) -> Sequence[str]:

    if '\\' in __path:
        return PureWindowsPath(__path).parts

    return PurePosixPath(__path).parts



class Label(str):

    @property
    def prefix(self) -> str:
        prefix, *_ = split(self)
        return prefix or str(self)
