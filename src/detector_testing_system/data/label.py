import os


class Label(str):

    @property
    def prefix(self) -> str:
        prefix, *_ = os.path.split(self)
        return prefix or str(self)
