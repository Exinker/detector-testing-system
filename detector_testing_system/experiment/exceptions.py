
class BaseDataError(Exception):

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(*args)

        self.message = message

    def __str__(self) -> str:
        return self.message


class EmptyArrayError(BaseDataError):
    pass
