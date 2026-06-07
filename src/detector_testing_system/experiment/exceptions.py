
class BaseError(Exception):

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(*args)

        self.message = message

    def __str__(self) -> str:
        return self.message


class ConfigError(BaseError):
    pass


class ConfigNotFoundError(ConfigError):
    pass


class ConfigParseError(ConfigError):
    pass


class ConfigExposureError(ConfigError):
    pass


class FitError(BaseError):
    pass
