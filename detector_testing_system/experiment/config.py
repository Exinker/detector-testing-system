from configparser import ConfigParser
from dataclasses import dataclass

from vmk_spectrum3_wrapper.types import MilliSecond
from vmk_spectrum3_wrapper.units import Units


@dataclass
class ExperimentConfig:
    device_id: str
    device_ip: str
    device_units: Units

    detector: str

    check_exposure_flag: bool
    check_exposure_min: MilliSecond
    check_exposure_max: MilliSecond

    check_total_flag: bool

    check_source_flag: bool
    check_source_tau: MilliSecond
    check_source_n_frames: int

    @classmethod
    def from_ini(cls, filepath: str) -> 'ExperimentConfig':
        parser = ConfigParser(inline_comment_prefixes='#')

        flag = parser.read(filepath)
        assert flag, f'File {filepath} not found!'

        config = cls(
            device_id=parser.get('device', 'id'),
            device_ip=parser.get('device', 'ip'),
            device_units=parser.get('device', 'units'),

            detector=parser.get('detector', 'type'),

            check_exposure_flag={
                'False': False,
                'True': True,
            }[parser.get('check', 'check_exposure_flag')],
            check_exposure_min=float(parser.get('check', 'check_exposure_min')),
            check_exposure_max=float(parser.get('check', 'check_exposure_max')),

            check_total_flag={
                'False': False,
                'True': True,
            }[parser.get('check', 'check_total_flag')],

            check_source_flag={
                'False': False,
                'True': True,
            }[parser.get('check', 'check_source_flag')],
            check_source_tau=float(parser.get('check', 'check_source_tau')),
            check_source_n_frames=int(parser.get('check', 'check_source_n_frames')),
        )

        return config
