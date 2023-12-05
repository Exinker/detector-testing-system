import os
from configparser import ConfigParser

# --------        dirs        --------
INI_DIRECTORY = os.path.join('.', 'ini')  # os.path.join('.', 'data') or r'\\fileserver\Users\vaschenko\test42'
DATA_DIRECTORY = os.path.join('.', 'data')  # os.path.join('.', 'data') or r'\\fileserver\Users\vaschenko\test42'
IMG_DIRECTORY = os.path.join('.', 'img')  # os.path.join('.', 'data') or r'\\fileserver\Users\vaschenko\test42'


# --------        config        --------
class Config:
    '''Experiment's config.'''

    def __init__(self, id: str) -> None:

        # ini parser
        parser = ConfigParser(inline_comment_prefixes='#')
        parser.read(os.path.join(INI_DIRECTORY, f'{id}.ini'))

        # --------        device        --------
        self.device_id = id
        self.device_ip = parser.get('device', 'ip')
        self.device_units = parser.get('device', 'units')

        # --------        detector        --------
        self.detector = parser.get('detector', 'type')

        # --------        check        --------
        self.check_exposure_flag = {
            'False': False,
            'True': True,
        }[parser.get('check', 'check_exposure_flag')]
        self.check_exposure_min = float(parser.get('check', 'check_exposure_min'))
        self.check_exposure_max = float(parser.get('check', 'check_exposure_max'))

        self.check_total_flag = {
            'False': False,
            'True': True,
        }[parser.get('check', 'check_total_flag')]

        self.check_source_flag = {
            'False': False,
            'True': True,
        }[parser.get('check', 'check_source_flag')]
        self.check_source_tau = float(parser.get('check', 'check_source_tau'))
        self.check_source_n_frames = int(parser.get('check', 'check_source_n_frames'))

        # --------        device        --------





        # --------        device        --------





        # --------        device        --------





        # --------        device        --------








        # --------        tests        --------
        self.test()

    def test(self) -> None:
        pass
