import pickle
from typing import Any, Mapping

from vmk_spectrum3_wrapper.units import Units

from detector_testing_system.data.data import Data
from detector_testing_system.data.datum import Datum


SOURCE_VERSION = 1
TARGET_VERSION = 2


def migrate(dat: Mapping[str, Any], label: str) -> Mapping[str, Any]:

    data = Data(
        map(_load_v1_datum, dat.get('data', [])),
        label=dat.get('label', label),
    )

    migrated = dict(data.dumps())
    migrated['version'] = TARGET_VERSION
    return migrated


def _load_v1_datum(dat: Mapping[str, Any]) -> Datum:

    for key in [
        'intensity',
        'exposure',
        'n_frames',
        'started_at',
        'units',
    ]:
        if key not in dat:
            raise ValueError(f'Data format v1 must contain key: {key}!')

    tau = dat['exposure']
    if isinstance(tau, bytes):
        tau = pickle.loads(tau)

    units = {
        'Units.digit': Units.digit,
        'Units.percent': Units.percent,
        'Units.electron': Units.electron,
    }.get(dat['units'], Units.percent)

    return Datum.create(
        intensity=pickle.loads(dat['intensity']),
        tau=tau,
        n_frames=dat['n_frames'],
        started_at=dat['started_at'],
        units=units,
    )
