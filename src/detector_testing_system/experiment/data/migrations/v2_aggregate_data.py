import pickle
from typing import Any, Mapping

from vmk_spectrum3_wrapper.units import Units

from detector_testing_system.experiment.data.data import Data, Datum


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
    if 'intensity' not in dat:
        raise ValueError('Data format v1 must contain intensity!')

    units = {
        'Units.digit': Units.digit,
        'Units.percent': Units.percent,
        'Units.electron': Units.electron,
    }.get(dat.get('units'), Units.percent)

    tau = dat.get('exposure')
    if isinstance(tau, bytes):
        tau = pickle.loads(tau)

    return Datum.create(
        intensity=pickle.loads(dat.get('intensity')),
        tau=tau,
        n_frames=dat.get('n_frames'),
        started_at=dat.get('started_at'),
        units=units,
    )
