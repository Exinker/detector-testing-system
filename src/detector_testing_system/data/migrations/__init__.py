import pickle
from typing import Any, Callable, Mapping

from detector_testing_system import ROOT
from detector_testing_system.data.data import Data, get_data_version
from detector_testing_system.data.migrations import v2_aggregate_data


Migration = Callable[[Mapping[str, Any], str], Mapping[str, Any]]
MIGRATIONS: Mapping[int, tuple[int, Migration]] = {
    v2_aggregate_data.SOURCE_VERSION: (
        v2_aggregate_data.TARGET_VERSION,
        v2_aggregate_data.migrate,
    ),
}


def migrate_data(label: str) -> Data:
    """Migrate `./data/<label>/data.pkl` to the current data format version"""

    filedir = ROOT / 'data' / label

    filepath = filedir / 'data.pkl'
    with open(filepath, 'rb') as file:
        dat = pickle.load(file)

    source_version = get_data_version(dat)
    if source_version == Data.DATA_VERSION:
        raise ValueError('Data is already in the current format!')
    if source_version > Data.DATA_VERSION:
        raise ValueError(f'Data format v{source_version} is newer than supported v{Data.DATA_VERSION}!')

    backup_filepath = filedir / f'v{source_version}.data.pkl'
    if backup_filepath.exists():
        raise FileExistsError(f'Backup file already exists: {backup_filepath}')

    migrated = _migrate(dat, label=label)
    data = Data.loads(migrated, label=label)

    filepath.rename(backup_filepath)
    with open(filepath, 'wb') as file:
        pickle.dump(migrated, file)

    return data


def _migrate(dat: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    migrated = dat
    version = get_data_version(migrated)

    while version < Data.DATA_VERSION:

        source_version = version
        try:
            target_version, migration = MIGRATIONS[version]
        except KeyError as error:
            raise ValueError(f'Migration from data format v{version} is not registered!') from error

        migrated = migration(migrated, label)
        version = get_data_version(migrated)
        if version != target_version:
            raise ValueError(
                f'Migration from data format v{source_version} produced unexpected result; '
                f'expected v{target_version}.',
            )

    return migrated
