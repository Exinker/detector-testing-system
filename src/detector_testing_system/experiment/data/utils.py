import numpy as np

from vmk_spectrum3_wrapper.config import DEFAULT_DETECTOR
from vmk_spectrum3_wrapper.detector import Detector
from vmk_spectrum3_wrapper.types import Array

from detector_testing_system.experiment.data.data import Data, Datum


def split_data_by_detector(
    __data: Data,
    detector: Detector = DEFAULT_DETECTOR,
) -> tuple[Data]:
    n_pixels = detector.config.n_pixels
    assert __data.n_numbers % n_pixels == 0, 'Invalid detector is selected!'

    def select(datum: Datum, index: slice) -> Datum:
        return Datum(
            intensity=datum.intensity[:, index],
            exposure=datum.exposure,
            n_frames=datum.n_frames,
            started_at=datum.started_at,
            units=datum.units,
        )

    return tuple([
        Data([select(datum, slice(n_pixels*(n), n_pixels*(n+1))) for datum in __data.data], label=__data.label)
        for n in range(__data.n_numbers // n_pixels)
    ])


def create_mask(__data: Data, bounds: tuple[int, int]) -> Array[bool]:
    lb, ub = bounds
    mask = np.full(__data.n_numbers, False)
    mask[lb:ub] = True

    return mask
