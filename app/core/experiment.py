import os
from collections.abc import Sequence
from datetime import datetime
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from libspectrum2_wrapper.alias import Array, MilliSecond
from libspectrum2_wrapper.device import Device

from .config import DIRECTORY, EXPOSURE_MAX, EXPOSURE_MIN
from .config import CHECK_SOURCE_FLAG, CHECK_SOURCE_TAU, CHECK_SOURCE_N_FRAMES, CHECK_TOTAL_FLAG, CHECK_EXPOSURE_FLAG
from .data import Data, read_datum, read_data


def check_source(func: Callable) -> Callable:
    """Check a stability of the light source."""

    def wrapper(device: Device, *args, **kwargs):
        if CHECK_SOURCE_FLAG is False:
            return func(device, *args, **kwargs)

        #
        before = read_datum(
            device,
            exposure=CHECK_SOURCE_TAU,
            n_frames=CHECK_SOURCE_N_FRAMES,
        )
        experiment = func(device, *args, **kwargs)
        after = read_datum(
            device,
            exposure=CHECK_SOURCE_TAU,
            n_frames=CHECK_SOURCE_N_FRAMES,
        )

        duration = after.started_at - before.started_at
        bias = np.mean(after.mean - before.mean)

        # show
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)

        plt.sca(ax_left)
        plt.plot(
            before.mean,
            label='before',
        )
        plt.plot(
            after.mean,
            label='after',
        )
        plt.xlabel(r'number')
        plt.ylabel(r'$U$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.sca(ax_right)
        plt.plot(
            (after.mean - before.mean).squeeze(),
            color='black',
        )
        plt.axhline(
            bias,
            color='red', linestyle='solid', linewidth=2,
        )
        content = [
            '{started_at} / {finished_at}'.format(
                started_at=datetime.strftime(datetime.fromtimestamp(before.started_at), '%Y-%m-%d %H:%M:%S'),
                finished_at=datetime.strftime(datetime.fromtimestamp(after.started_at), '%H:%M:%S'),
            ),
            'duration: {value}'.format(
                value=datetime.strftime(datetime.utcfromtimestamp(duration), '%H:%M:%S'),
            ),
            'bias: {value:.2f} [%/h]'.format(
                value=bias * (3600 / duration),
            ),
        ]
        plt.text(
            0.05/2, 0.95,
            '\n'.join(content),
            color='red',
            transform=ax_right.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'number')
        plt.ylabel(r'$\Delta U$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

        #
        return experiment

    return wrapper


def check_total(func: Callable) -> Callable:
    """Check an estimation of experiment's total time."""

    def wrapper(device: Device, params: Sequence[tuple[int, Array[MilliSecond]]], *args, **kwargs):
        if CHECK_TOTAL_FLAG is False:
            return func(device, params, *args, **kwargs)

        # calculate total
        total = 0
        for n_frames, exposure in params:
            total += (n_frames + 1) * np.sum(exposure)  # FIXME: +1 frame
        total = total/1000

        n_exposures = sum([len(exposure) for n_frames, exposure in params])
        total += EXPOSURE_MAX * n_exposures

        #
        while True:
            answer = input('\tTotal time: {total}. Do you want to continue [y/n]?'.format(
                total=datetime.strftime(datetime.utcfromtimestamp(total), '%H:%M:%S'),
            ))
            match answer:
                case 'Y' | 'y' | '':
                    return func(device, params, *args, **kwargs)
                case 'N' | 'n':
                    return None
                case _:
                    print(answer, repr(answer))

    return wrapper


def check_delay(func: Callable) -> Callable:
    """Check a delay between exposure changes."""

    def wrapper(device: Device, params: Sequence[tuple[int, Array[MilliSecond]]], *args, **kwargs):
        if CHECK_EXPOSURE_FLAG is False:
            return func(device, params, *args, **kwargs)

        # calculate exposure_max
        exposure_min = min([min(exposure) for _, exposure in params])
        exposure_max = max([max(exposure) for _, exposure in params])

        #
        if exposure_min < EXPOSURE_MIN:
            raise ValueError('Check a min exposure or change `EXPOSURE_MIN`!')
        if exposure_max > EXPOSURE_MAX:
            raise ValueError('Check a max exposure or change `EXPOSURE_MAX`!')

        return func(device, params, *args, **kwargs)

    return wrapper


@check_delay
@check_total
@check_source
def run_experiment(device: Device, params: Sequence[tuple[int, Sequence[MilliSecond]]], label: str| None = None, force: bool = False) -> None:
    """Run experiment with given params"""

    if label is None:
        label = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    # read or load data
    filedir = os.path.join(DIRECTORY, label)
    filepath = os.path.join(filedir, 'data.pkl')
    if force or not os.path.isfile(filepath):

        # read
        data = Data([], units=device.storage.units, label=label)
        for n_frames, exposure in params:
            tmp = read_data(
                device,
                exposure=exposure,
                n_frames=n_frames,
            )
            data.add(tmp.data)

        # save
        data.save()
