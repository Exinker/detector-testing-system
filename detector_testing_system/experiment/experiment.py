import os
from collections.abc import Sequence
from datetime import datetime
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.types import Array, MilliSecond
from vmk_spectrum3_wrapper.device import Device

from detector_testing_system.experiment import Data, read_data
from detector_testing_system.experiment.config import ExperimentConfig


def check_source(func: Callable) -> Callable:
    """Check a stability of the light source."""

    def wrapper(device: Device, config: ExperimentConfig, *args, **kwargs):
        if config.check_source_flag is False:
            return func(device, config, *args, **kwargs)

        #
        before = read_data(
            device,
            exposure=[config.check_source_tau],
            n_frames=config.check_source_n_frames,
        )
        experiment = func(device, config, *args, **kwargs)
        after = read_data(
            device,
            exposure=[config.check_source_tau],
            n_frames=config.check_source_n_frames,
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
        plt.text(
            0.05/2, 0.95,
            '\n'.join([
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
            ]),
            color='red',
            transform=ax_right.transAxes,
            ha='left', va='top',
        )
        plt.xlabel(r'number')
        plt.ylabel(r'$\Delta U$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

        return experiment

    return wrapper


def check_total(func: Callable) -> Callable:
    """Check an estimation of experiment's total time."""

    def wrapper(device: Device, config: ExperimentConfig, params: Sequence[tuple[int, Array[MilliSecond]]], *args, **kwargs):
        if config.check_total_flag is False:
            return func(device, config, params, *args, **kwargs)

        # calculate total
        total = 0
        for n_frames, exposure in params:
            total += (n_frames + 1) * np.sum(exposure)  # FIXME: +1 frame

        n_exposures = sum([len(exposure) for n_frames, exposure in params])
        total += device._change_exposure_delay * n_exposures

        total = total/1e+3

        #
        while True:
            answer = input('\tTotal time: {total}. Do you want to continue [y/n]?'.format(
                total=datetime.strftime(datetime.utcfromtimestamp(total), '%H:%M:%S'),
            ))
            match answer:
                case 'Y' | 'y' | '':
                    return func(device, config, params, *args, **kwargs)
                case 'N' | 'n':
                    return None
                case _:
                    print(answer, repr(answer))

    return wrapper


def check_exposure(func: Callable) -> Callable:
    """Check an exposure."""

    def wrapper(device: Device, config: ExperimentConfig, params: Sequence[tuple[int, Array[MilliSecond]]], *args, **kwargs):
        if config.check_exposure_flag is False:
            return func(device, config, params, *args, **kwargs)

        # calculate exposure_max
        exposure_min = min([min(exposure) for _, exposure in params])
        exposure_max = max([max(exposure) for _, exposure in params])

        #
        if exposure_min < config.check_exposure_min:
            raise ValueError('Check a min exposure or change `check_exposure_min`!')
        if exposure_max > config.check_exposure_max:
            raise ValueError('Check a max exposure or change `check_exposure_max`!')

        return func(device, config, params, *args, **kwargs)

    return wrapper


@check_exposure
@check_total
@check_source
def run_experiment(device: Device, config: ExperimentConfig, params: Sequence[tuple[int, Sequence[MilliSecond]]], label: str | None = None, force: bool = False) -> None:
    """Run experiment with given params"""

    if label is None:
        label = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    # read or load data
    filedir = os.path.join('.', 'data', label)
    filepath = os.path.join(filedir, 'data.pkl')
    if force or not os.path.isfile(filepath):

        # read
        data = Data([], units=device.storage.units, label=label)
        for n_frames, exposure in params:
            dat = read_data(
                device,
                exposure=exposure,
                n_frames=n_frames,
            )
            data.add(dat.data)

        # save
        data.save()
