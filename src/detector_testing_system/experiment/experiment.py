from collections.abc import Sequence
from datetime import datetime
from functools import wraps
import itertools
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from vmk_spectrum3_wrapper.device import Device
from vmk_spectrum3_wrapper.types import Array, MilliSecond

from detector_testing_system.experiment.config import ExperimentConfig
from detector_testing_system.experiment.data import Data, read_data


def check_source(func: Callable) -> Callable:
    """Check a stability of the light source."""

    @wraps(func)
    def wrapper(device: Device, config: ExperimentConfig, *args, **kwargs):
        if config.check_source_flag is False:
            return func(device, config, *args, **kwargs)

        before = read_data(
            device,
            exposure=[config.check_source_tau],
            n_frames=config.check_source_n_frames,
            verbose=False,
        )
        experiment = func(device, config, *args, **kwargs)
        after = read_data(
            device,
            exposure=[config.check_source_tau],
            n_frames=config.check_source_n_frames,
            verbose=False,
        )

        duration = after.started_at - before.started_at
        bias = np.mean(after.average - before.average)

        if config.check_source_show:
            fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)

            plt.sca(ax_left)
            plt.plot(
                before.average.squeeze(),
                linestyle='none', marker='.', markersize=2,
                label='before',
            )
            plt.plot(
                after.average.squeeze(),
                linestyle='none', marker='.', markersize=2,
                label='after',
            )
            plt.xlabel(r'number')
            plt.ylabel(r'$U$')
            plt.grid(color='grey', linestyle=':')
            plt.legend()

            plt.sca(ax_right)
            plt.plot(
                (after.average - before.average).squeeze(),
                linestyle='none', marker='.', markersize=2,
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
                        value=bias * (60*60 / duration),
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

    @wraps(func)
    def wrapper(device: Device, config: ExperimentConfig, params: Sequence[tuple[int, Array[MilliSecond]]], *args, **kwargs):
        if config.check_total_flag is False:
            return func(device, config, params, *args, **kwargs)

        total = 0
        for n_frames, exposure in params:
            total += (n_frames + 1) * np.sum(exposure)  # FIXME: +1 frame
        n_exposures = sum([len(exposure) for n_frames, exposure in params])
        total += device.config.change_exposure_timeout * n_exposures
        total = total/1e+3

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
    """Check an exposure time."""

    @wraps(func)
    def wrapper(device: Device, config: ExperimentConfig, params: Sequence[tuple[int, Array[MilliSecond]]], *args, **kwargs):
        if config.check_exposure_flag is False:
            return func(device, config, params, *args, **kwargs)

        if min([min(exposure) for _, exposure in params]) < config.check_exposure_min:
            raise ValueError('Check a min exposure or change `check_exposure_min`!')
        if max([max(exposure) for _, exposure in params]) > config.check_exposure_max:
            raise ValueError('Check a max exposure or change `check_exposure_max`!')
        return func(device, config, params, *args, **kwargs)

    return wrapper


@check_exposure
@check_total
@check_source
def run_experiment(
    device: Device,
    config: ExperimentConfig,
    params: Sequence[tuple[int, Sequence[MilliSecond]]],
    label: str | None = None,
    force: bool = False,
) -> None:
    """Run experiment with given params."""

    if label is None:
        label = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    filedir = os.path.join('.', 'data', label)
    filepath = os.path.join(filedir, 'data.pkl')
    if force or not os.path.isfile(filepath):

        data = []
        for n_frames, exposure in params:
            _data = read_data(
                device=device,
                exposure=exposure,
                n_frames=n_frames,
                verbose=True,
            )
            data.append(_data)
        data = Data.create(tuple(itertools.chain(*data)), label=label)

        data.save()
