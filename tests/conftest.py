import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from faker import Faker

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'

if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def _stub_module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


if importlib.util.find_spec('pyspectrum3') is None:
    _pyspectrum3_names = (
        'AssemblyConfigFile',
        'AssemblyContext',
        'AssemblyInfo',
        'AssemblyParams',
        'AssemblyStatus',
        'AsyncDriverException',
        'AutoConfig',
        'ChipType',
        'ConfigFileParser',
        'DefaultCopyPipeFilter',
        'DeviceConfigFile',
        'DeviceManager',
        'DriverConfig',
        'DriverException',
        'ExceptionProducer',
        'ExceptionType',
        'Exposure',
        'FrameState',
        'Measurement',
        'PipeFilter',
        'SwapFile',
    )
    _stub_module(
        'pyspectrum3',
        VERSION='test',
        **{
            name: (
                type(name, (Exception,), {})
                if name.endswith('Exception')
                else type(name, (), {})
            )
            for name in _pyspectrum3_names
        },
    )

if importlib.util.find_spec('cmake_example') is None:
    _stub_module('cmake_example', VERSION='test')

if importlib.util.find_spec('vmk_spectrum3') is None:
    _stub_module('vmk_spectrum3', VERSION='test')

if importlib.util.find_spec('vmk_spectrum3_wrapper') is None:
    class _Unit:
        def __init__(self, label, value_max):
            self.label = label
            self.value_max = value_max

        def __str__(self):
            return f'Units.{self.label}'

    class _Units:
        digit = _Unit('digit', 4096)
        percent = _Unit('percent', 100)
        electron = _Unit('electron', 1)

    wrapper = _stub_module('vmk_spectrum3_wrapper', VERSION='test')
    types = _stub_module('vmk_spectrum3_wrapper.types', Array=list, MilliSecond=float)
    units = _stub_module('vmk_spectrum3_wrapper.units', U=float, Units=_Units)
    device = _stub_module('vmk_spectrum3_wrapper.device', Device=object)
    detector = _stub_module('vmk_spectrum3_wrapper.detector', Detector=object)
    config = _stub_module('vmk_spectrum3_wrapper.config', DEFAULT_DETECTOR=SimpleNamespace(config=SimpleNamespace(n_pixels=1)))
    filters = _stub_module(
        'vmk_spectrum3_wrapper.measurement_manager.filters',
        ClipFilter=object,
        PipeFilter=object,
        ScaleFilter=object,
    )
    measurement_manager = _stub_module('vmk_spectrum3_wrapper.measurement_manager', filters=filters)
    wrapper.types = types
    wrapper.units = units
    wrapper.device = device
    wrapper.detector = detector
    wrapper.config = config
    wrapper.measurement_manager = measurement_manager

from vmk_spectrum3_wrapper.units import Units

from detector_testing_system.data import Data, Datum


@dataclass
class DataModel:

    exposure: np.ndarray
    bias: float
    efficiency: float
    read_noise: float
    dark_current: float
    light_current: float = 0.0
    n_pixels: int = 4
    n_frames: int = 64
    units: object = Units.percent
    seed: int | None = None
    bias_spread: float = 0.0
    efficiency_spread: float = 0.0
    read_noise_spread: float = 0.0
    dark_current_spread: float = 0.0

    def generate(self, label: str = 'synthetic spectra') -> Data:
        rng = np.random.default_rng(self.seed) if self.seed is not None else None
        parameters = self._pixel_parameters(rng)
        data = []

        for index, tau in enumerate(self.exposure):
            average = self._average(tau, parameters)
            variance = self._variance(average, parameters)
            intensity = self._intensity(average, variance, rng)

            data.append(Datum.create(
                intensity=intensity,
                tau=float(tau),
                n_frames=self.n_frames,
                started_at=float(index),
                units=self.units,
            ))

        return Data(data, label=label)

    def _pixel_parameters(self, rng: np.random.Generator | None) -> dict[str, np.ndarray]:
        return {
            'bias': self._pixel_values(self.bias, self.bias_spread, rng),
            'efficiency': self._pixel_values(self.efficiency, self.efficiency_spread, rng),
            'read_noise': self._pixel_values(self.read_noise, self.read_noise_spread, rng),
            'dark_current': self._pixel_values(self.dark_current, self.dark_current_spread, rng),
        }

    def _pixel_values(self, center: float, spread: float, rng: np.random.Generator | None) -> np.ndarray:
        if spread == 0 or rng is None:
            return np.full(self.n_pixels, center, dtype=float)

        values = rng.normal(loc=center, scale=spread, size=self.n_pixels)
        return values - np.mean(values) + center

    def _average(self, tau: float, parameters: dict[str, np.ndarray]) -> np.ndarray:
        return (
            parameters['bias']
            + self.light_current * tau
            + parameters['dark_current'] * tau / 1000
        )

    def _variance(self, average: np.ndarray, parameters: dict[str, np.ndarray]) -> np.ndarray:
        photo_signal = np.maximum(average - parameters['bias'], 0)
        return parameters['read_noise']**2 + photo_signal / parameters['efficiency']

    def _intensity(
        self,
        average: np.ndarray,
        variance: np.ndarray,
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        if rng is not None:
            return rng.normal(
                loc=average,
                scale=np.sqrt(variance),
                size=(self.n_frames, self.n_pixels),
            )

        frames = _normalized_frames(self.n_frames)
        return average + frames[:, np.newaxis] * np.sqrt(variance)


def _normalized_frames(n_frames: int) -> np.ndarray:
    frames = np.linspace(-1, 1, n_frames)
    return (frames - np.mean(frames)) / np.std(frames, ddof=1)


def assert_mean_close(values: np.ndarray, value: float, k: int = 3) -> None:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(len(values))

    assert np.isclose(value, mean, atol=k*sem)


def _fake_float(seed: int, min_value: float, max_value: float) -> float:
    fake = Faker()
    fake.seed_instance(seed)

    return float(fake.pyfloat(
        left_digits=2,
        right_digits=2,
        min_value=min_value,
        max_value=max_value,
    ))


@pytest.fixture
def bias() -> float:
    return _fake_float(seed=1, min_value=2.0, max_value=8.0)


@pytest.fixture
def efficiency() -> float:
    return _fake_float(seed=2, min_value=4.0, max_value=8.0)


@pytest.fixture
def read_noise() -> float:
    return _fake_float(seed=3, min_value=0.3, max_value=1.2)


@pytest.fixture
def dark_current() -> float:
    return _fake_float(seed=4, min_value=5.0, max_value=20.0)


@pytest.fixture
def light_current() -> float:
    return _fake_float(seed=5, min_value=6.0, max_value=9.0)


@pytest.fixture
def exposure() -> np.ndarray:
    return np.arange(1, 17, dtype=float)


@pytest.fixture
def data_model(bias, efficiency, read_noise, dark_current, light_current, exposure):

    def inner(
        *,
        is_noised: bool = True,
        is_lighted: bool = True,
        **overrides,
    ) -> Data:
        parameters = {
            'exposure': exposure,
            'bias': bias,
            'efficiency': efficiency,
            'read_noise': read_noise,
            'dark_current': dark_current,
            'light_current': light_current if is_lighted else 0.0,
        }

        if is_noised:
            parameters.update({
                'n_pixels': 4096,
                'n_frames': 256,
                'seed': 42,
                'bias_spread': 0.5,
                'efficiency_spread': 0.5,
                'read_noise_spread': 0.1,
            })

        parameters.update(overrides)

        return DataModel(**parameters).generate()

    return inner
