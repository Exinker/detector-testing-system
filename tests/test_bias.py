import numpy as np

from detector_testing_system.characteristic.bias import calculate_bias, research_bias
from detector_testing_system.data import Trace

from tests.conftest import assert_value_in_3sigma_interval


def test_calculate_bias_for_noiseless_signal(data_model, bias):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )
    n = 0
    trace = data.trace(n)

    value = calculate_bias(trace, threshold=(0, 100))

    assert np.isclose(value, bias)


def test_research_bias_for_noisy_detector_report(data_model, bias):
    data = data_model(
        is_noised=True,
        is_lighted=True,
    )

    values = research_bias(data, threshold=(0, 100))

    assert len(values) == 4096
    assert_value_in_3sigma_interval(values, bias)


def test_research_bias_for_noiseless_detector(data_model, bias):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )

    values = research_bias(data, threshold=(0, 100))

    np.testing.assert_allclose(
        values,
        np.full(data.n_numbers, bias),
    )
