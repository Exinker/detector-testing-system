import numpy as np

from detector_testing_system.characteristic.dark_current import calculate_dark_current, research_dark_current
from detector_testing_system.output import Output

from tests.conftest import assert_value_in_3sigma_interval


def test_calculate_dark_current_for_noiseless_signal(data_model, dark_current):
    data = data_model(
        is_noised=False,
        is_lighted=False,
    )
    output = Output.create(data=data, n=2)

    value = calculate_dark_current(output, threshold=(0, 100))

    assert np.isclose(value, dark_current)


def test_research_dark_current_for_noisy_detector_report(data_model, dark_current):
    data = data_model(
        is_noised=True,
        is_lighted=False,
        seed=43,
        dark_current_spread=2.0,
    )

    values = research_dark_current(data, threshold=(0, 100))

    assert len(values) == 4096
    assert_value_in_3sigma_interval(values, dark_current)


def test_research_dark_current_for_noiseless_detector(data_model, dark_current):
    data = data_model(
        is_noised=False,
        is_lighted=False,
    )

    values = research_dark_current(data, threshold=(0, 100))

    np.testing.assert_allclose(
        values,
        np.full(data.n_numbers, dark_current),
    )
