import numpy as np

from detector_testing_system.characteristic.current import calculate_current, research_current

from tests.conftest import assert_mean_close


def test_calculate_dark_current_for_noiseless_signal(data_model, dark_current):
    data = data_model(
        is_noised=False,
        is_lighted=False,
    )
    trace = data.trace(2)

    result = calculate_current(
        trace,
        model=None,
    )

    assert np.isclose(result.value, dark_current / 1000)


def test_research_dark_current_for_noisy_detector_report(data_model, dark_current):
    data = data_model(
        is_noised=True,
        is_lighted=False,
        seed=43,
        dark_current_spread=2.0,
    )

    result = research_current(data)

    assert len(result.value) == 4096
    assert_mean_close(result.value, dark_current / 1000)


def test_research_dark_current_for_noiseless_detector(data_model, dark_current):
    data = data_model(
        is_noised=False,
        is_lighted=False,
    )

    result = research_current(data)

    np.testing.assert_allclose(
        result.value,
        np.full(data.n_numbers, dark_current / 1000),
    )
