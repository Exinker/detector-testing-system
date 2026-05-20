import numpy as np

from detector_testing_system.characteristic.read_noise import research_read_noise, research_relative_read_noise

from tests.conftest import assert_value_in_3sigma_interval


def test_research_read_noise_for_noiseless_dark_detector(data_model, read_noise):
    data = data_model(
        dark_current=0.0,
        is_noised=False,
        is_lighted=False,
    )

    values = research_read_noise(data)

    np.testing.assert_allclose(
        values,
        np.full(data.n_numbers, read_noise),
    )


def test_research_read_noise_for_noisy_dark_detector_report(data_model, read_noise):
    data = data_model(
        dark_current=0.0,
        is_noised=True,
        is_lighted=False,
        seed=44,
    )

    values = research_read_noise(data)

    assert len(values) == 4096
    assert_value_in_3sigma_interval(values, read_noise)


def test_research_relative_read_noise_for_noiseless_dark_detector(data_model):
    data = data_model(
        dark_current=0.0,
        is_noised=False,
        is_lighted=False,
    )

    relative_read_noise = research_relative_read_noise(data)

    np.testing.assert_allclose(
        relative_read_noise,
        np.zeros(data.n_numbers),
        atol=1e-12,
    )
