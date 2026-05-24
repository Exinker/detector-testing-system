import numpy as np

from detector_testing_system.characteristic.efficiency import calculate_efficiency, research_efficiency
from detector_testing_system.output import Output

from tests.conftest import assert_value_in_3sigma_interval


def test_calculate_efficiency_for_noiseless_signal(data_model, efficiency):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )
    output = Output.create(data=data, n=2)

    value = calculate_efficiency(output, threshold=(0, 100))

    assert np.isclose(value, efficiency)


def test_research_efficiency_for_noisy_detector_report(data_model, efficiency):
    data = data_model(
        is_noised=True,
        is_lighted=True,
    )

    values = research_efficiency(data, threshold=(0, 100))

    assert len(values) == 4096
    assert_value_in_3sigma_interval(values, efficiency)


def test_research_efficiency_for_noiseless_detector(data_model, efficiency):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )

    values = research_efficiency(data, threshold=(0, 100))

    np.testing.assert_allclose(
        values,
        np.full(data.n_numbers, efficiency),
    )


def test_efficiency_returns_nan_for_generated_negative_variance_slope(data_model):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )
    output = Output.create(data=data, n=0)
    output.variance = -0.25 * output.average + 20.0

    efficiency = calculate_efficiency(output, threshold=(0, 100))

    assert np.isnan(efficiency)


def test_research_efficiency_honors_cell_mask(data_model, efficiency):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )

    values = research_efficiency(
        data,
        threshold=(0, 100),
        mask=np.array([True, False, True, False]),
    )

    np.testing.assert_allclose(values[[0, 2]], efficiency)
    assert np.isnan(values[1])
    assert np.isnan(values[3])
