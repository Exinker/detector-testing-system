import numpy as np

from detector_testing_system.characteristic.efficiency import (
    calculate_efficiency,
    research_efficiency,
)
from detector_testing_system.data import filter_trace_factory

from tests.conftest import assert_mean_close


def test_calculate_efficiency_for_noiseless_signal(data_model, efficiency):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )
    trace = data.trace(2)

    result = calculate_efficiency(
        trace=trace,
        filter=filter_trace_factory(
            threshold=(0, 100),
        ),
    )

    assert np.isclose(result.value, efficiency)


def test_research_efficiency_for_noisy_detector_report(data_model, efficiency):
    data = data_model(
        is_noised=True,
        is_lighted=True,
    )

    result = research_efficiency(
        data=data,
        filter=filter_trace_factory(
            threshold=(0, 100),
        ),
    )

    assert len(result.value) == 4096
    assert_mean_close(result.value, efficiency)


def test_research_efficiency_for_noiseless_detector(data_model, efficiency):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )

    result = research_efficiency(
        data=data,
        filter=filter_trace_factory(
            threshold=(0, 100),
        ),
    )

    np.testing.assert_allclose(
        result.value,
        np.full(data.n_numbers, efficiency),
    )


def test_efficiency_returns_nan_for_generated_negative_variance_slope(data_model):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )
    trace = data.trace(0)
    trace.variance = -0.25 * trace.u + 20.0

    result = research_efficiency(
        data=data,
        filter=filter_trace_factory(
            threshold=(0, 100),
        ),
    )

    assert np.all(np.isnan(result.value))


def test_research_efficiency_honors_cell_mask(data_model, efficiency):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )

    result = research_efficiency(
        data=data,
        filter=filter_trace_factory(
            threshold=(0, 100),
        ),
        mask=np.array([True, False, True, False]),
    )

    np.testing.assert_allclose(result.value[[0, 2]], efficiency)
    assert np.isnan(result.value[1])
    assert np.isnan(result.value[3])
