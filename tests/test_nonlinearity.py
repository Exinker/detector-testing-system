import numpy as np

from detector_testing_system.characteristic.nonlinearity import calculate_nonlinearity, research_nonlinearity


def test_calculate_nonlinearity_for_noiseless_linear_signal(data_model):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )
    trace = data.trace(2)

    result = calculate_nonlinearity(trace)

    np.testing.assert_allclose(result.dark_current.xi, np.zeros_like(trace.tau), atol=1e-12)
    assert np.isclose(result.value, 0.0, atol=1e-12)


def test_research_nonlinearity_for_noiseless_linear_detector(data_model):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )

    result = research_nonlinearity(data)

    np.testing.assert_allclose(
        result.value,
        np.zeros(data.n_numbers),
        atol=1e-12,
    )


def test_research_nonlinearity_for_noisy_linear_detector_report(data_model):
    data = data_model(
        is_noised=True,
        is_lighted=True,
    )

    result = research_nonlinearity(data)
    nonlinearity = result.value
    mean = np.mean(nonlinearity)
    sem = np.std(nonlinearity, ddof=1) / np.sqrt(len(nonlinearity))

    assert len(nonlinearity) == 4096
    assert mean + 3 * sem < 1.0
