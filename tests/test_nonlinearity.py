import numpy as np

from detector_testing_system.characteristic.nonlinearity import calculate_nonlinearity, research_nonlinearity
from detector_testing_system.output import Output


def test_calculate_nonlinearity_for_noiseless_linear_signal(data_model):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )
    output = Output.create(data=data, n=2)

    xi, alpha = calculate_nonlinearity(output)

    np.testing.assert_allclose(xi, np.zeros_like(output.exposure), atol=1e-12)
    assert np.isclose(alpha, 0.0, atol=1e-12)


def test_research_nonlinearity_for_noiseless_linear_detector(data_model):
    data = data_model(
        is_noised=False,
        is_lighted=True,
    )

    nonlinearity = research_nonlinearity(data)

    np.testing.assert_allclose(
        nonlinearity,
        np.zeros(data.n_numbers),
        atol=1e-12,
    )


def test_research_nonlinearity_for_noisy_linear_detector_report(data_model):
    data = data_model(
        is_noised=True,
        is_lighted=True,
    )

    nonlinearity = research_nonlinearity(data)
    mean = np.mean(nonlinearity)
    sem = np.std(nonlinearity, ddof=1) / np.sqrt(len(nonlinearity))

    assert len(nonlinearity) == 4096
    assert mean + 3 * sem < 1.0
