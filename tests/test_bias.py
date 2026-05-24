import numpy as np
from numpy.testing import assert_allclose

from vmk_spectrum3_wrapper.types import U

from detector_testing_system.characteristic.bias import calculate_bias, research_bias

from tests.conftest import assert_mean_close


def test_calculate_bias_noiseless(
    data_model,
    bias: U,
):
    data = data_model(
        is_noised=False,
    )
    n = 0

    result = calculate_bias(
        trace=data.trace(n),
    )

    assert np.isclose(result.value, bias)


def test_research_bias(
    data_model,
    bias: U,
):
    data = data_model(
        is_noised=True,
    )

    result = research_bias(
        data,
    )

    assert_mean_close(result.value, bias, k=3)


def test_research_bias_noiseless(
    data_model,
    bias: U,
):
    data = data_model(
        is_noised=False,
    )

    result = research_bias(
        data,
    )

    assert_allclose(
        result.value,
        np.full(data.n_numbers, bias),
    )
