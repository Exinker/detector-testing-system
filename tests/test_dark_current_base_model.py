from types import SimpleNamespace

import numpy as np
import pytest

from detector_testing_system.characteristic.current import (
    BaseCurrentModel,
    JNormCurrentModel,
)
from detector_testing_system.data import Trace, filter_trace_factory
from detector_testing_system.experiment import FitError


def create_trace() -> Trace:
    return Trace(
        u=np.array([-1., 2., 5., 12.]),
        variance=np.zeros(4),
        tau=np.array([1., 2., 3., 4.]),
        n=0,
        label=SimpleNamespace(),
        units=SimpleNamespace(value_max=10),
    )


def create_jnorm_trace() -> Trace:
    tau = np.array([1., 2., 3., 4., 5., 6.])
    return Trace(
        u=2 * tau + 1,
        variance=np.zeros(6),
        tau=tau,
        n=0,
        label=SimpleNamespace(),
        units=SimpleNamespace(value_max=20),
    )


def test_base_dark_current_model_uses_default_filter() -> None:
    trace = create_trace()
    expected = filter_trace_factory()(trace)

    result = BaseCurrentModel(weighted=False).fit(trace)

    np.testing.assert_array_equal(result.mask, expected)


def test_base_dark_current_model_uses_custom_filter() -> None:
    trace = create_trace()
    filter = filter_trace_factory(
        threshold=(0, 10),
        interval=(2, 3),
    )

    result = BaseCurrentModel(
        weighted=False,
        filter=filter,
    ).fit(trace)

    np.testing.assert_array_equal(result.mask, filter(trace))


def test_base_dark_current_model_raises_for_empty_filter_result() -> None:
    trace = create_trace()
    filter = filter_trace_factory(
        threshold=(100, 200),
    )

    with pytest.raises(FitError):
        BaseCurrentModel(filter=filter).fit(trace)


def test_jnorm_dark_current_model_uses_custom_filter() -> None:
    trace = create_jnorm_trace()
    filter = filter_trace_factory(
        interval=(3, 5),
    )

    result = JNormCurrentModel(
        epsilon=0,
        min_points=2,
        filter=filter,
    ).fit(trace)

    expected = filter(trace)
    np.testing.assert_array_equal(result.mask, expected)


def test_jnorm_dark_current_model_raises_when_filter_has_too_few_points() -> None:
    trace = create_jnorm_trace()
    filter = filter_trace_factory(
        interval=(3, 3),
    )

    with pytest.raises(FitError):
        JNormCurrentModel(
            epsilon=0,
            min_points=2,
            filter=filter,
        ).fit(trace)
