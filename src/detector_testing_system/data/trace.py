from dataclasses import dataclass

from vmk_spectrum3_wrapper.types import Array, MilliSecond, Number, U
from vmk_spectrum3_wrapper.units import Units

from detector_testing_system.data.label import Label


@dataclass
class Trace:

    u: Array[U]
    variance: Array[U]
    tau: Array[MilliSecond]
    n: Number
    label: Label
    units: Units
