from dataclasses import dataclass

import numpy as np

from vmk_spectrum3_wrapper.types import Array, MilliSecond, Number, U
from vmk_spectrum3_wrapper.units import Units


@dataclass
class Trace:

    u: Array[U]
    variance: Array[U]
    tau: Array[MilliSecond]
    n: Number
    label: str
    units: Units
