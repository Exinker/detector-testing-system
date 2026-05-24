from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt

from vmk_spectrum3_wrapper.types import Array, MilliSecond, Number, U
from vmk_spectrum3_wrapper.units import Units

from detector_testing_system.data.label import Label
from detector_testing_system.types import AxesView


TraceFilter = Callable[['Trace'], Array[bool]]


@dataclass
class Trace:

    u: Array[U]
    variance: Array[U]
    tau: Array[MilliSecond]
    n: Number
    label: Label
    units: Units

    def show(
        self,
        view: AxesView | None,
        verbose: bool = False,
    ) -> None:
        view = view or {}

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.scatter(
            self.tau, self.u,
            c='red', s=10,
            label=rf'$U_{{{self.n}}}$',
        )
        if verbose:
            plt.text(
                0.05/2, 0.95,
                '\n'.join([
                ]),
                transform=ax.transAxes,
                ha='left', va='top',
            )

        plt.xlabel(r'$\tau$ {units}'.format(units=r'[$ms$]'))
        plt.ylabel(r'$U$ {units}'.format(units=self.units.label))

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)

        plt.show()
