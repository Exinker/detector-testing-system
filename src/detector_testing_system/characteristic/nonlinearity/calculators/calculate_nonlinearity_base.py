import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from vmk_spectrum3_wrapper.types import Array, U

from detector_testing_system.characteristic.dark_current.models import BaseDarkCurrentModel
from detector_testing_system.characteristic.nonlinearity.results import NonlinearityResultABC
from detector_testing_system.data import Trace
from detector_testing_system.types import AxesView


class BaseNonlinearityResult(NonlinearityResultABC):

    def _plot_left(
        self,
        ax: Axes,
        view: AxesView | None,
        color: str | None = None,
        verbose: bool = False,
        hat_label: str | None = r'$\hat{U}$',
    ) -> None:
        view = view or {}
        color = color or 'red'

        trace = self.dark_current.trace
        mask = self.dark_current.mask

        plt.sca(ax)
        plt.scatter(
            trace.tau, trace.u,
            c='grey', s=10,
        )
        plt.scatter(
            trace.tau[mask], trace.u[mask],
            c=color, s=10,
            label=rf'$U_{{{trace.n}}}$',
        )
        plt.plot(
            trace.tau, self.dark_current.interpolate(trace.tau),
            color='black', linestyle='solid', linewidth=1,
            label=hat_label,
        )
        if verbose:
            plt.text(
                0.95, 0.05/2,
                '\n'.join([
                    fr'$a = {{{self.dark_current.value:.4f}}}$',
                    fr'$b = {{{self.dark_current.bias:.4f}}}$',
                ]),
                transform=ax.transAxes,
                ha='right', va='bottom',
            )

        plt.xlabel(r'$\tau$ [ms]')
        plt.ylabel(r'$U$ {units}'.format(units=trace.units.label))
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        ax.set(**view)

    def _plot_right(
        self,
        ax: Axes,
        view: AxesView | None,
        color: str | None = None,
        verbose: bool = False,
    ) -> None:
        view = view or {}
        color = color or 'red'

        trace = self.dark_current.trace
        mask = self.dark_current.mask

        plt.sca(ax)
        plt.scatter(
            trace.u, self.dark_current.xi,
            c='grey', s=10,
        )
        plt.scatter(
            trace.u[mask], self.dark_current.xi[mask],
            c=color, s=10,
            label=rf'$U_{{{trace.n}}}$',
        )
        if verbose:
            plt.text(
                0.95, 0.95,
                '\n'.join([
                    trace.label.prefix,
                    fr'$\alpha: {{{self.value:.2f}}}$ [%]',
                ]),
                transform=ax.transAxes,
                ha='right', va='top',
            )
            plt.text(
                0.95, 0.05/2,
                '\n'.join([
                    r'$error = 100\frac{\hat{U} - U_{i}}{a \tau}$',
                ]),
                transform=ax.transAxes,
                ha='right', va='bottom',
            )

        plt.xlabel(r'$U$ {units}'.format(units=trace.units.label))
        plt.ylabel(r'$error$ [%]')

        plt.grid(color='grey', linestyle=':')

        ax.set(**view)


def calculate_nonlinearity_base(
    trace: Trace,
    model: BaseDarkCurrentModel,
) -> BaseNonlinearityResult:

    if not model.weighted:
        raise ValueError('To calculate nonlinearity use weighted model only!')

    dark_current = model.fit(trace)
    alpha = _calculate_alpha(xi=dark_current.xi)

    return BaseNonlinearityResult(
        dark_current=dark_current,
        value=alpha,
    )


def _calculate_alpha(xi: Array[U]) -> float:
    """Calculate nonlinearity coefficient (alpha)"""
    return (np.max(xi) - np.min(xi)) / 2
