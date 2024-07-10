import fire
import numpy as np

from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.bootstrap import BootstrapSystem
from bmn.debug_utils import disable_debug
from bmn.solver import minimize


def run(nu, L):

    matrix_system = MatrixSystem(
        operator_basis=["X0", "X1", "X2", "P0", "P1", "P2"],
        commutation_rules_concise={
            ("P0", "X0"): -1j,
            ("P1", "X1"): -1j,
            ("P2", "X2"): -1j,
        },
        hermitian_dict={
            "P0": True,
            "P1": True,
            "P2": True,
            "X0": True,
            "X1": True,
            "X2": True,
        },
    )

    # scale variables as P = sqrt(N) P', X = sqrt(N) X'
    hamiltonian = SingleTraceOperator(
        data={
            # kinetic term
            ("P0", "P0"): 1 / 2,
            ("P1", "P1"): 1 / 2,
            ("P2", "P2"): 1 / 2,
            # quadratic term
            ("X0", "X0"): nu**2 / 2,
            ("X1", "X1"): nu**2 / 2,
            ("X2", "X2"): nu**2 / 2,
            # cubic term
            ("X0", "X1", "X2"): 6 * 1j * nu,
            # quadratic term (XY)
            ("X0", "X1", "X0", "X1"): -1 / 4,
            ("X1", "X0", "X1", "X0"): -1 / 4,
            ("X0", "X1", "X1", "X0"): 1 / 4,
            ("X1", "X0", "X0", "X1"): 1 / 4,
            # quadratic term (XZ) TODO check sign
            ("X0", "X2", "X0", "X2"): -1 / 4,
            ("X2", "X0", "X2", "X0"): -1 / 4,
            ("X0", "X2", "X2", "X0"): 1 / 4,
            ("X2", "X0", "X0", "X2"): 1 / 4,
            # quadratic term (YZ)
            ("X1", "X2", "X1", "X2"): -1 / 4,
            ("X2", "X1", "X2", "X1"): -1 / 4,
            ("X1", "X2", "X2", "X1"): 1 / 4,
            ("X2", "X1", "X1", "X2"): 1 / 4,
        }
    )

    gauge = MatrixOperator(
        data={
            ("X0", "P0"): 1j,
            ("P0", "X0"): -1j,
            ("X1", "P1"): 1j,
            ("P1", "X1"): -1j,
            ("X2", "P2"): 1j,
            ("P2", "X2"): -1j,
            (): 3,
        }
    )

    bootstrap = BootstrapSystem(
        matrix_system=matrix_system,
        hamiltonian=hamiltonian,
        gauge=gauge,
        half_max_degree=L,
        odd_degree_vanish=False,
        simplify_quadratic=False,
    )

    bootstrap.get_null_space_matrix()

    disable_debug()

    param, success = minimize(
        bootstrap=bootstrap,
        op=bootstrap.hamiltonian,
        init_scale=1e2,
        verbose=False,
        maxiters=25,
        reg=5e-4,
        eps=5e-4,
    )

    energy = bootstrap.get_operator_expectation_value(
        st_operator=hamiltonian,
        param=param
        )

    print(f"problem success: {success}, min energy found: {energy:.6f}")
    return success, energy, param


if __name__ == "__main__":

    success, energy, param = fire.Fire(run)
