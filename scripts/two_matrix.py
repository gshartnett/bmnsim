import datetime
import json
import os
from datetime import timezone
from typing import Optional

import fire
import numpy as np

from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.bootstrap import BootstrapSystem
from bmn.brezin import compute_Brezin_energy
from bmn.debug_utils import disable_debug
from bmn.newton_solver import solve_bootstrap


def run_bootstrap(g, L, m=1, verbose=True):

    matrix_system = MatrixSystem(
        operator_basis=["X1", "Pi1", "X2", "Pi2"],
        commutation_rules_concise={
            ("Pi1", "X1"): 1,
            ("Pi2", "X2"): 1,
        },
        hermitian_dict={"Pi1": False, "X1": True, "Pi2": False, "X2": True},
    )

    # scale variables as P = sqrt(N) P', X = sqrt(N) X'
    hamiltonian = SingleTraceOperator(
        data={
            ("Pi1", "Pi1"): -1,
            ("Pi2", "Pi2"): -1,
            ("X1", "X1"): m**2,
            ("X2", "X2"): m**2,
            ("X1", "X2", "X1", "X2"): -g * 2,
            ("X2", "X1", "X2", "X1"): -g * 2,
            ("X1", "X2", "X2", "X1"): g * 2,
            ("X2", "X1", "X1", "X2"): g * 2,
        }
    )

    # <tr G O > = 0 might need to be applied only for O with deg <= L-2
    gauge = MatrixOperator(
        data={
            ("X1", "Pi1"): 1,
            ("Pi1", "X1"): -1,
            ("X2", "Pi2"): 1,
            ("Pi2", "X2"): -1,
            (): 2,
        }
    )

    symmetry_generators = [
        SingleTraceOperator(data={("X1", "Pi2"): 1, ("X2", "Pi1"): -1})
    ]

    bootstrap = BootstrapSystem(
        matrix_system=matrix_system,
        hamiltonian=hamiltonian,
        gauge=gauge,
        max_degree_L=L,
        odd_degree_vanish=True,
        simplify_quadratic=True,
        symmetry_generators=symmetry_generators,
        verbose=verbose,
        save_path=f"data/two_matrix_L_{L}",
    )

    param, optimization_result = solve_bootstrap(
        bootstrap=bootstrap,
        st_operator_to_minimize=bootstrap.hamiltonian,
        init_scale=1e2,
        maxiters=100,
        maxiters_cvxpy=10_000,
        tol=1e-4,
        reg=1e6,
        eps=1e-4,
        radius=1e5,
    )

    energy = bootstrap.get_operator_expectation_value(
        st_operator=hamiltonian, param=param
    )
    x_2 = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(data={("X1", "X1"): 1, ("X2", "X2"): 1}), param=param
    )
    x_4 = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(data={("X1", "X1", "X1", "X1"): 1, ("X2", "X2", "X2", "X2"): 1}), param=param
    )
    p_2 = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(data={("Pi1", "Pi1"): -1, ("Pi2", "Pi2"): -1}), param=param
    )
    p_4 = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(data={("Pi1", "Pi1", "Pi1", "Pi1"): 1, ("Pi2", "Pi2", "Pi2", "Pi2"): 1}), param=param
    )

    result = {
        "energy": energy,
        "x^2": x_2,
        "x^4": x_4,
        "p^2": p_2,
        "p^4": p_4,
    }

    result = result | optimization_result

    return result


if __name__ == "__main__":

    fire.Fire()
