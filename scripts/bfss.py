import fire
from typing import Optional
import numpy as np
import os
import json
from datetime import timezone
import datetime
from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.bootstrap import BootstrapSystem
from bmn.brezin import (
    compute_Brezin_energy,
)
from bmn.debug_utils import disable_debug
from bmn.newton_solver import minimize as minimize_newton


def run_bootstrap(
    energy: float, L: int, verbose:bool=False
    ) -> tuple[bool, float, np.ndarray]:
    """
    Perform the bootstrap optimization for a single instance of the model.

    Parameters
    ----------
    energy : float
        The energy of the states we are interested in.
    L : int
        The bootstrap degree
    verbose : bool, optional
        Verbose mode, by default False

    Returns
    -------
    tuple[bool, float, np.ndarray]
        _description_
    """

    matrix_system = MatrixSystem(
        operator_basis=["X0", "X1", "X2", "Pi0", "Pi1", "Pi2"],
        commutation_rules_concise={
            ("Pi0", "X0"): 1,  # use Pi' = i P to ensure reality
            ("Pi1", "X1"): 1,
            ("Pi2", "X2"): 1,
        },
        hermitian_dict={
            "Pi0": False,
            "X0": True,
            "Pi1": False,
            "X1": True,
            "Pi2": False,
            "X2": True,
            },
    )

    # lambda = 1 here
    hamiltonian = SingleTraceOperator(
        data={
            ("Pi0", "Pi0"): -0.5,
            ("Pi1", "Pi1"): -0.5,
            ("Pi2", "Pi2"): -0.5,
            # quartic term (XY)
            ("X0", "X1", "X0", "X1"): -1 / 4,
            ("X1", "X0", "X1", "X0"): -1 / 4,
            ("X0", "X1", "X1", "X0"): 1 / 4,
            ("X1", "X0", "X0", "X1"): 1 / 4,
            # quartic term (XZ)
            ("X0", "X2", "X0", "X2"): -1 / 4,
            ("X2", "X0", "X2", "X0"): -1 / 4,
            ("X0", "X2", "X2", "X0"): 1 / 4,
            ("X2", "X0", "X0", "X2"): 1 / 4,
            # quartic term (YZ)
            ("X1", "X2", "X1", "X2"): -1 / 4,
            ("X2", "X1", "X2", "X1"): -1 / 4,
            ("X1", "X2", "X2", "X1"): 1 / 4,
            ("X2", "X1", "X1", "X2"): 1 / 4,
        }
    )

    # <tr G O > = 0
    gauge = MatrixOperator(data={
        ("X0", "Pi0"): 1,
        ("Pi0", "X0"): -1,
        ("X1", "Pi1"): 1,
        ("Pi1", "X1"): -1,
        ("X2", "Pi2"): 1,
        ("Pi2", "X2"): -1,
        (): 3,
        })

    bootstrap = BootstrapSystem(
        matrix_system=matrix_system,
        hamiltonian=hamiltonian,
        gauge=gauge,
        max_degree_L=L,
        odd_degree_vanish=True,
        simplify_quadratic=True,
    )

    bootstrap.get_null_space_matrix()

    disable_debug()

    param, success = minimize_newton(
        bootstrap=bootstrap,
        op=bootstrap.hamiltonian,
        init_scale=1e2,
        verbose=verbose,
        maxiters=10,
    )

    energy = bootstrap.get_operator_expectation_value(
        st_operator=hamiltonian, param=param
    )

    x_squared = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(data={("X0", "X0"): 1, ("X1", "X1"): 1, ("X2", "X2"): 1}), param=param
    )

    p_squared = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(data={("Pi0", "Pi0"): -1, ("Pi1", "Pi1"): -1, ("Pi2", "Pi2"): -1}), param=param
    )

    x_4 = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(data={("X0", "X0", "X0", "X0"): 1, ("X1", "X1", "X1", "X1"): 1, ("X2", "X2", "X2", "X2"): 1}), param=param
    )

    expectation_values = {
        "energy": energy,
        "x_squared": x_squared,
        "p_squared": p_squared,
        "x_4": x_4,
    }

    return success, expectation_values, param


def scan_bootstrap(L, verbose=False):

    path = f"data/bfss_L_{L}"
    if not os.path.exists(path):
        os.makedirs(path)

    n_grid = 20
    g4_max = 16
    g6_max = 16

    g2_values = [-1, 1]
    g4_values = np.concatenate((np.linspace(-g4_max, 0, n_grid), np.linspace(0, g4_max, n_grid)[1:]))
    g6_values = np.linspace(0, g6_max, n_grid)

    for g2 in g2_values:
        for g4 in g4_values:
            for g6 in g6_values:

                # only run models with bounded-by-below potentials
                if (g6 > 0) or (g4 > 0):

                    # get the current UTC timestamp
                    timestamp = datetime.datetime.now(timezone.utc).replace(tzinfo=timezone.utc).timestamp()
                    timestamp = int(1e6 * timestamp)

                    print(f"\n\n solving problem with g2 = {g2}, g4 = {g4}, g6 = {g6} \n\n")

                    success, expectation_values, param = run_bootstrap(
                        g2=g2,
                        g4=g4,
                        g6=g6,
                        L=L,
                        verbose=verbose
                        )

                    # record results
                    result = {
                        "g2": g2,
                        "g4": g4,
                        "g6": g6,
                        "success": success,
                        "param": list(param),
                    }
                    result = result | expectation_values

                    # print(f"Completed run for g={g}, success={success}, energy={energy}")
                    with open(f"{path}/{timestamp}.json", 'w') as f:
                        json.dump(result, f)

if __name__ == "__main__":

    fire.Fire()