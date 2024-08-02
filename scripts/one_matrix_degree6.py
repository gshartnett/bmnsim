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
from bmn.solver import minimize as minimize_old


def run_bootstrap(
    g2: float, g4: float, g6: float, L: int, verbose:bool=False
    ) -> tuple[bool, float, np.ndarray]:
    """
    Perform the bootstrap optimization for a single instance of the model.

    Parameters
    ----------
    g2 : float
        The g2 parameter
    g4 : float
        The g4 parameter
    g6 : float
        The g6 parameter
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
        operator_basis=["X", "Pi"],
        commutation_rules_concise={
            ("Pi", "X"): 1,  # use Pi' = i P to ensure reality
        },
        hermitian_dict={"Pi": False, "X": True},
    )

    hamiltonian = SingleTraceOperator(
        data={
            ("Pi", "Pi"): -0.5,
            ("X", "X"): 0.5 * g2,
            ("X", "X", "X", "X"): g4 / 4,
            ("X", "X", "X", "X", "X", "X"): g6 / 6,
        }
    )

    # <tr G O > = 0 might need to be applied only for O with deg <= L-2
    gauge = MatrixOperator(data={("X", "Pi"): 1, ("Pi", "X"): -1, (): 1})

    bootstrap = BootstrapSystem(
        matrix_system=matrix_system,
        hamiltonian=hamiltonian,
        gauge=gauge,
        max_degree_L=L,
        odd_degree_vanish=True,
        simplify_quadratic=True,
        verbose=False,
        save_path=f"data/one_matrix_degree_6_L_{L}",
    )

    #bootstrap.get_null_space_matrix()

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
    x_2 = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(data={("X", "X"): 1}), param=param
    )
    x_4 = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(data={("X", "X", "X", "X"): 1}), param=param
    )
    p_2 = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(data={("Pi", "Pi"): -1}), param=param
    )
    p_4 = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(data={("Pi", "Pi", "Pi", "Pi"): 1}), param=param
    )

    expectation_values = {
        "energy": energy,
        "x^2": x_2,
        "x^4": x_4,
        "p^2": p_2,
        "p^4": p_4,
    }

    # compare against Brezin et al (if applicable)
    if g2 == 1 and g6 == 0:
        exact_energy = compute_Brezin_energy( g4 / 4)
        print(
            f"problem success: {success}, min energy found: {energy:.6f}, exact (L=inf) value = {exact_energy:.6f}"
        )
        print(f"energy error = {energy - exact_energy:.4e}")

    return success, expectation_values, param


def scan_bootstrap(L, verbose=False):

    path = f"data/one_matrix_degree_6_L_{L}"
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
