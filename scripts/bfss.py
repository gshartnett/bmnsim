import datetime
import json
import os
from datetime import timezone
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
import tqdm
import fire
import numpy as np

from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.bootstrap import BootstrapSystem
from bmn.brezin import compute_Brezin_energy
from bmn.newton_solver import solve_bootstrap


def run_bootstrap(
    energy: float,
    L: int,
    st_operator_to_bound: SingleTraceOperator=SingleTraceOperator(data={("X0", "X0"): 1, ("X1", "X1"): 1, ("X2", "X2"): 1}),
    bound_direction: str = 'lower',
    verbose: bool = True,
    path_suffix: Optional[str] = None,
    ) -> tuple[float, np.ndarray]:
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

    # get the current UTC timestamp
    timestamp = (
        datetime.datetime.now(timezone.utc)
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    timestamp = int(1e6 * timestamp)

    # set-up the matrix algebra system
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

    # build the Hamiltonian (lambda = 1 here)
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

    # build the gauge operator
    gauge = MatrixOperator(
        data={
            ("X0", "Pi0"): 1,
            ("Pi0", "X0"): -1,
            ("X1", "Pi1"): 1,
            ("Pi1", "X1"): -1,
            ("X2", "Pi2"): 1,
            ("Pi2", "X2"): -1,
            (): 3,
        }
    )

    symmetry_generators = [
        SingleTraceOperator(data={("X1", "Pi2"): 1, ("X2", "Pi1"): -1}), # (1, 2)
        SingleTraceOperator(data={("X0", "Pi2"): 1, ("X2", "Pi0"): -1}), # (0, 2)
        SingleTraceOperator(data={("X0", "Pi1"): 1, ("X1", "Pi0"): -1}), # (0, 1)
    ]

    # build the bootstrap
    bootstrap = BootstrapSystem(
        matrix_system=matrix_system,
        hamiltonian=hamiltonian,
        gauge=gauge,
        max_degree_L=L,
        odd_degree_vanish=True,
        simplify_quadratic=True,
        symmetry_generators=symmetry_generators,
        verbose=verbose,
        save_path=f"data/bfss_L_{L}",
    )
    # load previously-computed constraints
    bootstrap.load_constraints(f"data/bfss_L_{L}")

    # adjust the operator if we wish to obtain an upper bound
    if bound_direction == 'lower':
        sign = 1
    else:
        sign = -1

    # solve
    param, optimization_result = solve_bootstrap(
        bootstrap=bootstrap,
        st_operator_to_minimize=sign * st_operator_to_bound,
        init_scale=1e2,
        maxiters=10,
        maxiters_cvxpy=10_00,
        tol=1e-7,
        reg=1e6,
        eps=1e-4,
    )

    # record various expectation values
    hamiltonian_ev = bootstrap.get_operator_expectation_value(
        st_operator=hamiltonian, param=param
    )

    x_squared = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(
            data={("X0", "X0"): 1, ("X1", "X1"): 1, ("X2", "X2"): 1}
        ),
        param=param,
    )

    p_squared = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(
            data={("Pi0", "Pi0"): -1, ("Pi1", "Pi1"): -1, ("Pi2", "Pi2"): -1}
        ),
        param=param,
    )

    x_4 = bootstrap.get_operator_expectation_value(
        st_operator=SingleTraceOperator(
            data={
                ("X0", "X0", "X0", "X0"): 1,
                ("X1", "X1", "X1", "X1"): 1,
                ("X2", "X2", "X2", "X2"): 1,
            }
        ),
        param=param,
    )

    result = {
        "energy": energy,
        "st_operator_to_bound": st_operator_to_bound.__str__(),
        "bound_direction": bound_direction,
        "hamiltonian_ev": hamiltonian_ev,
        "x_squared": x_squared,
        "p_squared": p_squared,
        "x_4": x_4,
    }
    result = result | optimization_result

    # set-up save path
    path = f"data/bfss_L_{L}"
    if path_suffix is not None:
        path += "_" + path_suffix
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"{path}/{timestamp}.json", "w") as f:
        json.dump(result, f)

    return result


def scan_bootstrap(L:int, parallel:bool=False):

    path = f"data/bfss_L_{L}"
    if not os.path.exists(path):
        os.makedirs(path)

    n_grid = 20

    energies = np.exp(np.linspace(np.log(0.1), np.log(100), n_grid))
    bound_directions = ['lower', 'upper']
    operators_to_bound = {
        "x2": SingleTraceOperator(data={("X0", "X0"): 1, ("X1", "X1"): 1, ("X2", "X2"): 1}),
        "p2": SingleTraceOperator(data={("Pi0", "Pi0"): -1, ("Pi1", "Pi1"): -1, ("Pi2", "Pi2"): -1}),
        "x4": SingleTraceOperator(data={
                ("X0", "X0", "X0", "X0"): 1,
                ("X1", "X1", "X1", "X1"): 1,
                ("X2", "X2", "X2", "X2"): 1,
            }),
            }

    run_configs = []
    for bound_direction in bound_directions:
        for st_op in operators_to_bound:
            for energy in energies:
                run_configs.append({'bound_direction': bound_direction, 'st_op': st_op, 'energy': energy})

    if not parallel:
        for run_config in run_configs:
            run_scan_for_single_grid_point(config_filepath=config_filepath, i=i, j=j)
    else:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_scan_for_single_grid_point, config_filepath, i, j) for run_config in run_configs]
        for future in futures:
            future.result()
        print('finished!')


if __name__ == "__main__":

    fire.Fire()