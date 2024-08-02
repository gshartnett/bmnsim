import os
import pickle
import sys
from datetime import (
    datetime,
    timezone,
)

import fire
import numpy as np

from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.bootstrap_complex import BootstrapSystemComplex
from bmn.debug_utils import disable_debug
from bmn.solver import minimize


def run(nu, L, radius_squared=None, save_path=None):

    if save_path is None:
        save_path = f"data/mini_bmn_L_{L}"

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
            # quartic term (XY)
            ("X0", "X1", "X0", "X1"): -1 / 4,
            ("X1", "X0", "X1", "X0"): -1 / 4,
            ("X0", "X1", "X1", "X0"): 1 / 4,
            ("X1", "X0", "X0", "X1"): 1 / 4,
            # quartic term (XZ) TODO check sign
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

    bootstrap = BootstrapSystemComplex(
        matrix_system=matrix_system,
        hamiltonian=hamiltonian,
        gauge=gauge,
        max_degree_L=L,
        odd_degree_vanish=False,
        simplify_quadratic=False,
        verbose=True,
        save_path=save_path,
        fraction_operators_to_retain=0.3,
    )

    # disable_debug()

    extent_observable = SingleTraceOperator(
        data={("X0", "X0"): 1, ("X1", "X1"): 1, ("X2", "X2"): 1}
    )

    if radius_squared is None:
        op_cons = [
            (SingleTraceOperator(data={(): 1}), 1),
        ]
    else:
        op_cons = [
            (SingleTraceOperator(data={(): 1}), 1),
            (extent_observable, radius_squared),
        ]

    param, success = minimize(
        bootstrap=bootstrap,
        op=bootstrap.hamiltonian,
        op_cons=op_cons,
        init_scale=1e2,
        verbose=True,
        maxiters=25,
        reg=5e-4,
        eps=5e-4,
    )

    np.save(bootstrap.save_path + "/param.npy", param)

    energy = bootstrap.get_operator_expectation_value(
        st_operator=hamiltonian, param=param
    )

    extent_expectation = bootstrap.get_operator_expectation_value(
        st_operator=extent_observable, param=param
    )

    result = {
        "param": param,
        "energy": energy,
        "extent": extent_expectation,
        "L": L,
        "nu": nu,
        "success": success,
        "fraction_operators_to_retain": bootstrap.fraction_operators_to_retain,
    }

    now_utc = datetime.now(timezone.utc)
    now_utc = int(datetime.now(timezone.utc).timestamp() * 1000)
    with open(save_path + f"/result_{now_utc}", "wb") as f:
        pickle.dump(result, f)

    print(
        f"problem success: {success}, min energy found: {energy:.6f}, r^2: {extent_expectation:.6f}"
    )

    return success, energy, param


if __name__ == "__main__":

    success, energy, param = fire.Fire(run)

    # for nu in np.linspace(0.5, 10, 10):
    #    for radius_squared in np.linspace(0.5, 10, 10):
    #        run(nu=nu, L=3, radius_squared=radius_squared)
