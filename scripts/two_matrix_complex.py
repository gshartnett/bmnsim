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
from bmn.newton_solver import minimize as minimize_newton
from bmn.solver import minimize


def run(
        mass,
        gauge_coupling,
        L,
        fraction_operators_to_retain=1.0,
        save_path=None,
        init=None,
        ):

    if save_path is None:
        save_path=f"data/two_matrix_complex_{L}"

    matrix_system = MatrixSystem(
        operator_basis=["X0", "X1", "P0", "P1"],
        commutation_rules_concise={
            ("P0", "X0"): -1j,
            ("P1", "X1"): -1j,
        },
        hermitian_dict={
            "P0": True,
            "P1": True,
            "X0": True,
            "X1": True,
        },
    )

    # scale variables as P = sqrt(N) P', X = sqrt(N) X'
    hamiltonian = SingleTraceOperator(
        data={
            # kinetic term
            ("P0", "P0"): 1,
            ("P1", "P1"): 1,
            # quadratic term
            ("X0", "X0"): mass**2,
            ("X1", "X1"): mass**2,
            # quadratic term (XY)
            ("X0", "X1", "X0", "X1"): -gauge_coupling**2,
            ("X1", "X0", "X1", "X0"): -gauge_coupling**2,
            ("X0", "X1", "X1", "X0"): gauge_coupling**2,
            ("X1", "X0", "X0", "X1"): gauge_coupling**2,
        }
    )

    gauge = MatrixOperator(
        data={
            ("X0", "P0"): 1j,
            ("P0", "X0"): -1j,
            ("X1", "P1"): 1j,
            ("P1", "X1"): -1j,
            (): 2,
        }
    )

    symmetry_generators = [
        SingleTraceOperator(data={('X0', 'P1'): 1, ('X1', 'P0'): -1})
        ]

    bootstrap = BootstrapSystemComplex(
        matrix_system=matrix_system,
        hamiltonian=hamiltonian,
        gauge=gauge,
        max_degree_L=L,
        odd_degree_vanish=True,
        simplify_quadratic=False,
        verbose=True,
        save_path=save_path,
        #symmetry_generators=symmetry_generators,
        fraction_operators_to_retain=fraction_operators_to_retain,
    )

    op_cons = [
        (SingleTraceOperator(data={(): 1}), 1),
        ]

    param, success = minimize_newton(
        bootstrap=bootstrap,
        op=bootstrap.hamiltonian,
        init=init,
        init_scale=1e2,
        maxiters=50,
    )

    '''
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
    '''

    np.save(bootstrap.save_path + "/param.npy", param)

    energy = bootstrap.get_operator_expectation_value(
        st_operator=hamiltonian,
        param=param
        )
    energy_imag = np.imag(energy)
    if np.abs(energy_imag) > 1e-10:
        raise ValueError("Error, found appreciable imaginary component to energy.")
    energy = np.real(energy)

    result = {
        'param': param,
        'energy': energy,
        'L': L,
        'mass': mass,
        'gauge_coupling': gauge_coupling,
        'success':success,
        'fraction_operators_to_retain':bootstrap.fraction_operators_to_retain,
    }

    now_utc = datetime.now(timezone.utc)
    now_utc = int(datetime.now(timezone.utc).timestamp() * 1000)
    with open(save_path + f"/result_{now_utc}", 'wb') as f:
        pickle.dump(result, f)

    print(f"problem success: {success}, min energy found: {energy:.6f}")

    for op in bootstrap.operator_list:
        if len(op) <= 2:
            op_expectation_value = bootstrap.get_operator_expectation_value(
                st_operator=SingleTraceOperator(data={op: 1}),
                param=param
                )
            print(f"op = {op}, EV = {op_expectation_value}")

    return success, energy, param


if __name__ == "__main__":

    success, energy, param = fire.Fire(run)

    #for nu in np.linspace(0.5, 10, 10):
    #    for radius_squared in np.linspace(0.5, 10, 10):
    #        run(nu=nu, L=3, radius_squared=radius_squared)