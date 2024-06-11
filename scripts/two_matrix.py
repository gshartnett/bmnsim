import numpy as np
import fire
from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.bootstrap import BootstrapSystem
from bmn.debug_utils import disable_debug
from bmn.solver import minimize
from bmn.brezin import compute_Brezin_energy, compute_Brezin_energy_Han_conventions

# plot settings
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 5.0
plt.rcParams["xtick.minor.size"] = 3.0
plt.rcParams["ytick.major.size"] = 5.0
plt.rcParams["ytick.minor.size"] = 3.0
plt.rcParams["lines.linewidth"] = 2
plt.rc("font", family="serif", size=16)
matplotlib.rc("text", usetex=True)
matplotlib.rc("legend", fontsize=16)
matplotlib.rcParams["axes.prop_cycle"] = cycler(
    color=["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42", "#FFB5B8"]
)
matplotlib.rcParams.update(
    {"axes.grid": True, "grid.alpha": 0.75, "grid.linewidth": 0.5}
)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def run_two_matrix(g, L, m=1, init=None):

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
            ("X1", "X2", "X1", "X2"): -g*2,
            ("X2", "X1", "X2", "X1"): -g*2,
            ("X1", "X2", "X2", "X1"): g*2,
            ("X2", "X1", "X1", "X2"): g*2,
            }
    )

    # <tr G O > = 0 might need to be applied only for O with deg <= L-2
    # gauge = MatrixOperator(data={('X', 'P'): 1j, ('P', 'X'): -1j, ():1})
    gauge = MatrixOperator(data={
        ("X1", "Pi1"): 1,
        ("Pi1", "X1"): -1,
        ("X2", "Pi2"): 1,
        ("Pi2", "X2"): -1,
        (): 2
        })

    bootstrap = BootstrapSystem(
        matrix_system=matrix_system,
        hamiltonian=hamiltonian,
        gauge=gauge,
        half_max_degree=L,
        odd_degree_vanish=True,
        simplify_quadratic=False,
    )

    bootstrap.get_null_space_matrix()

    disable_debug()

    param, success = minimize(
        bootstrap=bootstrap,
        op=bootstrap.hamiltonian,
        init=init,
        init_scale=1e2,
        verbose=False,
        maxiters=25,
        reg=5e-4,
        eps=5e-4,
    )

    """
    for op in bootstrap.operator_list:
        vec = bootstrap.single_trace_to_coefficient_vector(
            st_operator=SingleTraceOperator(data={op: 1}), return_null_basis=True
        )
        op_expectation_value = vec @ param
        print(f"op = {op}, EV = {op_expectation_value}")
    """

    vec = bootstrap.single_trace_to_coefficient_vector(
        st_operator=hamiltonian, return_null_basis=True
    )
    energy = vec @ param
    #exact_energy = compute_Brezin_energy_Han_conventions(g)
    print(f"problem success: {success}, min energy found: {energy:.6f}") #, exact (L=inf) value = {exact_energy:.6f}")
    return success, energy, param


if __name__ == "__main__":

    fire.Fire(run_two_matrix)