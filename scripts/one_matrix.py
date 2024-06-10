import numpy as np

from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.bootstrap import BootstrapSystem
from bmn.debug_utils import disable_debug
from bmn.solver import minimize

g = 2
L = 5

matrix_system = MatrixSystem(
    # operator_basis=['X', 'P'],
    operator_basis=["X", "Pi"],
    commutation_rules_concise={
        # ('P', 'X'): -1j,
        ("Pi", "X"): 1,  # use Pi' = i P to ensure reality
    },
    # hermitian_dict={'P': True, 'X': True},
    hermitian_dict={"Pi": False, "X": True},
)

# scale variables as P = sqrt(N) P', X = sqrt(N) X'
hamiltonian = SingleTraceOperator(
    # data={("P", "P"): 1, ("X", "X"): 1, ("X", "X", "X", "X"): 7}
    data={("Pi", "Pi"): -1, ("X", "X"): 1, ("X", "X", "X", "X"): g}
)

# <tr G O > = 0 might need to be applied only for O with deg <= L-2
# gauge = MatrixOperator(data={('X', 'P'): 1j, ('P', 'X'): -1j, ():1})
gauge = MatrixOperator(data={("X", "Pi"): 1, ("Pi", "X"): -1, (): 1})

bootstrap = BootstrapSystem(
    matrix_system=matrix_system,
    hamiltonian=hamiltonian,
    gauge=gauge,
    half_max_degree=L,
    odd_degree_vanish=True,
)

bootstrap.get_null_space_matrix()

disable_debug()

param, success = minimize(
    bootstrap=bootstrap,
    op=bootstrap.hamiltonian,
    init_scale=1.0,
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
print(f"problem success: {success}, min energy: {energy}")
