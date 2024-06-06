import numpy as np
from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.bootstrap import BootstrapSystem
from bmn.debug_utils import disable_debug
from bmn.solver import minimize

nu = 1.0 * 0
L = 2

matrix_system = MatrixSystem(
    operator_basis=['X0', 'X1', 'X2', 'Pi0', 'Pi1', 'Pi2'],
    commutation_rules_concise = {
        ('Pi0', 'X0'): 1,
        ('Pi1', 'X1'): 1,
        ('Pi2', 'X2'): 1,
    },
    hermitian_dict={'Pi0': False, 'Pi1': False, 'Pi2': False, 'X0': True, 'X1': True, 'X2': True},
)

# scale variables as P = sqrt(N) P', X = sqrt(N) X'
hamiltonian = SingleTraceOperator(
        data={
           # kinetic term
            ("Pi0", "Pi0"): -1/2,
            ("Pi1", "Pi1"): -1/2,
            ("Pi2", "Pi2"): -1/2,
            # quadratic term
            ('X0', 'X0'): nu**2 / 2,
            ('X1', 'X1'): nu**2 / 2,
            ('X2', 'X2'): nu**2 / 2,
            # cubic term
            ('X0', 'X1', 'X2'): 6 * 1j * nu,
            # quadratic term (XY)
            ('X0', 'X1', 'X0', 'X1'): - 1/4,
            ('X1', 'X0', 'X1', 'X0'): -1/4,
            ('X0', 'X1', 'X1', 'X0'): 1/4,
            ('X1', 'X0', 'X0', 'X1'): 1/4,
            # quadratic term (XZ) TODO check sign
            ('X0', 'X2', 'X0', 'X2'): - 1/4,
            ('X2', 'X0', 'X2', 'X0'): -1/4,
            ('X0', 'X2', 'X2', 'X0'): 1/4,
            ('X2', 'X0', 'X0', 'X2'): 1/4,
            # quadratic term (YZ)
            ('X1', 'X2', 'X1', 'X2'): - 1/4,
            ('X2', 'X1', 'X2', 'X1'): -1/4,
            ('X1', 'X2', 'X2', 'X1'): 1/4,
            ('X2', 'X1', 'X1', 'X2'): 1/4,
            }
            )

gauge = MatrixOperator(data={
    ('X0', 'Pi0'): 1,
    ('Pi0', 'X0'): -1,
    ('X1', 'Pi1'): 1,
    ('Pi1', 'X1'): -1,
    ('X2', 'Pi2'): 1,
    ('Pi2', 'X2'): -1,
    ():1
    })

bootstrap = BootstrapSystem(
    matrix_system=matrix_system,
    hamiltonian=hamiltonian,
    gauge=gauge,
    half_max_degree=L,
    odd_degree_vanish=False,
)

bootstrap.get_null_space_matrix();

#disable_debug()

param, success = minimize(
    bootstrap=bootstrap,
    op=bootstrap.hamiltonian,
    init=1*np.random.normal(size=bootstrap.param_dim_null),
    verbose=False,
    maxiters=25,
    reg=5e-4,
    eps=5e-4,
)

#print(f"problem status: {prob_status}, min energy: {prob_value}")