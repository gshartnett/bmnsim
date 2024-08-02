from itertools import (
    chain,
    product,
)
from numbers import Number
from typing import (
    Self,
    Union,
)

import cvxpy as cp
import numpy as np
import scipy
import scipy.sparse as sparse
import sympy as sp
from scipy.linalg import qr
from scipy.sparse import (
    coo_matrix,
    csc_matrix,
    csr_matrix,
)
from scipy.sparse.linalg import (
    splu,
    svds,
)
from sksparse.cholmod import cholesky

from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.bootstrap import BootstrapSystem
from bmn.debug_utils import debug
from bmn.linear_algebra import (
    create_sparse_matrix_from_dict,
    get_null_space_sparse,
)
#from bmn.solver import get_quadratic_constraint_vector_dense as get_quadratic_constraint_vector
from bmn.solver import get_quadratic_constraint_vector_sparse as get_quadratic_constraint_vector


def sdp_minimize(
    vec,
    bootstrap_array_sparse,
    A1,
    b1,
    A2,
    b2,
    init,
    radius = np.inf,
    maxiters=10_000,
    eps=1e-4,
    reg=1e-4,
    verbose=True,
    ):
    """
    Finds the parameters such that
            1. All bootstrap tables are positive semidefinite;
            2. ||param - init||_2 <= radius;
            3. A.dot(param) = b;
            4. vec.dot(param) + reg * np.linalg.norm(param) is minimized.
    Arguments:
            vec (numpy array of shape (num_variables,))
            tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then
                    has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square
                    matrices. The constraint is that these matrices must be positive semidefinite.
            A (sparse matrix of shape (num_constraints, num_variables)), b (numpy array of shape (num_constraints,)):
                    linear constraints that A.dot(param) = b
            init (numpy array of shape (num_variables,)): the initial parameters
            radius (float): radius of the trust region
            reg (float): regularization parameter
            maxiters (int), eps (float), verbose (bool): options for the convex solver
    Returns:
            param (numpy array of shape (num_variables,)): the optimal parameters found
    """
    # initialize the cvxpy parameter vector in the null space
    num_variables = init.size
    param = cp.Variable(num_variables)

    # build the constraints
    # 1. ||param - init||_2 <= radius
    # 2. the PSD bootstrap constraint(s)
    # 3. A @ param == 0
    size = int(np.sqrt(bootstrap_array_sparse.shape[0]))
    constraints = [cp.reshape(bootstrap_array_sparse @ param, (size, size)) >> 0]
    constraints += [A1 @ param == b1]
    constraints += [cp.norm(param - init) <= radius]

    # the loss to minimize
    loss = vec @ param + reg * cp.norm(A2 @ param - b2)

    # solve the above described optimization problem
    prob = cp.Problem(cp.Minimize(loss), constraints)
    prob.solve(verbose=verbose, max_iters=maxiters, eps=eps, solver=cp.SCS)

    if str(prob.status) != "optimal":
        debug("WARNING: sdp_minimize unexpected status: " + prob.status)
    return param.value, prob.status, prob.value


def minimize(
    bootstrap,
    op,
    init=None,
    init_scale=1.0,
    op_cons=[(SingleTraceOperator(data={(): 1}), 1)],
    maxiters=25,
    verbose=True,
    savefile="",
    ):

    # get the quantities needed for numerics
    _ = bootstrap.build_linear_constraints().tocsr()
    quadratic_constraints = bootstrap.build_quadratic_constraints()
    print(f"Building bootstrap table")
    bootstrap_array_sparse = bootstrap.build_bootstrap_table()

    quadratic_constraints['linear'] = quadratic_constraints['linear']
    quadratic_constraints['quadratic'] = quadratic_constraints['quadratic']

    if init is None:
        # initial parameter vector
        print(f"Initializing randomly")
        init = init_scale * np.random.normal(size=bootstrap.param_dim_null)

        # print(f"Initializing from all 1's")
        #init = np.ones(shape=bootstrap.param_dim_null)

        # print(f"Initializing from all 0's")
        # init = np.zeros(shape=bootstrap.param_dim_null)

    #print(f"Rescale to normalize")
    #init = bootstrap.scale_param_to_enforce_normalization(init)  # rescale to normalize

    param = init

    # vector corresponding to op to minimize (typically the Hamiltonian)
    vec = bootstrap.single_trace_to_coefficient_vector(op, return_null_basis=True)
    vec = vec.real # TODO check this!

    # the loss function to minimize, i.e., the value of op
    # vec = operator_to_vector(sol, op)
    #loss = lambda param: vec.dot(param)

    # extra constraints in op_cons, i.e., <o> = v for o, v in op_cons
    A = sparse.csr_matrix((0, bootstrap.param_dim_null))
    b = np.zeros(0)
    for (op, value) in op_cons:
        A = sparse.vstack(
            [
                A,
                sparse.csr_matrix(
                    bootstrap.single_trace_to_coefficient_vector(
                        op, return_null_basis=True
                    )
                ),
            ]
        )
        b = np.append(b, value)

    # iterate over steps
    for step in range(maxiters):
        print(f"step = {step+1}/{maxiters}")
        debug(f"step = {step+1}/{maxiters}")

        # combine the constraints from op_cons and linearized quadratic constraints, i.e., grad * (new_param - param) + val = 0
        val, grad = get_quadratic_constraint_vector(
            quadratic_constraints, param, compute_grad=True
        )

        #for i in range(len(val)):
        #    print(f"val[i] = {val[i]}")
        #    print(f"grad val[i] = {np.max(np.abs(grad[i]))}")


        #if step == 0:
        #    quadratic_constraint_scale_vector = 1/100 * np.abs(np.random.normal(size=10))
        #    print(quadratic_constraint_scale_vector)
        #val = val / quadratic_constraint_scale_vector
        #grad = grad / quadratic_constraint_scale_vector[:, None]
        #print(f"min |q_I| = {min(abs(val))}, max |q_I| = {max(abs(val))}")

        #scale = 1e0
        #if step > 1115:
        #    A = sparse.vstack([A, grad])
        #    b = np.append(b, (grad.dot(param) - val))

        A2 = grad
        b2 = np.asarray(grad.dot(param) - val)[0]

        param, _, _ = sdp_minimize(
            vec=vec,
            bootstrap_array_sparse=bootstrap_array_sparse,
            A1=A,
            b1=b,
            A2=A2,
            b2=b2,
            init=param,
            verbose=verbose,
            radius=1e8,
            reg=1e6,
        )

        if param is None:
            assert 1==0

        # combine the constraints from op_cons and linearized quadratic constraints, i.e., grad * (new_param - param) + val = 0
        val, grad = get_quadratic_constraint_vector(
            quadratic_constraints, param, compute_grad=True
        )

    return param, True