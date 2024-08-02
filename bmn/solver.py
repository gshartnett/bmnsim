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
    vstack,
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

"""
TODO
add Typing
clean up variable names
acknowledge https://github.com/gshartnett/matrix-bootstrap/blob/main/solver.py
"""


def compute_L2_norm_of_linear_constraints(A, b, param):
    return sum(np.square(A @ param - b))


def compute_L2_norm_of_quadratic_constraints(quadratic_constraints, param):
    linear_term = quadratic_constraints["linear"] @ param
    param_product = np.outer(param, param).reshape((len(param) ** 2))
    quadratic_term = quadratic_constraints["quadratic"] @ param_product
    return np.sum(np.square(linear_term + quadratic_term))


def minimal_eigval(bootstrap_array_sparse, parameter_vector_null):
    dim = int(np.sqrt(bootstrap_array_sparse.shape[0]))
    bootstrap_matrix = np.reshape(
        bootstrap_array_sparse.dot(parameter_vector_null), (dim, dim)
    )

    if not np.allclose(
        (bootstrap_matrix - bootstrap_matrix.T), np.zeros_like(bootstrap_matrix)
    ):
        violation = np.max((bootstrap_matrix - bootstrap_matrix.T))
        raise ValueError(f"Bootstrap matrix is not symmetric, violation = {violation}.")

    return scipy.linalg.eigvalsh(bootstrap_matrix)[0]


def sdp_init(
    bootstrap_array_sparse, A, b, init, reg=1, maxiters=5_000, eps=1e-4, verbose=True
):
    """
    Finds the parameters such that
            1. All bootstrap tables are positive semidefinite;
            2. ||A.dot(param) - b||^2_2 + reg * ||param - init||^2_2 is minimized.
    Arguments:
            tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then
                    has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square
                    matrices. The constraint is that these matrices must be positive semidefinite.
            A (sparse matrix of shape (num_constraints, num_variables)), b (numpy array of shape (num_constraints,)):
                    linear constraints that A.dot(param) = b
            init (numpy array of shape (num_variables,)): the initial parameters
            reg (float): regularization
            maxiters (int), eps (float), verbose (bool): options for the convex solver
    Returns:
            param (numpy array of shape (num_variables,)): the optimal parameters found
    """

    # initialize the cvxpy parameter vector in the null space
    num_variables = init.size
    param = cp.Variable(num_variables)

    # the PSD constraint(s) (multiple if bootstrap matrix is block diagonal)
    size = int(np.sqrt(bootstrap_array_sparse.shape[0]))
    constraints = [cp.reshape(bootstrap_array_sparse @ param, (size, size)) >> 0]

    # solve the above described optimization problem
    prob = cp.Problem(
        cp.Minimize(
            cp.sum_squares(A @ param - b) + 0 * reg * cp.sum_squares(param - init)
        ),
        constraints,
    )
    prob.solve(verbose=verbose, max_iters=maxiters, eps=eps, solver=cp.SCS)

    if str(prob.status) != "optimal":
        debug("WARNING: sdp_init unexpected status: " + prob.status)

    return param.value


def sdp_relax(
    bootstrap_array_sparse,
    A,
    b,
    init,
    radius,
    maxiters=10_000,
    eps=1e-4,
    relax_rate=0.8,
    verbose=True,
):
    """
    Finds the parameters such that
            1. All bootstrap tables are positive semidefinite;
            2. ||param - init||_2 <= 0.8 * radius;
            3. The violation of linear constraints ||A.dot(param) - b||_2 is minimized.
    Arguments:
            tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then
                    has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square
                    matrices. The constraint is that these matrices must be positive semidefinite.
            A (sparse matrix of shape (num_constraints, num_variables)), b (numpy array of shape (num_constraints,)):
                    linear constraints that A.dot(param) = b
            init (numpy array of shape (num_variables,)): the initial parameters
            radius (float): radius of the trust region
            maxiters (int), eps (float), verbose (bool): options for the convex solver
    Returns:
            param (numpy array of shape (num_variables,)): the optimal parameters found
    """
    # initialize the cvxpy parameter vector in the null space
    num_variables = init.size
    param = cp.Variable(num_variables)

    # build the constraints
    # 1. ||param - init||_2 <= relax_rate * radius
    # 2. the PSD bootstrap constraint(s)
    constraints = [cp.norm(param - init) <= relax_rate * radius]
    size = int(np.sqrt(bootstrap_array_sparse.shape[0]))
    constraints = [cp.reshape(bootstrap_array_sparse @ param, (size, size)) >> 0]

    # solve the above described optimization problem
    prob = cp.Problem(cp.Minimize(cp.norm(A @ param - b)), constraints)
    prob.solve(verbose=verbose, max_iters=maxiters, eps=eps, solver=cp.SCS)

    if str(prob.status) != "optimal":
        debug("WARNING: sdp_relax unexpected status: " + prob.status)

    return param.value


def sdp_minimize(
    vec,
    bootstrap_array_sparse,
    A,
    b,
    init,
    radius,
    reg=0,
    maxiters=10_000,
    eps=1e-4,
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
    constraints = [cp.norm(param - init) <= radius]
    size = int(np.sqrt(bootstrap_array_sparse.shape[0]))
    constraints = [cp.reshape(bootstrap_array_sparse @ param, (size, size)) >> 0]
    constraints += [A @ param == b]

    # the loss to minimize
    loss = vec @ param + reg * cp.norm(param)

    # solve the above described optimization problem
    prob = cp.Problem(cp.Minimize(loss), constraints)
    prob.solve(verbose=verbose, max_iters=maxiters, eps=eps, solver=cp.SCS)

    if str(prob.status) != "optimal":
        debug("WARNING: sdp_minimize unexpected status: " + prob.status)
    return param.value, prob.status, prob.value


def get_quadratic_constraint_vector_dense(
    quadratic_constraints: dict[str, np.ndarray],
    param: np.ndarray,
    compute_grad: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Compute the quadratic constraint vector
        A_{Iij} K_{ia} K_{jb} u_a u_b + B_{Ii} K_{Ia} u_a
    and optionally the gradient
        A_{Iij} K_{ia} K_{jb} u_b + A_{Iij} K_{ia} K_{jb} u_a + B_{Ii} K_{Ia}
    for a given parameter vector (in the null space) u.

    Parameters
    ----------
    quadratic_constraints : dict[str, np.ndarray]
        The quadratic and linear parts of the quadratic/cyclic constraints,
        represented as arrays.
    param : np.ndarray
        The parameter vector in the null space, u.
    compute_grad : bool, optional
        Controls whether the grad is computed, by default False.

    Returns
    -------
    Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
        The constraint vector and optionally its gradient.
    """

    # compute the constraints
    # linear term
    linear_term = quadratic_constraints["linear"] @ param

    # quadratic term
    param_product = np.outer(param, param).reshape((len(param) ** 2))
    quadratic_array = quadratic_constraints["quadratic"]
    quadratic_term = quadratic_constraints["quadratic"] @ param_product

    constraint_vector = linear_term + quadratic_term

    # return the constraint only if the gradient is not needed
    if not compute_grad:
        return constraint_vector

    # compute the gradient
    num_constraints = linear_term.shape[0]
    quadratic_array = np.asarray(quadratic_constraints["quadratic"].todense())
    quadratic_array = quadratic_array.reshape((num_constraints, len(param), len(param)))
    # compute the gradient matrix
    constraint_grad_quadratic_term_1 = np.einsum("Iij, i -> Ij", quadratic_array, param)
    constraint_grad_quadratic_term_2 = np.einsum("Iij, j -> Ii", quadratic_array, param)
    constraint_grad = (
        quadratic_constraints["linear"]
        + constraint_grad_quadratic_term_1
        + constraint_grad_quadratic_term_2
    )

    return constraint_vector, constraint_grad


def get_quadratic_constraint_vector_sparse(
    quadratic_constraints: dict[str, np.ndarray],
    param: np.ndarray,
    compute_grad: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Compute the quadratic constraint vector
        A_{Iij} K_{ia} K_{jb} u_a u_b + B_{Ii} K_{Ia} u_a
    and optionally the gradient
        A_{Iij} K_{ia} K_{jb} u_b + A_{Iij} K_{ia} K_{jb} u_a + B_{Ii} K_{Ia}
    for a given parameter vector (in the null space) u.

    Parameters
    ----------
    quadratic_constraints : dict[str, np.ndarray]
        The quadratic and linear parts of the quadratic/cyclic constraints,
        represented as arrays.
    param : np.ndarray
        The parameter vector in the null space, u.
    compute_grad : bool, optional
        Controls whether the grad is computed, by default False.

    Returns
    -------
    Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
        The constraint vector and optionally its gradient.
    """

    # compute the constraints
    # linear term
    linear_term = quadratic_constraints["linear"] @ param

    # quadratic term
    param_product = np.outer(param, param).reshape((len(param) ** 2))
    quadratic_term = quadratic_constraints["quadratic"] @ param_product

    constraint_vector = linear_term + quadratic_term

    # return the constraint only if the gradient is not needed
    if not compute_grad:
        return constraint_vector

    # compute the gradient
    num_constraints = linear_term.shape[0]
    quad_terms = [
        quadratic_constraints["quadratic"][i].reshape((len(param), len(param)))
        for i in range(num_constraints)
    ]
    quad_terms = vstack(
        [
            csr_matrix(quad_term @ param + quad_term.T @ param)
            for quad_term in quad_terms
        ]
    )
    constraint_grad = quadratic_constraints["linear"] + quad_terms

    # I found that this is critical, without it the result is wrong
    constraint_grad = constraint_grad.todense()

    return constraint_vector, constraint_grad


def minimize(
    bootstrap,
    op,
    init=None,
    init_scale=1.0,
    op_cons=[(SingleTraceOperator(data={(): 1}), 1)],
    maxiters=25,
    eps=5e-4,
    reg=5e-4,
    relax_rate=0.8,
    verbose=True,
    savefile="",
):
    """
    Minimizes the operator subject to the bootstrap positivity constraint, the quadratic cyclicity constraint, and the operator values
    constraints. The algorithm is a trust-region sequential semidefinite programming with regularization on l2 norm of the parameters.
    Arguments:
            op (TraceOperator): expectation value of this operator is to be minimized
            tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then
                    has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square
                    matrices. The constraint is that these matrices must be positive semidefinite.
            quad_cons (QuadraticSolution): the quadratic constraints from solving the cyclicity constraints
            op_cons (list of tuples (TraceOperator, float)): the extra constraints specifying the expectation values of the given operators
            init (numpy array of shape (num_variables,)): the initial value of the parameters
            eps (float): the accuracy goal
            reg (float): regulariation on parameters. The true objective to minimize will be expectation value of op + reg * l2-norm(param)
            verbose (bool): whether to print the optimization progress
            savefile (string): the filename to save the parameters
    Returns:
            param (numpy array of shape (num_variables,)): the optimal value of parameters found
    """
    # get the quantities needed for numerics
    linear_constraint_matrix = bootstrap.build_linear_constraints().tocsr()
    quadratic_constraints = bootstrap.build_quadratic_constraints()
    bootstrap_array_sparse = bootstrap.build_bootstrap_table()

    if init is None:
        # initial parameter vector
        print(f"Initializing randomly")
        init = init_scale * np.random.normal(size=bootstrap.param_dim_null)

        # print(f"Initializing from all 1's")
        # init = np.ones(shape=bootstrap.param_dim_null)

        # print(f"Initializing from all 0's")
        # init = np.zeros(shape=bootstrap.param_dim_null)

    # print(f"Rescale to normalize")
    # init = bootstrap.scale_param_to_enforce_normalization(init)  # rescale to normalize

    # vector corresponding to op to minimize (typically the Hamiltonian)
    vec = bootstrap.single_trace_to_coefficient_vector(op, return_null_basis=True)
    vec = vec.real  # TODO check this!

    # the loss function to minimize, i.e., the value of op
    # vec = operator_to_vector(sol, op)
    loss = lambda param: vec.dot(param)

    # extra constraints in op_cons, i.e., <o> = v for o, v in op_cons
    A_op = sparse.csr_matrix((0, bootstrap.param_dim_null))
    b_op = np.zeros(0)
    for op, value in op_cons:
        A_op = sparse.vstack(
            [
                A_op,
                sparse.csr_matrix(
                    bootstrap.single_trace_to_coefficient_vector(
                        op, return_null_basis=True
                    )
                ),
            ]
        )
        b_op = np.append(b_op, value)

    # initialize parameters from file or from scratch
    last_param = None
    if savefile and os.path.isfile(savefile + ".npz"):
        npzfile = np.load(savefile + ".npz")
        radius, obs = npzfile["radius"], npzfile["obs"]
        param = lsqr(linear_constraint_matrix, obs)[0]
        # sanity checks
        debug(
            "Error: {}".format(
                np.linalg.norm(linear_constraint_matrix.dot(param) - obs)
            )
        )
        debug(
            "minimal_eigval: {}".format(minimal_eigval(bootstrap_array_sparse, param))
        )
        debug("Data read successfully")
    else:
        debug("Starting from scratch...")
        # find an initial parameter close to init that makes all bootstrap matrices positive
        # print(f"INSIDE SOLVER, param_dim_null = {bootstrap.param_dim_null}, {A_op.shape, b_op.shape, init.shape}")
        param = sdp_init(
            bootstrap_array_sparse=bootstrap_array_sparse,
            A=A_op,
            b=b_op,
            init=init,
            verbose=verbose,
        )
        radius = np.linalg.norm(param) + 20
        print(f"initial ||param||_2 = {np.linalg.norm(param)}, R={radius}")

    # penalty parameter for violation of constraints
    mu = 1

    # optimization steps
    for step in range(maxiters):
        print(f"step = {step+1}/{maxiters}")
        debug(f"step = {step+1}/{maxiters}")
        # combine the constraints from op_cons and linearized quadratic constraints, i.e., grad * (new_param - param) + val = 0
        val, grad = get_quadratic_constraint_vector_sparse(
            quadratic_constraints, param, compute_grad=True
        )
        if step == 0:
            quadratic_constraint_scale_vector = (
                1 / 100 * np.abs(np.random.normal(size=10))
            )
            print(quadratic_constraint_scale_vector)
        val = val / quadratic_constraint_scale_vector
        grad = grad / quadratic_constraint_scale_vector[:, None]
        # print(f"min |q_I| = {min(abs(val))}, max |q_I| = {max(abs(val))}")

        A = sparse.vstack([A_op, grad])
        b = np.append(b_op, grad.dot(param) - val)

        # one step
        relaxed_param = sdp_relax(
            bootstrap_array_sparse=bootstrap_array_sparse,
            A=A,
            b=b,
            init=param,
            radius=radius,
            relax_rate=relax_rate,
            verbose=verbose,
        )
        new_param, prob_status, prob_value = sdp_minimize(
            vec,
            bootstrap_array_sparse,
            A,
            A.dot(relaxed_param),
            param,
            radius,
            reg=reg,
            verbose=verbose,
        )
        if new_param is None:
            # wrongly infeasible
            radius *= relax_rate  # GSH used to be 0.9
            print(
                f"  wrongly infeasible, changing radius from R={radius/relax_rate} to R={radius}"
            )
            continue
        # check progress
        cons_val = A.dot(param) - b
        maxcv = np.max(np.abs(cons_val))  # maximal constraint violation
        min_eig = minimal_eigval(bootstrap_array_sparse, param)
        # if verbose:
        if True:
            print(
                "Step {}: \tloss = {:.5f}, maxcv = {:.5f}, radius = {:.3e}, min_eig = {:+.5f}".format(
                    step, loss(param), maxcv, radius, min_eig
                )
            )
            print(
                "\t\tnorm = {:.5f}, update = {:.5e}".format(
                    np.linalg.norm(param), np.linalg.norm(new_param - param)
                )
            )
        if (
            last_param is not None
            and maxcv < eps
            and min_eig > -eps
            and loss(param) > loss(last_param) - 0.1 * eps
            and np.linalg.norm(param - last_param) < eps * radius
        ):
            if verbose:
                print("Accuracy goal achieved.")
            return param, True
        # compute the changes in the merit function and decide whether the step should be accepted
        dloss = (
            loss(new_param)
            - loss(param)
            + reg * (np.linalg.norm(new_param) - np.linalg.norm(param))
        )
        dcons = np.linalg.norm(
            np.append(
                A_op.dot(new_param) - b_op,
                get_quadratic_constraint_vector_sparse(
                    quadratic_constraints, new_param
                ),
            )
        ) - np.linalg.norm(cons_val)
        if -dcons > eps:
            old_mu = mu
            mu = max(mu, -2 * dloss / dcons)
            print(f"  adjusting mu from mu = {old_mu} to mu = {mu}")

        # the merit function is -loss(param) - reg * norm(param) - mu * norm(constraint vector)
        acred = -dcons  # actual constraint reduction
        ared = -dloss + mu * acred  # actual merit increase
        pcred = np.linalg.norm(cons_val) - np.linalg.norm(
            A.dot(new_param) - b
        )  # predicted constraint reduction
        pred = -dloss + mu * pcred  # predicted merit increase
        rho = ared / pred  # some sanity checks

        if verbose:
            print(
                "\t\tmu = {:.3f}, margin = {:+.3f}".format(
                    mu, -dloss - 0.5 * mu * dcons
                )
            )
            print(
                "\t\tacred = {:+.5f}, pcred = {:+.5f}, ared = {:+.5f}, pred = {:+.5f}, rho = {:+.3f}".format(
                    acred, pcred, ared, pred, rho
                )
            )
        if rho > 0.5:
            # accept
            if maxcv < eps and min_eig > -eps:
                last_param = param
                # radius *= 1.2
                radius *= 2 - relax_rate  # GSH used to be 1.2
                print(
                    f"  accept, adjusting R from R = {radius/(2 - relax_rate)} to R = {radius}"
                )
            param = new_param
            if savefile:
                obs = linear_constraint_matrix.dot(param)
                np.savez(savefile, radius=radius, obs=obs)
                debug("Data saved successfully")
        else:
            # reject
            old_radius = radius
            radius = relax_rate * np.linalg.norm(new_param - param)
            print(f"  reject, adjusting R from R = {old_radius} to R = {radius}")

    debug(
        "WARNING: minimize did not converge to precision {:.5f} within {} steps.".format(
            eps, maxiters
        )
    )
    return param, False
