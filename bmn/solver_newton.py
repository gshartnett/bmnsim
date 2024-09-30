from typing import Optional

import cvxpy as cp
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.linalg import ishermitian
from bmn.algebra import SingleTraceOperator
from bmn.bootstrap import BootstrapSystem
from bmn.debug_utils import debug
from bmn.solver_trustregion import (
    get_quadratic_constraint_vector_sparse as get_quadratic_constraint_vector,
)


def sdp_minimize(
    linear_objective_vector: np.ndarray,
    bootstrap_table_sparse: csr_matrix,
    linear_inhomogeneous_eq: tuple[csr_matrix, np.ndarray],
    linear_inhomogeneous_penalty: Optional[tuple[csr_matrix, np.ndarray]] = None,
    radius: float = np.inf,
    maxiters: int = 2500,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    eps_infeas: float = 1e-7,
    reg: float = 1e-4,
    penalty_reg: float = 1e6,
    verbose: bool = False,
    cvxpy_solver: str = "SCS",
) -> tuple[bool, str, np.ndarray]:
    """
    Performs the following SDP minimization over the vector variable x:

    A linear objective v^T x is minimized, subject to the constraints
        - The bootstrap matrix M(x) is positive semi-definite
        - The linear inhomogeneous equation A x = b is satisfied
        - A second linear inhomgeneous equation A' x = b' is imposed as a penalty
        in the objective function
        - The variable x is constrained to lie within a radius-R ball of an initial
        variable `init`

    Parameters
    ----------
    linear_objective_vector : np.ndarray
        A vector determining the objective function v^T x
    bootstrap_table_sparse : csr_matrix
        The bootstrap table (represented as a sparse CSR matrix)
    linear_inhomogeneous_eq : tuple[csr_matrix, np.ndarray]
        A tuple (A, b) of a matrix A and a vector b, which together
        define a linear inhomogeneous equation of the form A x = b.
        This equation will be imposed directly as a constraint.
    linear_inhomogeneous_penalty : tuple[csr_matrix, np.ndarra]
        A tuple (A, b) of a matrix A and a vector b, which together
        define a linear inhomogeneous equation of the form A x = b.
        This equation will be imposed indirectly as a penalty term
        added to the objective function.
    init : np.ndarray
        An initial value for the variable vector x.
    radius : float, optional
        The maximum allowable change in the initial vector, by default np.inf
    maxiters : int, optional
        The maximum number of iterations for the SDP optimization, by default 10_000
    eps : float, optional
        A tolerance variable for the SDP optimization, by default 1e-4
    reg : float, optional
        A regularization coefficient controlling the relative weight of the
        penalty term, by default 1e-4
    verbose : bool, optional
        An optional boolean flag used to set the verbosity, by default True

    Returns
    -------
    tuple[bool, str, np.ndarray]
        A tuple containing
            a boolean variable, where True corresponds to a successful execution of the optimization
            a str containing the optimization status
            a numpy array with the final, optimized vector
    """
    # CVXPY method (only SCS is supported for this particular convex problem)
    if cvxpy_solver == "SCS":
        solver = cp.SCS
    elif cvxpy_solver == "ECOS":
        solver = cp.ECOS
    elif cvxpy_solver == "OSQP":
        solver = cp.OSQP
    elif cvxpy_solver == "MOSEK":
        solver = cp.MOSEK
    else:
        raise NotImplementedError

    # set-up
    matrix_dim = int(np.sqrt(bootstrap_table_sparse.shape[0]))
    num_variables = bootstrap_table_sparse.shape[1]
    param = cp.Variable(num_variables) # declare the cvxpy param

    # build the constraints
    # 1. the PSD bootstrap constraint(s)
    # 2. A @ param == 0
    # 3. ||param||_2 <= radius

    # If the n x n bootstrap table is complex, split the corresponding bootstrap matrix
    # into a (2n, 2n) real matrix, rather than a (n, n) complex matrix.
    # Note: based on the param vector, it could be the case that a complex-valued bootstrap
    # table gives rise to a real bootstrap matrix - e.g., if odd-degree expectation values
    # vanish for the given param vector.
    if np.max(np.abs(bootstrap_table_sparse.imag)) > 1e-10:
        debug("Mapping complex bootstrap matrix to real")
        bootstrap_table_sparse_real = bootstrap_table_sparse.real
        bootstrap_table_sparse_imag = bootstrap_table_sparse.imag
        matrix_real = cp.reshape(bootstrap_table_sparse_real @ param, (matrix_dim, matrix_dim))
        matrix_imag = cp.reshape(bootstrap_table_sparse_imag @ param, (matrix_dim, matrix_dim))
        matrix_block = cp.bmat(
            [[matrix_real, -matrix_imag], [matrix_imag, matrix_real]]
        )
        constraints = [matrix_block >> 0]
    else:
        constraints = [cp.reshape(bootstrap_table_sparse @ param, (matrix_dim, matrix_dim)) >> 0]

    constraints += [
        linear_inhomogeneous_eq[0] @ param == linear_inhomogeneous_eq[1]
    ]
    constraints += [cp.norm(param) <= radius]

    # the loss to minimize
    loss = linear_objective_vector @ param

    # add possible penalty terms
    if linear_inhomogeneous_penalty is not None:
        penalty = cp.norm(
            linear_inhomogeneous_penalty[0] @ param - linear_inhomogeneous_penalty[1]
        )
        loss += penalty_reg * penalty
    #if reg is not None:
    loss += reg * cp.norm(param)
    #debug(f"ADDING NORM REG={reg}")
    #debug(loss)

    # solve the optimization problem
    prob = cp.Problem(cp.Minimize(loss), constraints)
    if cvxpy_solver == 'SCS':
        prob.solve(
            verbose=verbose,
            max_iters=maxiters,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            eps_infeas=eps_infeas,
            solver=solver,
        )
    elif cvxpy_solver == "MOSEK":
        prob.solve(
            verbose=verbose,
            #max_iters=maxiters,
            #eps_abs=eps_abs,
            #eps_rel=eps_rel,
            #eps_infeas=eps_infeas,
            solver=solver,
            accept_unknown=True, # https://www.cvxpy.org/tutorial/solvers/index.html
            mosek_params={
                "MSK_DPAR_OPTIMIZER_MAX_TIME": 100,
                "MSK_DPAR_BASIS_TOL_S": 1e-8,
                "MSK_DPAR_BASIS_TOL_X": 1e-8,
                "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
                "MSK_IPAR_SIM_MAX_ITERATIONS": 30_000_000
                },
            )

    if param.value is None:
        raise ValueError("sdp_minimize failed, None value returned.")

    # log information on the extent to which the constraints are satisfied
    ball_constraint = np.linalg.norm(param.value)
    violation_of_linear_constraints = np.linalg.norm(
        linear_inhomogeneous_eq[0] @ param.value - linear_inhomogeneous_eq[1]
    )
    min_bootstrap_eigenvalue = np.linalg.eigvalsh(
        (bootstrap_table_sparse @ param.value).reshape(matrix_dim, matrix_dim)
    )[0]
    debug(f"sdp_minimize status after maxiters_cvxpy {maxiters}: {prob.status}")
    #debug(f"sdp_minimize ||x||: {ball_constraint:.4e}")
    debug(f"sdp_minimize ||A x - b||: {violation_of_linear_constraints:.4e}")
    debug(
        f"sdp_minimize bootstrap matrix min eigenvalue: {min_bootstrap_eigenvalue:.4e}"
    )

    optimization_result = {
        "solver": cvxpy_solver,
        "prob.status": prob.status,
        "prob.value": prob.value,
        "maxiters_cvxpy": maxiters,
        "||x||": ball_constraint,
        "violation_of_linear_constraints": violation_of_linear_constraints,
        "min_bootstrap_eigenvalue": min_bootstrap_eigenvalue,
    }

    return param.value, optimization_result


def solve_bootstrap(
    bootstrap: BootstrapSystem,
    st_operator_to_minimize: SingleTraceOperator,
    st_operator_inhomo_constraints=[(SingleTraceOperator(data={(): 1}), 1)],
    init: Optional[np.ndarray] = None,
    maxiters: int = 25,
    maxiters_cvxpy: int = 2500,
    tol: float = 1e-5,
    init_scale: float = 1.0,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    eps_infeas: float = 1e-7,
    reg: float = 1e-4,
    penalty_reg: float = 1e6,
    penalty_reg_decay_rate: Optional[float] = None,
    PRNG_seed=None,
    radius: float = 1e8,
    cvxpy_solver: str = "SCS",
) -> np.ndarray:
    """
    Solve the bootstrap by minimizing the objective function subject to
    the bootstrap constraints.

    TODO: add more info on the Newton's method used here.

    Parameters
    ----------
    bootstrap : BootstrapSystem
        The bootstrap system to be solved
    st_operator_to_minimize : SingleTraceOperator
        The single-trace operator whose expectation value we wish to minimize
    st_operator_inhomo_constraints : list, optional
        The single-trace expectation value constraints, <tr(O)>=c,
        by default [ (SingleTraceOperator(data={(): 1}), 1) ]
    init : Optional[np.ndarray], optional
        The initial parameter vector, by default None
    maxiters : int, optional
        The maximum number of iterations for the optimization, by default 25
    tol : float, optional
        The tolerance for the quadratic constraint violations, by default 1e-8
    init_scale : float, optional
        An overall scale for the parameter vector, by default 1.0
    eps : float, optional
        The epsilon used in the cvxpy inner optimization problem, by default 1e-4
    reg : float, optional
        The regularization parameter for the penalty terms, by default 1e-4

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if PRNG_seed is not None:
        np.random.seed(PRNG_seed)
        debug(f"setting PRNG seed to {PRNG_seed}")

    # get the bootstrap constraints necessary for the optimization
    # linear constraints
    if bootstrap.linear_constraints is None:
        _ = bootstrap.build_linear_constraints().tocsr()

    # quadratic constraints
    if bootstrap.quadratic_constraints_numerical is None:
        bootstrap.build_quadratic_constraints()
    quadratic_constraints_numerical = bootstrap.quadratic_constraints_numerical

    # bootstrap table
    if bootstrap.bootstrap_table_sparse is None:
        bootstrap.build_bootstrap_table()
    bootstrap_table_sparse = bootstrap.bootstrap_table_sparse

    # confirm bootstrap table is consistent with hermitian bootstrap matrix
    debug(f"Boostrap matrix dtype: {bootstrap_table_sparse.dtype}")
    bootstrap_matrix_tmp = bootstrap_table_sparse @ np.random.normal(size=bootstrap.param_dim_null)
    bootstrap_matrix_tmp = bootstrap_matrix_tmp.reshape((bootstrap.bootstrap_matrix_dim, bootstrap.bootstrap_matrix_dim))
    if not ishermitian(bootstrap_matrix_tmp, atol=1e-12):
        raise ValueError("Error, bootstrap matrix is not Hermitian.")

    debug(f"Final bootstrap parameter dimension: {bootstrap.param_dim_null}")
    # initialize the variable vector
    if init is None:
        debug(f"Initializing randomly")
        init = init_scale * np.random.normal(size=bootstrap.param_dim_null)
    else:
        init = np.asarray(init)
        debug(f"Initializing as param={init}")
    param = init

    # map the single trace operator whose expectation value we wish to minimize to a coefficient vector
    linear_objective_vector = bootstrap.single_trace_to_coefficient_vector(
        st_operator_to_minimize, return_null_basis=True
    )
    if not np.allclose(
        linear_objective_vector.imag, np.zeros_like(linear_objective_vector)
    ):
        raise ValueError("Error, the coefficient vector is complex but should be real.")
    linear_objective_vector = linear_objective_vector.real

    # build the A, b matrix and vector for the linear inhomogeneous constraints
    # this will always include the constraint that tr(1) = 1, and possibly other constraints as well
    A = sparse.csr_matrix((0, bootstrap.param_dim_null))
    b = np.zeros(0)
    for op, value in st_operator_inhomo_constraints:

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
    linear_inhomogeneous_eq_no_quadratic = (A, b)

    # iterate over steps
    for step in range(maxiters):

        debug(f"\n\nstep: {step+1}/{maxiters}")
        debug(f"PRNG seed: {PRNG_seed}")
        debug(f"radius: {radius:.4e}")
        debug(f"reg: {reg:.4e}")
        debug(f"penalty_reg: {penalty_reg:.4e}")
        debug(f"eps_abs: {eps_abs:.4e}")
        debug(f"eps_rel: {eps_rel:.4e}")
        debug(f"eps_infeas: {eps_infeas:.4e}")
        debug(f"init_scale: {init_scale:.4e}")
        debug(f"tol: {tol:.4e}")
        debug(f"cvxpy_solver: {cvxpy_solver}")
        for pair in st_operator_inhomo_constraints:
            debug(f"st_operator_inhomo_constraints: {pair[0]}, val={pair[1]:.4f}")
        debug(f"st_op_to_minimize: {st_operator_to_minimize}")

        # build the Newton method update for the quadratic constraints, which
        # produces a second inhomogenous linear equation A' x = b'
        # here, the new equation is grad * (new_param - param) + val = 0
        # this equation will be imposed as a penalty
        quad_cons_val, quad_cons_grad = get_quadratic_constraint_vector(
            quadratic_constraints_numerical, param, compute_grad=True
        )

        # how to handle the quadratic constraints (in progress)
        if step < 4:
            # only use Ax=b for the non-quadratic constraints
            # impose the quadratic constraints via a penalty term
            debug(f"Not using Ax=b for quadratic constraints")
            linear_inhomogeneous_eq = linear_inhomogeneous_eq_no_quadratic
        else:

            debug(f"Using Ax=b for quadratic constraints")
            linear_inhomogeneous_eq = (
                sparse.vstack(
                    [linear_inhomogeneous_eq_no_quadratic[0], quad_cons_grad]
                ),
                np.append(
                    linear_inhomogeneous_eq_no_quadratic[1],
                    np.asarray(quad_cons_grad.dot(param) - quad_cons_val)[0],
                ),
            )
            penalty_reg = 0

        linear_inhomogeneous_penalty = (
            quad_cons_grad,
            np.asarray(quad_cons_grad.dot(param) - quad_cons_val)[0],
        )

        # perform the inner convex minimization
        try:
            param, optimization_result = sdp_minimize(
                linear_objective_vector=linear_objective_vector,
                bootstrap_table_sparse=bootstrap_table_sparse,
                linear_inhomogeneous_eq=linear_inhomogeneous_eq,
                linear_inhomogeneous_penalty=linear_inhomogeneous_penalty,
                #init=param,
                # verbose=verbose, # don't need to print out cvxpy info
                radius=radius,
                maxiters=maxiters_cvxpy,
                penalty_reg=penalty_reg,
                reg=reg,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                eps_infeas=eps_infeas,
                cvxpy_solver=cvxpy_solver,
            )
        except:
            return None, None

        # print out some diagnostic information
        quad_cons_val = get_quadratic_constraint_vector(
            quadratic_constraints_numerical, param, compute_grad=False
        )
        max_quad_constraint_violation = np.max(np.abs(quad_cons_val))
        quad_constraint_violation_norm = np.linalg.norm(quad_cons_val)
        optimization_result["max_quad_constraint_violation"] = (
            max_quad_constraint_violation
        )
        optimization_result["quad_constraint_violation_norm"] = (
            quad_constraint_violation_norm
        )

        # debug(f"penalty_reg * ||q_I|| = {penalty_reg * quad_constraint_violation_norm:.4e}")
        debug(f"norm(param) = {np.linalg.norm(param):.4e}")
        # debug(f"penalty_reg * quad_constraint_violation_norm < tol = {penalty_reg * quad_constraint_violation_norm < tol}")
        debug(
            f"max violation of quadratic constraints: {max_quad_constraint_violation:.4e}"
        )
        debug(f"objective: {linear_objective_vector @ param:.4f}")

        # add the seed to the result
        optimization_result["PRNG_seed"] = PRNG_seed

        # record the boostrap matrix
        optimization_result["bootstrap_matrix_real"] = bootstrap.get_bootstrap_matrix(param=param).real.tolist()
        optimization_result["bootstrap_matrix_imag"] = bootstrap.get_bootstrap_matrix(param=param).imag.tolist()

        # terminate early if the tolerance is satisfied
        # if penalty_reg * quad_constraint_violation_norm < tol:
        if step > (10 - 2) and max_quad_constraint_violation < tol:
            return param, optimization_result

        # decay the regularization parameter
        if penalty_reg_decay_rate is not None:
            penalty_reg = penalty_reg * penalty_reg_decay_rate

    return param, optimization_result
