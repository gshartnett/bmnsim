import cvxpy as cp
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from typing import Optional
from bmn.algebra import SingleTraceOperator
from bmn.debug_utils import debug
from bmn.solver import (
    get_quadratic_constraint_vector_sparse as get_quadratic_constraint_vector,
)
from bmn.bootstrap import BootstrapSystem


def newton_pseudoinverse(param, nsteps, quadratic_constraints_numerical):

    for step in range(nsteps):
        val, grad = get_quadratic_constraint_vector(
            quadratic_constraints=quadratic_constraints_numerical,
            param=param,
            compute_grad=True,
            )
        grad_pinv = np.linalg.pinv(grad)
        #np.allclose(grad @ grad_pinv @ grad, grad)
        #np.allclose(grad.T @ np.linalg.inv(grad @ grad.T), grad_pinv)
        param = param - np.asarray(grad_pinv @ val)[0]
        debug(f"Newton's method: step = {step}, val = {np.linalg.norm(val)}")

    return param


def sdp_minimize(
    linear_objective_vector: np.ndarray,
    bootstrap_table_sparse: csr_matrix,
    linear_inhomogeneous_eq: tuple[csr_matrix, np.ndarray],
    init: np.ndarray,
    linear_inhomogeneous_penalty: Optional[tuple[csr_matrix, np.ndarray]]=None,
    radius: float = np.inf,
    maxiters: int = 2500,
    eps: float = 1e-4,
    reg: float = 1e6,
    verbose: bool = False,
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
    # initialize the cvxpy parameter vector in the null space
    num_variables = init.size
    param = cp.Variable(num_variables)

    # build the constraints
    # 1. the PSD bootstrap constraint(s)
    # 2. A @ param == 0
    # 3. ||param - init||_2 <= radius
    size = int(np.sqrt(bootstrap_table_sparse.shape[0]))
    constraints = [cp.reshape(bootstrap_table_sparse @ param, (size, size)) >> 0]
    constraints += [linear_inhomogeneous_eq[0] @ param == linear_inhomogeneous_eq[1]]
    constraints += [cp.norm(param - init) <= radius]

    # the 1ss to minimize
    if linear_inhomogeneous_penalty is None:
        loss = linear_objective_vector @ param
    else:
        penalty = cp.norm(
            linear_inhomogeneous_penalty[0] @ param - linear_inhomogeneous_penalty[1]
        )
        loss = linear_objective_vector @ param + reg * penalty

    # solve the optimization problem
    prob = cp.Problem(cp.Minimize(loss), constraints)
    prob.solve(verbose=verbose, max_iters=maxiters, eps=eps, solver=cp.SCS)

    if param.value is None:
        return None, None

    # log information on the extent to which the constraints are satisfied
    ball_constraint = np.linalg.norm(param.value- init)
    violation_of_linear_constraints = np.linalg.norm(linear_inhomogeneous_eq[0] @ param.value - linear_inhomogeneous_eq[1])
    min_bootstrap_eigenvalue = np.linalg.eigvalsh((bootstrap_table_sparse @ param.value).reshape(size, size))[0]
    debug(f"sdp_minimize status after maxiters_cvxpy {maxiters}: {prob.status}")
    debug(f"sdp_minimize ||x - init||: {ball_constraint:.4e}")
    debug(f"sdp_minimize ||A x - b||: {violation_of_linear_constraints}")
    debug(f"sdp_minimize bootstrap matrix min eigenvalue: {min_bootstrap_eigenvalue}")

    optimization_result = {
        "prob.status": prob.status,
        "prob.value": prob.value,
        "maxiters_cvxpy": maxiters,
        "||x-init||": ball_constraint,
        "violation_of_linear_constraints": violation_of_linear_constraints,
        "min_bootstrap_eigenvalue": min_bootstrap_eigenvalue,
    }

    return param.value, optimization_result


def solve_bootstrap(
    bootstrap: BootstrapSystem,
    st_operator_to_minimize: SingleTraceOperator,
    st_operator_inhomo_constraints=[
        (SingleTraceOperator(data={(): 1}), 1)
        ],
    init:Optional[np.ndarray]=None,
    maxiters:int=25,
    maxiters_cvxpy:int=2500,
    tol:float=1e-5,
    init_scale:float=1.0,
    eps: float = 1e-4,
    reg: float = 1e6,
    use_newton = True,
    radius: float = 1e8,
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
    #np.random.seed(123)
    #debug(f"setting PRNG seed!")

    #reg_min = 1e4
    #reg_max = 1e7
    #reg_schedule = np.exp(np.linspace(np.log(reg_min), np.log(reg_max), maxiters))

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

    debug(f"Final bootstrap parameter dimension: {bootstrap.param_dim_null}")
    # initialize the variable vector
    if init is None:
        debug(f"Initializing randomly")
        init = init_scale * np.random.normal(size=bootstrap.param_dim_null)
    param = init

    # map the single trace operator whose expectation value we wish to minimize to a coefficient vector
    linear_objective_vector = bootstrap.single_trace_to_coefficient_vector(
        st_operator_to_minimize,
        return_null_basis=True
        )
    if not np.allclose(linear_objective_vector.imag, np.zeros_like(linear_objective_vector)):
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

    if False:
        param = newton_pseudoinverse(
            param=param,
            nsteps=15,
            quadratic_constraints_numerical=quadratic_constraints_numerical
            )

    # iterate over steps
    #for step in range(maxiters):
    step = 0
    while step < maxiters:

        debug(f"step = {step+1}/{maxiters}")
        #print(f"param[0] = {param[0]}")

        # build the Newton method update for the quadratic constraints, which
        # produces a second inhomogenous linear equation A' x = b'
        # here, the new equation is grad * (new_param - param) + val = 0
        # this equation will be imposed as a penalty

        quad_cons_val, quad_cons_grad = get_quadratic_constraint_vector(
            quadratic_constraints_numerical, param, compute_grad=True
        )
        #grad_pinv = np.linalg.pinv(quad_cons_grad)

        # how to handle the quadratic constraints (in progress)
        if step < 5:
            # only use Ax=b for the non-quadratic constraints
            # impose the quadratic constraints via a penalty term
            debug(f"Not using Ax=b for quadratic constraints")
            linear_inhomogeneous_eq = linear_inhomogeneous_eq_no_quadratic
            linear_inhomogeneous_penalty = (quad_cons_grad, np.asarray(quad_cons_grad.dot(param) - quad_cons_val)[0])

        else:
            # TODO still working on what I want to do here
            #linear_inhomogeneous_penalty = None
            debug(f"Using Ax=b for quadratic constraints")
            linear_inhomogeneous_penalty = (quad_cons_grad, np.asarray(quad_cons_grad.dot(param) - quad_cons_val)[0])
            linear_inhomogeneous_eq = (
                sparse.vstack([linear_inhomogeneous_eq_no_quadratic[0], quad_cons_grad]),
                np.append(linear_inhomogeneous_eq_no_quadratic[1], np.asarray(quad_cons_grad.dot(param) - quad_cons_val)[0])
                )

        #debug(f"radius={radius:.4e}, maxiters_cvxpy={maxiters_cvxpy}, reg={reg:.4e}, eps={eps:.4e}, init_scale={init_scale:.4e}")

        # perform the inner convex minimization
        param, optimization_result = sdp_minimize(
            linear_objective_vector=linear_objective_vector,
            bootstrap_table_sparse=bootstrap_table_sparse,
            linear_inhomogeneous_eq=linear_inhomogeneous_eq,
            linear_inhomogeneous_penalty=linear_inhomogeneous_penalty,
            init=param,
            #verbose=verbose, # don't need to print out cvxpy info
            radius=radius,
            maxiters=maxiters_cvxpy,
            reg=reg,
            eps=eps,
        )

        if param is None:
            param = init
            reg *= 0.75
            step = 0
            debug(f"param is None, resetting step=0 and reducing reg to reg={reg:.4e}")
            continue
        step += 1

        # print out some diagnostic information
        quad_cons_val = get_quadratic_constraint_vector(
            quadratic_constraints_numerical, param, compute_grad=False
        )
        max_quad_constraint_violation = np.max(np.abs(quad_cons_val))
        quad_constraint_violation_norm = np.linalg.norm(quad_cons_val)
        optimization_result["max_quad_constraint_violation"] = max_quad_constraint_violation
        optimization_result["quad_constraint_violation_norm"] = quad_constraint_violation_norm
        debug(
            f"objective: {linear_objective_vector @ param:.4f}, max violation of quadratic constraints: {max_quad_constraint_violation:.4e}, reg * ||q_I|| = {reg * quad_constraint_violation_norm:.4e}"
        )

        # terminate early if the tolerance is satisfied
        if reg * quad_constraint_violation_norm < tol:
            return param, optimization_result

    return param, optimization_result