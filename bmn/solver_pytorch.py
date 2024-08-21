from typing import Optional

import cvxpy as cp
import numpy as np
import scipy.sparse as sparse
import torch
import torch.optim as optim
from torch.nn import ReLU
from torch.optim.lr_scheduler import ExponentialLR

from bmn.algebra import SingleTraceOperator
from bmn.bootstrap import BootstrapSystem
from bmn.debug_utils import debug
from bmn.solver_trustregion import (
    get_quadratic_constraint_vector_sparse as get_quadratic_constraint_vector,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def solve_bootstrap(
    bootstrap: BootstrapSystem,
    st_operator_to_minimize: SingleTraceOperator,
    st_operator_inhomo_constraints=[(SingleTraceOperator(data={(): 1}), 1)],
    init: Optional[np.ndarray] = None,
    PRNG_seed=None,
    init_scale: float = 1e-2,
    lr=1e-1,
    gamma=0.999,
    num_epochs=10_000,
    penalty_reg=1e2,
) -> np.ndarray:

    if PRNG_seed is not None:
        np.random.seed(PRNG_seed)
        torch.manual_seed(PRNG_seed)
        debug(f"setting PRNG seed to {PRNG_seed}")

    # get the bootstrap constraints necessary for the optimization
    # linear constraints
    if bootstrap.linear_constraints is None:
        _ = bootstrap.build_linear_constraints().tocsr()

    # quadratic constraints
    if bootstrap.quadratic_constraints_numerical is None:
        bootstrap.build_quadratic_constraints()

    # bootstrap table
    if bootstrap.bootstrap_table_sparse is None:
        bootstrap.build_bootstrap_table()
    debug(f"Final bootstrap parameter dimension: {bootstrap.param_dim_null}")

    # build the Ax = b constraints
    A, b = [], []
    for st_operator, val in st_operator_inhomo_constraints:
        A.append(
            bootstrap.single_trace_to_coefficient_vector(
                st_operator, return_null_basis=True
            )
        )
        b.append(val)
    A = np.asarray(A)  # convert to numpy array
    b = np.asarray(b)
    A = torch.from_numpy(A).type(torch.float).to(device)  # convert to torch tensor
    b = torch.from_numpy(b).type(torch.float).to(device)

    # get the vector of the operator to bound (minimize)
    vec = bootstrap.single_trace_to_coefficient_vector(
        st_operator_to_minimize, return_null_basis=True
    )
    vec = torch.from_numpy(vec).type(torch.float).to(device)

    # build the bootstrap array
    bootstrap_array_torch = (
        torch.from_numpy(bootstrap.bootstrap_table_sparse.todense())
        .type(torch.float)
        .to(device)
    )

    # build the constraints
    quadratic_constraints = bootstrap.quadratic_constraints_numerical
    quadratic_constraint_linear = (
        torch.from_numpy(quadratic_constraints["linear"].todense())
        .type(torch.float)
        .to(device)
    )
    quadratic_constraint_quadratic = (
        torch.from_numpy(quadratic_constraints["quadratic"].todense())
        .type(torch.float)
        .to(device)
    )
    quadratic_constraint_quadratic = quadratic_constraint_quadratic.reshape(
        (
            len(quadratic_constraint_quadratic),
            bootstrap.param_dim_null,
            bootstrap.param_dim_null,
        )
    )

    def operator_loss(param):
        return vec @ param

    def get_quadratic_constraint_vector(param):
        quadratic_constraints = torch.einsum(
            "Iab, a, b -> I", quadratic_constraint_quadratic, param, param
        ) + torch.einsum("Ia, a -> I", quadratic_constraint_linear, param)
        return torch.square(quadratic_constraints)

    def quadratic_loss(param):
        return torch.norm(get_quadratic_constraint_vector(param))

    def Axb_loss(param):
        return torch.norm(A @ param - b)

    def psd_loss(param):
        bootstrap_matrix = (bootstrap_array_torch @ param).reshape(
            (bootstrap.bootstrap_matrix_dim, bootstrap.bootstrap_matrix_dim)
        )
        smallest_eigv = torch.linalg.eigvalsh(bootstrap_matrix)[0]
        return ReLU()(-smallest_eigv)

    def build_loss(param, penalty_reg=penalty_reg):
        loss = (
            operator_loss(param)
            + penalty_reg * psd_loss(param)
            + penalty_reg * quadratic_loss(param)
            + penalty_reg * Axb_loss(param)
        )
        return loss

    # initialize the variable vector
    if init is None:
        init = init_scale * torch.randn(bootstrap.param_dim_null)
        param = torch.linalg.lstsq(A.cpu(), b.cpu()).solution + init
        debug(f"Initializing param to be the least squares solution of Ax=b plus Gaussian noise with scale = {init_scale}.")
    else:
        debug(f"Initializing as param={init}")
    param = torch.tensor(param).type(torch.float).to(device)
    param.requires_grad = True

    # Training loop
    optimizer = optim.Adam([param], lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = build_loss(param)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            total_loss = loss.detach().cpu().item()
            operator_value = operator_loss(param).detach().cpu().item()
            violation_of_linear_constraints = Axb_loss(param).detach().cpu().item()
            min_bootstrap_eigenvalue = psd_loss(param).detach().cpu().item()
            quad_constraint_violation_norm = quadratic_loss(param).detach().cpu().item()

            debug(
                f"epoch: {epoch+1}/{num_epochs}, lr: {scheduler.get_last_lr()[0]:.3e} total_loss: {total_loss:.3e}: op_loss: {operator_value:.5f}, ||Ax-b||: {violation_of_linear_constraints:.3e}, min_eig: {min_bootstrap_eigenvalue:.3e}, ||quad cons||: {quad_constraint_violation_norm:.3e}"
            )

    optimization_result = {
        "solver": "pytorch",
        "num_epochs": num_epochs,
        "operator_loss": float(operator_loss(param).detach().cpu().item()),
        "violation_of_linear_constraints": float(Axb_loss(param).detach().cpu().item()),
        "min_bootstrap_eigenvalue": float(psd_loss(param).detach().cpu().item()),
        "quad_constraint_violation_norm": float(quadratic_loss(param).detach().cpu().item()),
        "max_quad_constraint_violation": float(torch.max(
            torch.abs(get_quadratic_constraint_vector(param))
        )
        .detach()
        .cpu()
        .item()),
    }

    # convert to a list of floats (no numpy float types, so that the result can be saved as a json later)
    param_as_list = [float(x) for x in list(param.detach().cpu().numpy())]

    return param_as_list, optimization_result
