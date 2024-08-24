from typing import Optional

import cvxpy as cp
import numpy as np
import scipy.sparse as sparse
import torch
import torch.optim as optim
from torch.nn import ReLU
from torch.optim.lr_scheduler import ExponentialLR
#import lightning as L

from bmn.algebra import SingleTraceOperator
from bmn.bootstrap import BootstrapSystem
from bmn.debug_utils import debug
from bmn.solver_trustregion import (
    get_quadratic_constraint_vector_sparse as get_quadratic_constraint_vector,
)
from bmn.linear_algebra import get_null_space_dense

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")


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
    tol=1e-6,
    patience=100,
    early_stopping_tol=1e-3,
) -> np.ndarray:

    print(f"torch device: {device}")

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

    A_null_space = get_null_space_dense(matrix=A)
    null_space_projector = 0*np.eye(A.shape[1]) + A_null_space @ np.linalg.pinv(A_null_space)

    A = torch.from_numpy(A).type(torch.float).to(device)  # convert to torch tensor
    b = torch.from_numpy(b).type(torch.float).to(device)
    null_space_projector = torch.from_numpy(null_space_projector).type(torch.float).to(device)

    #eturn null_space_projector

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

    def operator_loss(param_null, param_particular):
        param = null_space_projector @ param_null + param_particular
        return vec @ param

    def get_quadratic_constraint_vector(param):
        quadratic_constraints = torch.einsum(
            "Iab, a, b -> I", quadratic_constraint_quadratic, param, param
        ) + torch.einsum("Ia, a -> I", quadratic_constraint_linear, param)
        return torch.square(quadratic_constraints)

    def quadratic_loss(param_null, param_particular):
        param = null_space_projector @ param_null + param_particular
        return torch.norm(get_quadratic_constraint_vector(param))

    def Axb_loss(param_null, param_particular):
        param = null_space_projector @ param_null + param_particular
        return torch.norm(A @ param - b)

    def psd_loss(param_null, param_particular):
        param = null_space_projector @ param_null + param_particular
        bootstrap_matrix = (bootstrap_array_torch @ param).reshape(
            (bootstrap.bootstrap_matrix_dim, bootstrap.bootstrap_matrix_dim)
        )
        smallest_eigv = torch.linalg.eigvalsh(bootstrap_matrix)[0]
        return ReLU()(-smallest_eigv)

    def build_loss(param_null, param_particular, penalty_reg=penalty_reg):
        loss = (
            operator_loss(param_null, param_particular)
            + penalty_reg * psd_loss(param_null, param_particular)
            + penalty_reg * quadratic_loss(param_null, param_particular)
            + penalty_reg * Axb_loss(param_null, param_particular)
        )
        return loss

    # initialize the variable vector
    if init is None:
        init = init_scale * np.random.randn(bootstrap.param_dim_null)
        param_particular = np.linalg.lstsq(A.cpu().numpy(), b.cpu().numpy())[0]
        param = init
        debug(f"Initializing param to be the least squares solution of Ax=b plus Gaussian noise with scale = {init_scale}.")
    else:
        raise ValueError
        param = init
        debug(f"Initializing as param={init}")

    param = torch.tensor(param).type(torch.float).to(device)
    param = null_space_projector @ param

    param.requires_grad = True
    param_particular = torch.tensor(param_particular).type(torch.float).to(device)

    print(f"Axb_loss={Axb_loss(param_null=param, param_particular=param_particular)}")
    print(f"A @ param = {A @ (param + param_particular)}")

    # optimizer
    optimizer = optim.Adam([param], lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    # early stopping stuff
    old_loss = np.inf
    early_stopping_counter = 0

    for epoch in range(num_epochs):

        optimizer.zero_grad()
        loss = build_loss(param_null=param, param_particular=param_particular)
        loss.backward()
        optimizer.step()
        scheduler.step()

        '''
        #with torch.no_grad():
        param.requires_grad = False
        delta_param = param.clone().detach() - old_param
        print(delta_param)
        delta_param = null_space_projector @ delta_param
        print(delta_param)
        param = old_param + delta_param
        param.requires_grad = True

        assert 1== 0
        '''

        with torch.no_grad():
            # early stopping
            # TODO consider moving to pytorch lightning, might be overkill
            current_loss = loss.detach().cpu().item()
            violation_of_linear_constraints = Axb_loss(param_null=param, param_particular=param_particular).detach().cpu().item()
            min_bootstrap_eigenvalue = psd_loss(param_null=param, param_particular=param_particular).detach().cpu().item()
            quad_constraint_violation_norm = quadratic_loss(param_null=param, param_particular=param_particular).detach().cpu().item()
            tolerance_satisfied = (violation_of_linear_constraints < tol) and np.abs(min_bootstrap_eigenvalue) < tol and (quad_constraint_violation_norm < tol)
            if np.abs(current_loss - old_loss) < early_stopping_tol:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
            if tolerance_satisfied and (early_stopping_counter > patience):
                debug(f"Early stopping triggered after {epoch + 1} epochs.")
                break
            old_loss = current_loss

            #if epoch > 4000 and violation_of_linear_constraints > 1e-2:
            #    break

            if (epoch + 1) % 100 == 0:
                total_loss = loss.detach().cpu().item()
                operator_value = operator_loss(param_null=param, param_particular=param_particular).detach().cpu().item()
                violation_of_linear_constraints = Axb_loss(param_null=param, param_particular=param_particular).detach().cpu().item()
                min_bootstrap_eigenvalue = psd_loss(param_null=param, param_particular=param_particular).detach().cpu().item()
                quad_constraint_violation_norm = quadratic_loss(param_null=param, param_particular=param_particular).detach().cpu().item()

                debug(
                    f"epoch: {epoch+1}/{num_epochs}, lr: {scheduler.get_last_lr()[0]:.3e} total_loss: {total_loss:.3e}: op_loss: {operator_value:.5f}, ||Ax-b||: {violation_of_linear_constraints:.3e}, min_eig: {min_bootstrap_eigenvalue:.3e}, ||quad cons||: {quad_constraint_violation_norm:.3e}"
                )

        # clean up
        del loss

    param_final = param + param_particular

    optimization_result = {
        "solver": "pytorch",
        "num_epochs": num_epochs,
        "operator_loss": float(operator_loss(param_null=param, param_particular=param_particular).detach().cpu().item()),
        "violation_of_linear_constraints": float(Axb_loss(param_null=param, param_particular=param_particular).detach().cpu().item()),
        "min_bootstrap_eigenvalue": float(psd_loss(param_null=param, param_particular=param_particular).detach().cpu().item()),
        "quad_constraint_violation_norm": float(quadratic_loss(param_null=param, param_particular=param_particular).detach().cpu().item()),
        "max_quad_constraint_violation": float(torch.max(
            torch.abs(get_quadratic_constraint_vector(param_final))
        )
        .detach()
        .cpu()
        .item()),
    }

    # convert to a list of floats (no numpy float types, so that the result can be saved as a json later)
    param_final = [float(x) for x in list(param_final.detach().cpu().numpy())]

    return param_final, optimization_result
