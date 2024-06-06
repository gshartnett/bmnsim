import torch
import torch.optim as optim
from torch.nn import ReLU

return_null_basis = True

# build the Ax = b constraint
Avec = bootstrap.single_trace_to_coefficient_vector(
    SingleTraceOperator(data={(): 1}), return_null_basis=return_null_basis
)
Avec = torch.from_numpy(Avec).type(torch.float)

# Hamiltonian vector
Hvec = bootstrap.single_trace_to_coefficient_vector(
    bootstrap.hamiltonian, return_null_basis=return_null_basis
)
Hvec = torch.from_numpy(Hvec).type(torch.float)

# build the bootstrap array
bootstrap_array_sparse = bootstrap.build_bootstrap_table().todense()
bootstrap_array_torch = torch.from_numpy(bootstrap_array_sparse).type(torch.float)

# build the constraints
quadratic_constraints = bootstrap.build_quadratic_constraints(
    impose_linear_constraints=return_null_basis
)
quadratic_constraint_linear = torch.from_numpy(quadratic_constraints["linear"]).type(
    torch.float
)
quadratic_constraint_quadratic = torch.from_numpy(
    quadratic_constraints["quadratic"]
).type(torch.float)


def energy(param):
    return Hvec @ param


def get_quadratic_constraint_vector(param):
    quadratic_constraints = torch.einsum(
        "Iab, a, b -> I", quadratic_constraint_quadratic, param, param
    ) + torch.einsum("Ia, a -> I", quadratic_constraint_linear, param)
    return torch.square(quadratic_constraints)


def quadratic_loss(param):
    return torch.sum(get_quadratic_constraint_vector(param))


def Axb_loss(param):
    return torch.square(Avec @ param - 1)


def psd_loss(param):
    bootstrap_matrix = (bootstrap_array_torch @ param).reshape(
        (7, 7)
    )  # is this reshaping correct?
    smallest_eigv = torch.linalg.eigvalsh(bootstrap_matrix)[0]
    return ReLU()(-smallest_eigv)


def build_loss(param):
    lambda_psd = 1
    lambda_quadratic = 1
    lambda_Axb = 1
    loss = (
        energy(param)
        + lambda_psd * psd_loss(param)
        + lambda_quadratic * quadratic_loss(param)
        + lambda_Axb * Axb_loss(param)
    )
    return loss


## full param
if return_null_basis:
    param = torch.randn(bootstrap.param_dim_null, requires_grad=True)
else:
    param = torch.randn(bootstrap.param_dim, requires_grad=True)
optimizer = optim.SGD([param], lr=1e-3)

# Training loop
num_epochs = 100_000
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Clear previous gradients

    loss = build_loss(param)  # Compute the loss
    loss.backward()  # Compute gradients
    optimizer.step()  # Update the parameters

    # Print the loss for monitoring
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
