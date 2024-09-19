"""
Check the number of unconstrained terms after imposing SO(3)
symmetry against the number of isotropic tensors in d=3,
as listed here:

Kearsley, Elliot A., and Jeffrey T. Fong. "Linearly independent sets of isotropic Cartesian tensors of ranks up to eight."
J. Res. Natl. Bur. Stand., Sect. B 79 (1975): 49.
"""

import pickle
from sparseqr import qr
from bmn.linear_algebra import create_sparse_matrix_from_dict
from bmn.bootstrap import BootstrapSystem
from bmn.models import MiniBFSS

d = 3
L = 3
model = MiniBFSS(couplings={"lambda": 1})
checkpoint_path = f"checkpoints/MiniBFSS_L_{L}_symmetric"

bootstrap = BootstrapSystem(
    matrix_system=model.matrix_system,
    hamiltonian=model.hamiltonian,
    gauge_generator=model.gauge_generator,
    max_degree_L=L,
    symmetry_generators=model.symmetry_generators,
    checkpoint_path=checkpoint_path,
    verbose=False,
    )

# build the symmetries
#symmetry_constrained = bootstrap.generate_symmetry_constraints()
#symmetry_constrained_cleaned = bootstrap.clean_constraints(symmetry_constrained)
#with open("data/tmp/BFSS_L_3_linear_constraints_data.pkl", "wb") as f:
#    pickle.dump(symmetry_constrained_cleaned, f)


# load the symmetries
with open("data/tmp/BFSS_L_3_linear_constraints_data.pkl", "rb") as f:
    symmetry_constrained_cleaned = pickle.load(f)

# number of independent isotropic tensors by rank
num_isotropic_tensors_of_rank_n = {
    1: 0,
    2: 1,
    3: 1,
    4: 3,
    5: 6,
    6: 15,
    7: 45,
    8: 91,
}

# loop over ranks
for total_rank in [2, 3, 4, 5, 6]:

    # get subset of constraints for a given rank
    symmetry_constraints_by_rank = [op for op in symmetry_constrained_cleaned if op.max_degree == total_rank]

    # build the index-value dict
    index_value_dict = {}
    for idx_constraint, st_operator in enumerate(symmetry_constraints_by_rank):
        for op_str, coeff in st_operator:
            index_value_dict[(idx_constraint, bootstrap.operator_dict[op_str])] = coeff

    # return the constraint matrix
    linear_constraint_matrix = create_sparse_matrix_from_dict(
        index_value_dict=index_value_dict,
        matrix_shape=(len(symmetry_constraints_by_rank), len(bootstrap.operator_list)),
    )

    _, _, _, num_independent_constraints = qr(linear_constraint_matrix.transpose())
    num_isotropic_tensors = 2**total_rank * num_isotropic_tensors_of_rank_n[total_rank]
    check = ((2*d)**total_rank) == (num_independent_constraints + num_isotropic_tensors)

    print(f"rank: {total_rank}:")
    print(f"  terms: {(2*d)**total_rank}, constraints: {num_independent_constraints}, isotropic tensors: {num_isotropic_tensors}")
    print(f"  check: {check}")