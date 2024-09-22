import numpy as np
from bmn.config_utils import generate_configs_two_matrix, run_all_configs

L = 3
g4 = 1
g2 = 1

generate_configs_two_matrix(
    config_filename="test",
    config_dir=f"TwoMatrix_L_{L}_test",
    g2=g2,
    g4=g4,
    max_degree_L=L,
    impose_symmetries=True,
    load_from_previously_computed=False,
    odd_degree_vanish=True,
    simplify_quadratic=True,
    optimization_method="newton",
    maxiters_cvxpy=250_000,
    cvxpy_solver="MOSEK",
    )

# execute
run_all_configs(config_dir=f"TwoMatrix_L_{L}_test", parallel=False, check_if_exists_already=False)