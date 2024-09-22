import numpy as np
from bmn.config_utils import generate_configs_three_matrix, run_all_configs

L = 2

lambd = 1
nu = 1

g2 = nu**2
g3 = float(3 * nu * np.sqrt(lambd))
g4 = lambd

generate_configs_three_matrix(
    config_filename="test",
    config_dir=f"ThreeMatrix_L_{L}_test",
    g2=g2,
    g3=g3,
    g4=g4,
    max_degree_L=L,
    impose_symmetries=True,
    load_from_previously_computed=False,
    odd_degree_vanish=False,
    simplify_quadratic=True,
    optimization_method="newton",
    cvxpy_solver='MOSEK',
    #cvxpy_solver='SCS',
    maxiters_cvxpy=250_000,
    #radius=1e2,
    #init_scale=0,
    reg=1e2,
    )

# execute
run_all_configs(config_dir=f"ThreeMatrix_L_{L}_test", parallel=False, check_if_exists_already=False)