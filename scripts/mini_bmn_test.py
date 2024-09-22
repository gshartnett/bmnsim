import numpy as np
from bmn.config_utils import generate_configs_bmn, run_all_configs, run_bootstrap_from_config

# generate the config files
L = 3
lambd = 1
nu = 1.0

generate_configs_bmn(
    config_filename="test",
    config_dir=f"MiniBMN_L_{L}_test",
    checkpoint_path=f"MiniBMN_L_{L}_symmetric_nu_{nu}_lamb_{lambd}",
    nu=nu,
    lambd=lambd,
    max_degree_L=L,
    impose_symmetries=True,
    load_from_previously_computed=True,
    odd_degree_vanish=False,
    simplify_quadratic=True,
    #optimization_method='pytorch',
    optimization_method="newton",
    #maxiters_cvxpy=25_000,
    cvxpy_solver='MOSEK',
    #init_scale=1e0,
    #lr=1e-1,
    #radius=1e3,
    reg=1e4,
    )

# execute
run_all_configs(
    config_dir=f"MiniBMN_L_{L}_test",
    parallel=False,
    check_if_exists_already=False
    )