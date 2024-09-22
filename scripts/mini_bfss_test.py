import numpy as np
from bmn.config_utils import generate_configs_bfss, run_all_configs, run_bootstrap_from_config

# generate the config files
L = 3

generate_configs_bfss(
    config_filename="test",
    config_dir=f"MiniBFSS_L_{L}_test",
    checkpoint_path=f"MiniBFSS_L_{L}_symmetric",
    max_degree_L=L,
    impose_symmetries=True,
    load_from_previously_computed=False,
    odd_degree_vanish=False,
    simplify_quadratic=True,
    #optimization_method='pytorch',
    optimization_method="newton",
    maxiters_cvxpy=250_000,
    cvxpy_solver='MOSEK',
    #init_scale=1e0,
    #lr=1e-1,
    #radius=1e3,
    )

# execute
run_all_configs(
    config_dir=f"MiniBFSS_L_{L}_test",
    parallel=False,
    check_if_exists_already=False
    )