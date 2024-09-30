import numpy as np
from bmn.config_utils import generate_configs_bfss, run_all_configs, run_bootstrap_from_config

# generate the config files
L = 3

energy = 10.7
st_operator_to_minimize = "x_2"

generate_configs_bfss(
    config_filename="test",
    config_dir=f"MiniBFSS_L_{L}_test",
    checkpoint_path=f"MiniBFSS_L_{L}_symmetric",
    max_degree_L=L,
    st_operator_to_minimize=st_operator_to_minimize,
    st_operators_evs_to_set={"energy": energy},
    load_from_previously_computed=True,
    odd_degree_vanish=True,
    simplify_quadratic=True,
    #optimization_method='pytorch',
    optimization_method="newton",
    maxiters_cvxpy=250_000,
    cvxpy_solver='MOSEK',
    #init_scale=1e0,
    #lr=1e-1,
    #radius=1e3,
    reg=1e4,
    )

# execute
run_all_configs(
    config_dir=f"MiniBFSS_L_{L}_test",
    parallel=False,
    check_if_exists_already=False
    )