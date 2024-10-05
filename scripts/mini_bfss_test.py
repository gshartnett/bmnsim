import numpy as np
from bmn.config_utils import generate_configs_bfss, run_all_configs, run_bootstrap_from_config

# generate the config files
L = 3

energy = 1.0
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
    optimization_method="pytorch",
    lr=1e-2,
    penalty_reg=1e3,
    #optimization_method="newton",
    #cvxpy_solver='MOSEK',
    #maxiters=30,
    #init_scale=1e-2,
    #reg=1e-5,
    #penalty_reg=0,
    #tol=1e-7,
    #radius=1e6,
    )

# execute
run_all_configs(
    config_dir=f"MiniBFSS_L_{L}_test",
    parallel=False,
    check_if_exists_already=False
    )