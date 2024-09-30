import numpy as np
from bmn.config_utils import generate_configs_bmn, run_all_configs, run_bootstrap_from_config

# generate the config files
L = 3
lambd = 1
nu = 1.0
energy = 1
st_operator_to_minimize = "neg_commutator_squared"

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
    st_operator_to_minimize=st_operator_to_minimize,
    st_operators_evs_to_set={"energy": energy},
    optimization_method="newton",
    cvxpy_solver='MOSEK',
    maxiters=30,
    init_scale=1e-2,
    reg=1e-4,
    penalty_reg=1e4,
    tol=1e-6,
    radius=1e5,
    )

# execute
run_all_configs(
    config_dir=f"MiniBMN_L_{L}_test",
    parallel=False,
    check_if_exists_already=False
    )