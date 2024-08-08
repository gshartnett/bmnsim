import numpy as np
from bmn.config_utils import generate_configs_bmn, run_all_configs

# generate the config files
L = 2

generate_configs_bmn(
    config_filename=f"test",
    config_dir=f"MiniBMN_L_{L}_symmetric",
    load_from_previously_computed=True,
    impose_symmetries=True,
    tol=1e-1,
    maxiters=30,
    maxiters_cvxpy=20_000,
    init_scale=1e1,
    )

# execute
run_all_configs(config_dir=f"MiniBMN_L_{L}_symmetric", parallel=True)