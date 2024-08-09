import numpy as np
from bmn.config_utils import generate_configs_bmn, run_all_configs

# generate the config files
L = 3

generate_configs_bmn(
    config_filename=f"test",
    config_dir=f"MiniBMN_L_{L}",
    g2=1,
    g4=1,
    max_degree_L=L,
    load_from_previously_computed=True,
    impose_symmetries=False,
    simplify_quadratic=False,
    tol=1e-1,
    maxiters=30,
    maxiters_cvxpy=20_000,
    init_scale=1e1,
    )

# execute
run_all_configs(config_dir=f"MiniBMN_L_{L}", parallel=True)

# symmetries, fraction of operators, simplify_quadratic, load_From_previously_computed
# expectation values are complex in general