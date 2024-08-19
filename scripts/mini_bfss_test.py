import numpy as np
from bmn.config_utils import generate_configs_bfss, run_all_configs

# generate the config files
L = 3
dir_name = f"MiniBFSS_L_{L}_symmetric_test"
checkpoint_path = f"MiniBFSS_L_{L}_symmetric"

#for PRNG_seed in range(6):
generate_configs_bfss(
    config_filename=f"test",
    config_dir=dir_name,
    st_operator_to_minimize="energy",
    max_degree_L=L,
    load_from_previously_computed=True,
    checkpoint_path=checkpoint_path,
    #PRNG_seed=PRNG_seed,
    maxiters_cvxpy=10_000_000,
    eps=1e-6,
    )

# execute
run_all_configs(config_dir=dir_name, parallel=False)
