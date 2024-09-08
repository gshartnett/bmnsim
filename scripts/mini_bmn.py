import numpy as np
from bmn.config_utils import generate_configs_bmn, run_all_configs

# generate the config files
L = 3
nu = 1
lambd = 1
config_dir = f"MiniBMN_L_{L}_symmetric_nu_{nu}_lamb_{lambd}"
checkpoint_path = config_dir
checkpoint_path += "_pytorch"

generate_configs_bmn(
    config_filename=f"test",
    config_dir=config_dir,
    nu=1,
    lambd=1,
    max_degree_L=L,
    load_from_previously_computed=True,
    checkpoint_path=config_dir,
    impose_symmetries=True,
    odd_degree_vanish=False,
    #optimization_method="newton",
    optimization_method="pytorch",
    lr=1e0,
    init_scale=1e-2,
    )

# execute
run_all_configs(
    config_dir=config_dir,
    parallel=False,
    max_workers=3,
    check_if_exists_already=False
    )