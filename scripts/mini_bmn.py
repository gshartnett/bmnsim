import numpy as np
from bmn.config_utils import generate_configs_bmn, run_all_configs

# generate the config files
L = 3
dir_name = f"MiniBMN_L_{L}_symmetric"

generate_configs_bmn(
    config_filename=f"test",
    config_dir=dir_name,
    g2=1,
    g4=1,
    max_degree_L=L,
    load_from_previously_computed=True,
    checkpoint_path=dir_name,
    impose_symmetries=True,
    simplify_quadratic=False,
    )

# execute
run_all_configs(config_dir=f"MiniBMN_L_{L}_symmetric", parallel=False)

# symmetries, fraction of operators, simplify_quadratic, load_From_previously_computed
# expectation values are complex in general