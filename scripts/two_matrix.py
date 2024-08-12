import numpy as np
from bmn.config_utils import generate_configs_two_matrix, run_all_configs


## minimize energy, set g2 = 1 held fixed
L = 3
dir_name = f"TwoMatrix_L_{L}_symmetric"
for g4 in np.linspace(0.05, 15, 101):
    g4 = float(np.round(g4, decimals=6))
    generate_configs_two_matrix(
        config_filename=f"g4_{str(g4)}",
        config_dir=dir_name,
        checkpoint_path=dir_name,
        g2=1,
        g4=g4,
        max_degree_L=L,
        impose_symmetries=True,
        )
# execute
run_all_configs(config_dir=dir_name, parallel=True)