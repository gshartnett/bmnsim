import numpy as np
from bmn.config_utils import generate_configs_two_matrix, run_all_configs


for L in [3, 4]:

    # generate the config files
    for g4 in np.linspace(0.05, 10, 101):
        g4 = float(np.round(g4, decimals=6))

        generate_configs_two_matrix(
            config_filename=f"g4_{str(g4)}",
            config_dir=f"TwoMatrix_L_{L}_symmetric",
            g2=1,
            g4=g4,
            max_degree_L=L,
            impose_symmetries=True,
            maxiters_cvxpy=5_000,
            maxiters=100,
            radius=1e6,
            reg=1e6,
            )

    # execute
    run_all_configs(config_dir=f"TwoMatrix_L_{L}_symmetric", parallel=True)