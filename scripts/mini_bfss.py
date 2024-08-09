import numpy as np
from bmn.config_utils import generate_configs_bfss, run_all_configs

# generate the config files
L = 3
for st_operator_to_minimize in ["neg_commutator_squared", "x_2", "x_4"]:
    for energy in np.exp(np.linspace(np.log(0.1), np.log(100), 40)):
    #for energy in np.linspace(0.5, 2.5, 21):
        energy = float(np.round(energy, decimals=6))

        generate_configs_bfss(
            config_filename=f"energy_{str(energy)}_op_to_min_{st_operator_to_minimize}",
            config_dir=f"MiniBFSS_L_{L}_symmetric",
            st_operator_to_minimize=st_operator_to_minimize,
            st_operators_evs_to_set={"energy": energy},
            max_degree_L=L,
            load_from_previously_computed=True,
            impose_symmetries=True,
            tol=1e-4,
            reg=1e7,
            #maxiters=50,
            maxiters_cvxpy=25_000,
            #init_scale=1e0,
            )

# execute
run_all_configs(config_dir=f"MiniBFSS_L_{L}_symmetric", parallel=True)