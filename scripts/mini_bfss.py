import numpy as np
from bmn.config_utils import generate_configs_bfss, run_all_configs

# generate the config files
L = 3
#for energy in np.exp(np.linspace(np.log(0.1), np.log(100), 50)):
for energy in np.linspace(0.5, 2.5, 21):
    energy = float(np.round(energy, decimals=6))

    generate_configs_bfss(
        config_filename=f"energy_{str(energy)}",
        config_dir=f"MiniBFSS_L_{L}_symmetric",
        st_operator_to_minimize="x_2",
        st_operators_evs_to_set={"energy": energy},
        load_from_previously_computed=True,
        impose_symmetries=True,
        tol=1e-1,
        maxiters=30,
        maxiters_cvxpy=20_000,
        init_scale=1e1,
        )

# execute
run_all_configs(config_dir=f"MiniBFSS_L_{L}_symmetric", parallel=True)