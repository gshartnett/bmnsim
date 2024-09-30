import numpy as np
from bmn.config_utils import generate_configs_two_matrix, run_all_configs, run_bootstrap_from_config


## energy held fixed
L = 3
g2 = 1
checkpoint_path = f"TwoMatrix_L_{L}_symmetric_{g2}"

config_dir = f"TwoMatrix_L_{L}_symmetric_energy_fixed_g2_{g2}"
#config_dir = f"TwoMatrix_L_{L}_symmetric_energy_fixed_g2_{g2}_pytorch"
#config_dir = f"TwoMatrix_L_{L}_symmetric_energy_fixed_g2_{g2}_newton_Axb"

for st_operator_to_minimize in ["x_2", "neg_x_2", "x_4", "neg_x_4"]:
    for energy in np.linspace(-10, 30, 81):
        energy = float(np.round(energy, decimals=6))
        generate_configs_two_matrix(
            config_filename=f"energy_{str(energy)}_op_to_min_{st_operator_to_minimize}",
            config_dir=config_dir,
            g2=g2,
            g4=1,
            st_operator_to_minimize=st_operator_to_minimize,
            st_operators_evs_to_set={"energy": energy},
            max_degree_L=L,
            load_from_previously_computed=True,
            checkpoint_path=checkpoint_path,
            impose_symmetries=True,
            optimization_method="newton",
            cvxpy_solver='MOSEK',
            maxiters=30,
            init_scale=1e-2,
            reg=1e1,
            penalty_reg=1e4,
            tol=1e-6,
            )

# execute
run_all_configs(
    config_dir=config_dir,
    parallel=True,
    max_workers=6,
    check_if_exists_already=True
    )