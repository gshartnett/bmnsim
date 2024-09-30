import numpy as np
from bmn.config_utils import generate_configs_bmn, run_all_configs

# generate the config files
L = 3
lambd = 1

for st_operator_to_minimize in ["x_2", "x_4", "neg_x_2", "neg_commutator_squared"]:
    #for nu in np.linspace(0.1, 10, 100):
    for nu in [1.0]:
        for energy in np.linspace(-10, 30, 81):

            nu = float(np.round(nu, decimals=6))
            energy = float(np.round(energy, decimals=6))

            checkpoint_path = f"MiniBMN_L_{L}_symmetric_nu_{nu}_lamb_{lambd}"
            config_dir = f"MiniBMN_L_{L}_symmetric_tmp"
            config_filename = f"nu_{nu}_lambd_{lambd}_energy_{energy}_st_operator_to_minimize_{st_operator_to_minimize}"

            generate_configs_bmn(
                config_filename=config_filename,
                config_dir=config_dir,
                nu=nu,
                lambd=lambd,
                max_degree_L=L,
                load_from_previously_computed=True,
                checkpoint_path=checkpoint_path,
                impose_symmetries=True,
                odd_degree_vanish=False,
                st_operator_to_minimize=st_operator_to_minimize,
                st_operators_evs_to_set={"energy": energy},
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
    check_if_exists_already=False
    )