import numpy as np
from bmn.config_utils import generate_configs_bmn, run_all_configs, run_bootstrap_from_config

# generate the config files
L = 3
lambd = 1
PRNG_seed = 0


nu = 1.0
for st_operator_to_minimize in ["x_2", "neg_x_2"]:
#    for nu in np.linspace(0.1, 10, 100):
    #for energy in np.linspace(0, 10, 41):
    for energy in np.linspace(-10, 0, 41):
            nu = float(np.round(nu, decimals=6))
            energy = float(np.round(energy, decimals=6))

            config_dir = f"MiniBMN_L_{L}_symmetric_bound_x_2_pytorch"
            checkpoint_path = f"MiniBMN_L_{L}_symmetric_nu_{nu}_lamb_{lambd}"
            config_filename = f"operator_{st_operator_to_minimize}_nu_{str(nu)}_energy_{str(energy)}"

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
                #optimization_method="newton_Axb",
                optimization_method="pytorch",
                lr=1e0,
                init_scale=1e-2,
                patience=np.inf,
                PRNG_seed=PRNG_seed,
                )

# execute
'''
run_bootstrap_from_config(
    config_filename=config_filename,
    config_dir=config_dir,
    check_if_exists_already=False,
)
'''

run_all_configs(
    config_dir=config_dir,
    parallel=False,
    max_workers=6,
    check_if_exists_already=True
    )
