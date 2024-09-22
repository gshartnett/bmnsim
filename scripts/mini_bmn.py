import numpy as np
from bmn.config_utils import generate_configs_bmn, run_all_configs

# generate the config files
L = 3
lambd = 1
PRNG_seed = 0

#for st_operator_to_minimize in ["x_2", "x_4", "neg_commutator_squared"]:
for nu in np.linspace(0.1, 10, 100):
#for energy in np.linspace(-10, 0, 41):
    nu = float(np.round(nu, decimals=6))
    #energy = float(np.round(energy, decimals=6))

    config_dir = f"MiniBMN_L_{L}_symmetric"
    checkpoint_path = f"MiniBMN_L_{L}_symmetric_nu_{nu}_lamb_{lambd}"
    #config_filename = f"operator_{st_operator_to_minimize}_nu_{str(nu)}_energy_{str(energy)}"
    config_filename = f"test_nu_{nu}"

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
        #st_operator_to_minimize=st_operator_to_minimize,
        #st_operators_evs_to_set={"energy": energy},
        optimization_method="newton",
        maxiters_cvxpy=1,
        )

# execute
run_all_configs(
    config_dir=config_dir,
    parallel=True,
    max_workers=6,
    check_if_exists_already=True
    )
