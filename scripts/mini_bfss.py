import numpy as np
from bmn.config_utils import generate_configs_bfss, run_all_configs

# generate the config files
L = 3
impose_symmetries = True

if impose_symmetries:
    dir_name = f"MiniBFSS_L_{L}_symmetric"
else:
    dir_name = f"MiniBFSS_L_{L}"

num_seeds = 1
for st_operator_to_minimize in ["x_2", "neg_x_2", "x_4"]:
    for energy in np.linspace(0.5, 2.5, 30):
        for PRNG_seed in range(num_seeds):

            energy = float(np.round(energy, decimals=6))

            generate_configs_bfss(
                config_filename=f"energy_{str(energy)}_op_to_min_{st_operator_to_minimize}",
                config_dir=dir_name,
                st_operator_to_minimize=st_operator_to_minimize,
                st_operators_evs_to_set={"energy": energy},
                max_degree_L=L,
                load_from_previously_computed=True,
                checkpoint_path=f"MiniBFSS_L_{L}_symmetric",
                impose_symmetries=impose_symmetries,
                maxiters_cvxpy=10_000,
                init_scale=1e0,
                penalty_reg=1e6,
                #PRNG_seed=PRNG_seed,
                )

# execute
run_all_configs(config_dir=dir_name, parallel=True)

'''
# generate the config files
L = 3
dir_name = f"MiniBFSS_L_{L}_symmetric_min_energy"
checkpoint_path = f"MiniBFSS_L_{L}_symmetric"

#for PRNG_seed in range(6):
generate_configs_bfss(
    config_filename=f"test",
    config_dir=dir_name,
    st_operator_to_minimize="energy",
    checkpoint_path=checkpoint_path,
    max_degree_L=L,
    load_from_previously_computed=True,
    #PRNG_seed=PRNG_seed,
    maxiters_cvxpy=5_000,
    )

# execute
run_all_configs(config_dir=dir_name, parallel=False)
'''