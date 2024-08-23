import numpy as np
from bmn.config_utils import generate_configs_two_matrix, run_all_configs, run_bootstrap_from_config
import json

## energy held fixed
L = 3
g2 = 0
checkpoint_path = f"TwoMatrix_L_{L}_symmetric_energy_fixed_g2_{g2}"

config_dir = f"TwoMatrix_L_{L}_symmetric_energy_fixed_g2_{g2}"
#config_dir = f"TwoMatrix_L_{L}_symmetric_energy_fixed_g2_{g2}_pytorch"

for st_operator_to_minimize in ["x_2", "neg_x_2", "x_4", "neg_x_4"]:
    for energy in np.linspace(0.4, 3.0, 40):

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
            #optimization_method="pytorch",
            #lr=1e0,
            #init_scale=1e-2,
            )

# execute
run_all_configs(config_dir=config_dir, parallel=True, max_workers=3)


'''
# step down from large energy
L = 3
num_seeds = 3
g2 = 0
checkpoint_path = f"TwoMatrix_L_{L}_symmetric"
dir_name = f"TwoMatrix_L_{L}_symmetric_zero_mass_init_from_previous_soln"
energy_list = list(np.linspace(0.4, 1.5, 30))[::-1]
st_operator_to_minimize = "x_2"

for idx, energy in enumerate(energy_list):

    energy = float(np.round(energy, decimals=6))
    config_filename = f"energy_{str(energy)}_op_to_min_{st_operator_to_minimize}"

    if idx == 0:
        init = None
    else:
        previous_energy = float(np.round(energy_list[idx-1], decimals=6))
        previous_config_filename = f"energy_{str(previous_energy)}_op_to_min_{st_operator_to_minimize}"
        with open(f"data/{dir_name}/{previous_config_filename}.json") as f:
            result = json.load(f)
        init = result["param"]

    generate_configs_two_matrix(
        config_filename=config_filename,
        config_dir=dir_name,
        g2=g2,
        g4=1,
        st_operator_to_minimize=st_operator_to_minimize,
        st_operators_evs_to_set={"energy": energy},
        max_degree_L=L,
        load_from_previously_computed=True,
        checkpoint_path=checkpoint_path,
        impose_symmetries=True,
        init=init,
        )

    # execute
    run_bootstrap_from_config(config_filename=config_filename, config_dir=dir_name)
'''