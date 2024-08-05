import yaml

def generate_configs_one_matrix(
      max_degree_L,
      config_filename,
      g2,
      g4,
      g6,
      init_scale=1e2,
      maxiters=100,
      maxiters_cvxpy=10_000,
      tol=1e-4,
      reg=1e6,
      eps=1e-4,
      radius=1e5,
      odd_degree_vanish=True,
      simplify_quadratic=True,
      ):

    config_data = {
        "model": {
            "model name": "OneMatrix",
            "bootstrap class": "BootstrapSystem",
            "max_degree_L": max_degree_L,
            "odd_degree_vanish": odd_degree_vanish,
            "simplify_quadratic": simplify_quadratic,
            "couplings": {"g2": g2, "g4": g4, "g6": g6},
            "symmetry_generators": None,
        },
        "optimizer": {
           "init_scale": init_scale,
           "maxiters": maxiters,
           "maxiters_cvxpy": maxiters_cvxpy,
           "tol": tol,
           "reg": reg,
           "eps": eps,
           "radius": radius,
           }
        }

    with open(f"configs/{config_filename}.yaml", 'w') as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    return