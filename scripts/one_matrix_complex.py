import numpy as np
import fire
from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.bootstrap_complex import BootstrapSystemComplex
from bmn.debug_utils import disable_debug
from bmn.solver import minimize, minimal_eigval
from bmn.brezin import compute_Brezin_energy, compute_Brezin_energy_Han_conventions

# plot settings
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 5.0
plt.rcParams["xtick.minor.size"] = 3.0
plt.rcParams["ytick.major.size"] = 5.0
plt.rcParams["ytick.minor.size"] = 3.0
plt.rcParams["lines.linewidth"] = 2
plt.rc("font", family="serif", size=16)
matplotlib.rc("text", usetex=True)
matplotlib.rc("legend", fontsize=16)
matplotlib.rcParams["axes.prop_cycle"] = cycler(
    color=["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42", "#FFB5B8"]
)
matplotlib.rcParams.update(
    {"axes.grid": True, "grid.alpha": 0.75, "grid.linewidth": 0.5}
)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def run_one_matrix(g, L, init=None):

    matrix_system = MatrixSystem(
        operator_basis=['X', 'P'],
        commutation_rules_concise={
            ('P', 'X'): -1j,
        },
        hermitian_dict={'P': True, 'X': True},
    )

    # scale variables as P = sqrt(N) P', X = sqrt(N) X'
    hamiltonian = SingleTraceOperator(
        data={("P", "P"): 1, ("X", "X"): 1, ("X", "X", "X", "X"): g}
    )

    # <tr G O > = 0 might need to be applied only for O with deg <= L-2
    gauge = MatrixOperator(data={('X', 'P'): 1j, ('P', 'X'): -1j, ():1})

    bootstrap = BootstrapSystemComplex(
        matrix_system=matrix_system,
        hamiltonian=hamiltonian,
        gauge=gauge,
        max_degree_L=L,
        odd_degree_vanish=True,
        simplify_quadratic=False,
    )

    bootstrap.build_null_space_matrix()

    #disable_debug()

    param, success = minimize(
        bootstrap=bootstrap,
        op=bootstrap.hamiltonian,
        init=init,
        init_scale=1e2,
        verbose=False,
        maxiters=25,
        reg=5e-4,
        eps=5e-4,
    )

    energy = bootstrap.get_operator_expectation_value(
        st_operator=hamiltonian,
        param=param
        )
    energy = np.real(energy)
    exact_energy = compute_Brezin_energy_Han_conventions(g)
    print(f"problem success: {success}, min energy found: {energy:.6f}, exact (L=inf) value = {exact_energy:.6f}")

    return success, energy, param


def run_scan(L):
    g_values = np.linspace(0.1, 10, 15)
    results = {
        "g": g_values,
        "success": [],
        "energy": [],
        "param": [],
    }

    param = None
    for g in g_values:
        success, energy, param = run_one_matrix(g=g, L=L, init=param)
        results["success"].append(success)
        results["energy"].append(energy)
        results["param"].append(param)
        print(f"Completed run for g={g}, success={success}, energy={energy}")

    # compute the exact values (note the difference in conventions)
    exact_values = [compute_Brezin_energy_Han_conventions(g) for g in g_values]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(results["g"], results["energy"], "-o", label=f"Bootstrap L={L}")
    ax.plot(results["g"], exact_values, "-o", label=f"Exact")

    for i in range(len(g_values)):
        if results["success"][i]:
            print(results["g"][i], results["energy"][i])
            ax.scatter(
                results["g"][i],
                results["energy"][i],
                color="k",
                zorder=10,
                #label=f"Bootstrap L={L}, converged",
            )

    ax.legend(frameon=False)
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$E_0/N^2$")
    plt.savefig(f"figures/one_matrix_complex_scan_L_{L}.pdf")
    plt.show()


def run(L=3, g=1.0, scan=False):
    if not scan:
        success, energy, param = run_one_matrix(g=g, L=L)
        print(f"param = {param}")
        return
    else:
        return run_scan(L=L)


if __name__ == "__main__":

    fire.Fire(run)
