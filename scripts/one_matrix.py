import fire
import matplotlib

# plot settings
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.bootstrap import BootstrapSystem
from bmn.brezin import (
    compute_Brezin_energy,
    compute_Brezin_energy_Han_conventions,
)
from bmn.debug_utils import disable_debug
from bmn.newton_solver import minimize as minimize_newton
from bmn.solver import (
    minimal_eigval,
    minimize,
)

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


def run_one_matrix(g, L, init=None, verbose=False):

    matrix_system = MatrixSystem(
        # operator_basis=['X', 'P'],
        operator_basis=["X", "Pi"],
        commutation_rules_concise={
            # ('P', 'X'): -1j,
            ("Pi", "X"): 1,  # use Pi' = i P to ensure reality
        },
        # hermitian_dict={'P': True, 'X': True},
        hermitian_dict={"Pi": False, "X": True},
    )

    # scale variables as P = sqrt(N) P', X = sqrt(N) X'
    hamiltonian = SingleTraceOperator(
        # data={("P", "P"): 1, ("X", "X"): 1, ("X", "X", "X", "X"): 7}
        data={("Pi", "Pi"): -1, ("X", "X"): 1, ("X", "X", "X", "X"): g}
    )

    # <tr G O > = 0 might need to be applied only for O with deg <= L-2
    # gauge = MatrixOperator(data={('X', 'P'): 1j, ('P', 'X'): -1j, ():1})
    gauge = MatrixOperator(data={("X", "Pi"): 1, ("Pi", "X"): -1, (): 1})

    bootstrap = BootstrapSystem(
        matrix_system=matrix_system,
        hamiltonian=hamiltonian,
        gauge=gauge,
        max_degree_L=L,
        odd_degree_vanish=True,
        simplify_quadratic=True,
    )

    bootstrap.get_null_space_matrix()

    disable_debug()

    param, success = minimize_newton(
        bootstrap=bootstrap,
        op=bootstrap.hamiltonian,
        init=init,
        init_scale=1e2,
        verbose=verbose,
        maxiters=50,
    )

    """
    param, success = minimize(
        bootstrap=bootstrap,
        op=bootstrap.hamiltonian,
        init=init,
        init_scale=1e1,
        verbose=verbose,
        maxiters=100,
        reg=5e-3,
        eps=9e-4,
    )
    """

    """
    for op in bootstrap.operator_list:
        vec = bootstrap.single_trace_to_coefficient_vector(
            st_operator=SingleTraceOperator(data={op: 1}), return_null_basis=True
        )
        op_expectation_value = vec @ param
        print(f"op = {op}, EV = {op_expectation_value}")
    """

    energy = bootstrap.get_operator_expectation_value(
        st_operator=hamiltonian, param=param
    )
    exact_energy = compute_Brezin_energy_Han_conventions(g)
    print(
        f"problem success: {success}, min energy found: {energy:.6f}, exact (L=inf) value = {exact_energy:.6f}"
    )
    print(f"energy error = {energy - exact_energy:.4e}")
    return success, energy, param


def run_scan(L):
    g_values = np.linspace(0.1, 10, 10)
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
                # label=f"Bootstrap L={L}, converged",
            )

    ax.legend(frameon=False)
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$E_0/N^2$")
    plt.show()


def run(L=3, g=1.0, scan=False, verbose=False):
    if not scan:
        success, energy, param = run_one_matrix(g=g, L=L, verbose=verbose)
        print(f"param = {param}")
        return
    else:
        return run_scan(L=L)


if __name__ == "__main__":

    fire.Fire(run)
