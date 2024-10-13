import json
import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RBFInterpolator
from scipy import optimize
from bmn.brezin import compute_Brezin_energy
from scipy import interpolate, integrate
import inspect
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    #color=['#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2'][::-1]
)
matplotlib.rcParams.update(
    {"axes.grid": False, "grid.alpha": 0.75, "grid.linewidth": 0.5}
)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

np.set_printoptions(linewidth=160)


def load_data(
        datadir,
        names_in_filename,
        tol=1e-6,
        delete_param=True
        ):

    # grab the data files
    names_in_filename.append('.json')
    files = []
    for f in os.listdir(datadir):
        if all(name in f for name in names_in_filename):
            files.append(f)
    print(f"number of files found: {len(files)}")

    if len(files) == 0:
        return

    # build dataframe
    data = []
    for file in files:
        with open(f"{datadir}/{file}") as f:
            result = json.load(f)
        if delete_param:
            del result["param"] # remove param vector
        #result["energy"] = float(file.split('_')[1][:-5]) # add g4 coupling
        result["filename"] = file

        if (
            (np.abs(result["min_bootstrap_eigenvalue"]) < tol) &
            (np.abs(result["violation_of_linear_constraints"]) < tol) &
            (np.abs(result["quad_constraint_violation_norm"]) < tol)
        ):
            data.append(result)

    df = pd.DataFrame(data)
    if len(df) == 0:
        return df.copy()

    df.sort_values("energy", inplace=True)
    max_violation_linear = df["violation_of_linear_constraints"].max()
    max_violation_quadratic = df["max_quad_constraint_violation"].max()
    max_violation_PSD = df["min_bootstrap_eigenvalue"].abs().max()

    print(f"number of loaded data points: {len(data)}")
    print(f"max violation of linear constraints:{max_violation_linear:.4e}")
    print(f"max violation of PSD constraints:{max_violation_PSD:.4e}")
    print(f"max violation of quadratic constraints:{max_violation_quadratic:.4e}\n")

    return df.copy()


def add_extension(filename, extension):
    if extension not in filename:
        filename = f"{filename}.{extension}"
    return filename


def make_figure_regularization_scan_one_matrix(extension='pdf'):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    # load data
    data = []
    for L in [3, 4]:
        path =f"data/OneMatrix_L_{L}_reg_scan"
        files = [f for f in os.listdir(path) if ".json" in f]
        print(f"L={L}, number of data points found: {len(files)}")

        for file in files:
            with open(f"{path}/{file}") as f:
                result = json.load(f)
            del result["param"] # remove param vector
            if result["max_quad_constraint_violation"] < 1e-2:
                result["L"] = int(L)
                result["reg"] = float(file.split('_')[3][:-5])
                data.append(result)
    df = pd.DataFrame(data)
    df.sort_values("energy", inplace=True)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for idx, L in enumerate([3, 4]):

        ax[0].plot(df[df["L"] == L]["reg"], np.abs(df[df["L"] == L]["energy"] - compute_Brezin_energy(g_value=1/4)), 'o', label=f"L={L}")
        ax[0].set_xlabel("regularization")
        ax[0].set_ylabel("abs(energy - Brezin energy)")
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].legend(fontsize=10)

        ax[1].plot(df[df["L"] == L]["reg"], np.abs(df[df["L"] == L]["min_bootstrap_eigenvalue"]), 'o', label=f"min bootstrap eigenvalue, L={L}")
        ax[1].plot(df[df["L"] == L]["reg"], np.abs(df[df["L"] == L]["max_quad_constraint_violation"]), 'o', label=f"max_quad_constraint_violation L={L}")
        ax[1].plot(df[df["L"] == L]["reg"], np.abs(df[df["L"] == L]["violation_of_linear_constraints"]), 'o', label=f"violation_of_linear_constraints L={L}")
        ax[1].set_xlabel("regularization")
        ax[1].set_ylabel("constraint violation")
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].legend(fontsize=10)
    plt.suptitle(r"OneMatrix model $g2=1$, $g4=1$, $g6=0$, minimizing energy")

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split('_')[2:])
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_regularization_scan_two_matrix_massless(extension='pdf'):

    this_function_name = inspect.currentframe().f_code.co_name
    print("=========================================")
    print(f"running function: {this_function_name}\n")

    # load the data
    L = 3
    path = "data/TwoMatrix_L_3_g2_0_g4_1_reg_scan"
    data = []
    files = [f for f in os.listdir(path) if ".json" in f]
    print(f"Number of data points found: {len(files)}")
    for file in files:
        with open(f"{path}/{file}") as f:
            result = json.load(f)
        del result["param"] # remove param vector
        if result["max_quad_constraint_violation"] < 1e-2:
            result["reg"] = float(file.split('_')[3][:-5])
            data.append(result)
    df = pd.DataFrame(data)
    df.sort_values("energy", inplace=True)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(df["reg"], df["energy"], 'o', label=f"L={L}")
    ax[0].set_xlabel("regularization")
    ax[0].set_ylabel("energy")
    ax[0].set_xscale('log')
    ax[0].legend(fontsize=10)

    energy_asymptotic = df["energy"].max()
    ax[0].axhline(energy_asymptotic, color='k', linestyle='--', label="asymptotic energy")
    ax[0].text(1e-7, energy_asymptotic, f"{energy_asymptotic:.4f}", fontsize=14, verticalalignment='bottom')

    ax[1].plot(df["reg"], np.abs(df["min_bootstrap_eigenvalue"]), 'o', label=f"min bootstrap eigenvalue, L={L}")
    ax[1].plot(df["reg"], np.abs(df["max_quad_constraint_violation"]), 'o', label=f"max_quad_constraint_violation L={L}")
    ax[1].plot(df["reg"], np.abs(df["violation_of_linear_constraints"]), 'o', label=f"violation_of_linear_constraints L={L}")
    ax[1].set_xlabel("regularization")
    ax[1].set_ylabel("constraint violation")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].legend(fontsize=10)
    plt.suptitle(r"TwoMatrix model $g2=1$, $g4=1$, minimizing energy")

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split('_')[2:])
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_regularization_scan_two_matrix_massive(extension='pdf'):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    # load the data
    L = 3
    path = "data/TwoMatrix_L_3_g2_1_g4_1_reg_scan"
    data = []
    files = [f for f in os.listdir(path) if ".json" in f]
    print(f"Number of data points found: {len(files)}")
    for file in files:
        with open(f"{path}/{file}") as f:
            result = json.load(f)
        del result["param"] # remove param vector
        if result["max_quad_constraint_violation"] < 1e-2:
            result["reg"] = float(file.split('_')[3][:-5])
            data.append(result)
    df = pd.DataFrame(data)
    df.sort_values("energy", inplace=True)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(df["reg"], df["energy"], 'o', label=f"L={L}")
    ax[0].set_xlabel("regularization")
    ax[0].set_ylabel("energy")
    ax[0].set_xscale('log')
    ax[0].legend(fontsize=10)

    energy_asymptotic = df["energy"].max()
    ax[0].axhline(energy_asymptotic, color='k', linestyle='--', label="asymptotic energy")
    ax[0].text(1e-7, energy_asymptotic, f"{energy_asymptotic:.4f}", fontsize=14, verticalalignment='bottom')

    ax[1].plot(df["reg"], np.abs(df["min_bootstrap_eigenvalue"]), 'o', label=f"min bootstrap eigenvalue, L={L}")
    ax[1].plot(df["reg"], np.abs(df["max_quad_constraint_violation"]), 'o', label=f"max_quad_constraint_violation L={L}")
    ax[1].plot(df["reg"], np.abs(df["violation_of_linear_constraints"]), 'o', label=f"violation_of_linear_constraints L={L}")
    ax[1].set_xlabel("regularization")
    ax[1].set_ylabel("constraint violation")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].legend(fontsize=10)
    plt.suptitle(r"TwoMatrix model $g2=1$, $g4=1$, minimizing energy")

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split('_')[2:])
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_x2_bound_two_matrix(extension='pdf', reg=1e-4):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    # load the data
    L = 3
    path = f"data/TwoMatrix_L_{L}_symmetric_energy_fixed_g2_1.0_reg_{reg:.2e}"
    print(f"reg = {reg:.2e}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax_inset = inset_axes(ax, width="35%", height="35%", loc='center right')

    # lower bound
    df = load_data(path, names_in_filename=["op_to_min_x_2"], tol=1e-5)
    operator_to_minimize = "x_2"
    ax.scatter(df["energy"], df["x_2"], edgecolor='k', zorder=10, s=30, label=f'lower bound, L={L}', color=colors[0])
    ax.plot(df["energy"], df["x_2"], color=colors[0])
    ax_inset.scatter(df["energy"], df["x_2"], edgecolor='k', zorder=10, s=30, color=colors[0])
    ax_inset.plot(df["energy"], df["x_2"], color=colors[0])

    # upper bound
    df = load_data(path, names_in_filename=["op_to_min_neg_x_2"], tol=1e-5)
    operator_to_minimize = "neg_x_2"
    ax.plot(df["energy"], df["x_2"], color=colors[0])
    ax.scatter(df["energy"], df["x_2"], edgecolor='k', zorder=10, s=30, label=f'upper bound, L={L}', color=colors[0])
    ax_inset.scatter(df["energy"], df["x_2"], edgecolor='k', zorder=10, s=30, color=colors[0])
    ax_inset.plot(df["energy"], df["x_2"], color=colors[0])

    ax.set_xlabel(r"$\lambda^{-1/3} N^{-2} E$")
    ax.set_ylabel(r"$\lambda^{-2/3} N^{-2}$ Tr$(X^2)$")
    ax.set_title(r"TwoMatrix $g_2=1$, $g_4=1$" + f" reg={reg:.2e}")
    min_energy = df["energy"].min()
    ax.text(0.05, 0.95, f"Min energy={min_energy:.4f}", transform=ax.transAxes, fontsize=16, verticalalignment='top')
    ax.axvline(min_energy, color='k', linestyle='--')

    ax_inset.set_xlim(0.9, 1.5)  # Adjust as needed
    ax_inset.set_ylim(0.8, 1.2)  # Adjust as needed

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split('_')[2:]) + f" reg={reg:.2e}"
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


if __name__ == "__main__":

    extension = 'png'

    # one matrix
    make_figure_regularization_scan_one_matrix(extension)

    # two matrix
    make_figure_regularization_scan_two_matrix_massless(extension)
    make_figure_regularization_scan_two_matrix_massive(extension)

    for reg in [1e-4, 1e-1, 1e1]:
        make_figure_x2_bound_two_matrix(extension, reg)