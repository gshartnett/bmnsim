import fire
import numpy as np
from scipy.integrate import trapezoid as trapz
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult


class BornOppenheimer:
    """
    The Born-Oppenheimer model for the 2 matrix model,
    see https://doi.org/10.1103/PhysRevLett.125.041601.
    """
    def __init__(self, m, g):
        self.m = m
        self.g = g

    def normalization_constraint(self, rho: np.ndarray, x_grid: np.ndarray) -> float:
        """
        The normalization constraint equation, integral(rho) - 1.

        Parameters
        ----------
        rho : np.ndarray
            The collective coordinates.
        x_grid : np.ndarray
            The grid of x-values to consider.

        Returns
        -------
        float
            The RHS of the constraint equation (should be zero).
        """
        integral_rho = trapz(rho, x_grid)
        return integral_rho - 1.0

    def omega_matrix(self, x_grid: np.ndarray) -> np.ndarray:
        """
        The omega(x, y) term as a vectorized function.

        Parameters
        ----------
        x_grid : np.ndaray
            The grid of x-values to consider.

        Returns
        -------
        np.ndarray
            The omega(x, y) values as a 2D numpy array.
        """
        x_i, x_j = np.meshgrid(x_grid, x_grid)
        return np.sqrt(self.m**2 + self.g**2 * (x_i - x_j) ** 2)

    def local_energy_density(self, rho: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        """
        The local energy term.

        Parameters
        ----------
        rho : np.ndaray
            The collective coordinates.
        x_grid : np.ndaray
            The grid of x-values to consider.

        Returns
        -------
        np.ndarray
            The local energy term (represented as a 1D array).
        """
        return (np.pi**2 / 3) * rho**2 + self.m**2 * x_grid**2 * rho

    def non_local_energy_term(self, rho: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        """
        Non-local energy term using NumPy vectorization.

        Parameters
        ----------
        rho : np.ndaray
            The collective coordinates.
        x_grid : np.ndaray
            The grid of x-values to consider.

        Returns
        -------
        np.ndarray
            The non-local energy term (represented as a 1D array).
        """
        # compute the omega matrix for all pairs (x_i, x_j)
        omega_mat = self.omega_matrix(x_grid)

        # compute the outer product of rho with itself
        rho_outer = np.outer(rho, rho)

        # element-wise multiply the rho_outer with omega_mat
        delta_x = x_grid[1] - x_grid[0]  # assumes a uniform grid
        non_local_energy = np.sum(rho_outer * omega_mat) * delta_x**2

        return non_local_energy

    def E_BO_discretized(self, rho: np.ndarray, x_grid: np.ndarray) -> float:
        """
        The discretized Born-Oppenheimer energy functional.

        Parameters
        ----------
        rho : np.ndaray
            The collective coordinates.
        x_grid : np.ndaray
            The grid of x-values to consider.

        Returns
        -------
        float
            The energy.
        """
        # local energy: sum over the grid points using trapz
        local_energy = trapz(
            [self.local_energy_density(rho_i, x_i) for rho_i, x_i in zip(rho, x_grid)],
            x_grid,
        )

        # non-local energy: double sum over grid points
        non_local_energy = self.non_local_energy_term(rho, x_grid)

        return local_energy + non_local_energy

    def solve(self, x_grid: np.ndarray) -> OptimizeResult:
        """
        Find the minimium allowable energy of the Born-Oppenheimer functional.

        Parameters
        ----------
        x_grid : np.ndarray
            The grid of x-values to consider.

        Returns
        -------
        OptimizeResult
            The optimization result.
        """

        # bounds for rho (rho >= 0)
        bounds = [(0, None)] * len(x_grid)  # rho(x) >= 0 for all x

        # initial guess for rho (e.g., uniform distribution)
        x_min, x_max = x_grid[0], x_grid[-1]
        initial_rho = np.ones_like(x_grid) * 1 / (x_max - x_min)

        # Minimize the energy functional E_BO with the normalization constraint
        result = minimize(
            self.E_BO_discretized,
            initial_rho,
            args=(x_grid),
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "eq",
                "fun": lambda rho: self.normalization_constraint(rho, x_grid),
            },
        )

        return result


def main(m: float=1, g: float=1, npoints: int=250):
    # set-up the BO model
    born_oppenheimer = BornOppenheimer(m=m, g=g)

    # define the x grid
    x_min, x_max = -3, 3
    x_grid = np.linspace(x_min, x_max, npoints)

    # solve and print the result
    result = born_oppenheimer.solve(x_grid=x_grid)
    optimal_energy = result.fun

    print(f"Minimum BO energy for m={m}, g={g}: E={optimal_energy:.4f}")

if __name__ == "__main__":
    fire.Fire(main)