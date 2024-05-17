import numpy as np
import sympy as sp
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.quantum_info import SparsePauliOp
import symengine.lib.symengine_wrapper as symengine_wrapper
from typing import Union


class SpecialUnitaryGroup:
    """
    https://en.wikipedia.org/wiki/Structure_constants
    https://arxiv.org/pdf/2108.07219.pdf
    """

    def __init__(self, N):
        self.N = N
        self.structure_constants = self.generate_structure_constants()
        self.generators = self.generate_generators()
        self.check()

    def alpha(self, n, m):
        """Symmetric generator indices"""
        assert (1 <= m) and (m < n) and (n <= self.N)
        return n**2 + 2 * (m - n) - 1

    def beta(self, n, m):
        """Anti-symmetric generator indices"""
        assert (1 <= m) and (m < n) and (n <= self.N)
        return n**2 + 2 * (m - n)

    def gamma(self, n):
        """Diagonal generator indices"""
        assert (1 <= n) and (n <= self.N)
        return n**2 - 1

    def generate_generators(self):
        """
        Generate the generators.
        Note the mixed conventions: the literature uses one-based indexing
        and Python uses zero-based.
        """
        generators = {}
        for n in range(1, self.N + 1):
            for m in range(1, n):
                matrix = np.zeros((self.N, self.N))
                matrix[m - 1, n - 1] = 1
                matrix[n - 1, m - 1] = 1
                generators[self.alpha(n, m)] = matrix / 2

                matrix = np.zeros((self.N, self.N))
                matrix[m - 1, n - 1] = 1
                matrix[n - 1, m - 1] = -1
                generators[self.beta(n, m)] = -1j * matrix / 2

            if n > 1:
                matrix = np.zeros((self.N, self.N))
                matrix[n - 1, n - 1] = 1 - n
                for l in range(1, n):
                    matrix[l - 1, l - 1] = 1
                generators[self.gamma(n)] = matrix / np.sqrt(2 * n * (n - 1))

        return generators

    def generate_structure_constants(self):
        """
        Generate the structure constants.
        I found it easier to use unrestricted for loops and exceptions,
        rather than restrict the for loops.
        """
        structure_constants = {}

        for n in range(1, self.N + 1):
            for m in range(1, self.N + 1):
                for k in range(1, self.N + 1):
                    try:
                        structure_constants[
                            (self.alpha(n, m), self.alpha(k, n), self.beta(k, m))
                        ] = (1 / 2)
                    except:
                        pass

                    try:
                        structure_constants[
                            (self.alpha(n, m), self.alpha(n, k), self.beta(k, m))
                        ] = (1 / 2)
                    except:
                        pass

                    try:
                        structure_constants[
                            (self.alpha(n, m), self.alpha(k, m), self.beta(k, n))
                        ] = (1 / 2)
                    except:
                        pass

                    try:
                        structure_constants[
                            (self.beta(n, m), self.beta(k, m), self.beta(k, n))
                        ] = (1 / 2)
                    except:
                        pass

                    try:
                        if m > 1: # avoid any zero entries
                            structure_constants[
                                (self.alpha(n, m), self.beta(n, m), self.gamma(m))
                            ] = -np.sqrt((m - 1) / (2 * m))
                    except:
                        pass

                    try:
                        structure_constants[
                            (self.alpha(n, m), self.beta(n, m), self.gamma(n))
                        ] = np.sqrt(n / (2 * (n - 1)))
                    except:
                        pass

                    if m < k and k < n:
                        try:
                            structure_constants[
                                (self.alpha(n, m), self.beta(n, m), self.gamma(k))
                            ] = np.sqrt(1 / (2 * k * (k - 1)))
                        except:
                            pass

        # symmetrize
        for key, value in list(
            structure_constants.items()
        ):  # convert to list to avoid size-changing during iteration
            i, j, k = key
            structure_constants[(i, k, j)] = -value
            structure_constants[(j, i, k)] = -value
            structure_constants[(j, k, i)] = value
            structure_constants[(k, i, j)] = value
            structure_constants[(k, j, i)] = -value

        return structure_constants

    def check(self):
        """
        Check that the structure constants and generators obey the correct
        relation.
        """
        for i in range(1, self.N**2 - 1):
            for j in range(i + 1, self.N**2):
                term1 = np.dot(self.generators[i], self.generators[j]) - np.dot(
                    self.generators[j], self.generators[i]
                )

                term2 = sum(
                    1j * self.structure_constants.get((i, j, k), 0) * self.generators[k]
                    for k in range(1, self.N**2)
                )

                if not np.allclose(term1, term2):
                    print(i, j)
                    print(term1)
                    print(term2)
                    raise AssertionError


def drop_small_coeffs(operator: SparsePauliOp, tol=1e-10) -> SparsePauliOp:
    new_operator = 0 * operator
    for term in operator:

        coeffs = term.coeffs
        if len(coeffs) != 1:
            raise ValueError()

        term_is_zero = False
        if type(term.coeffs[0]) == ParameterExpression:
            if type(term.coeffs[0].sympify()) == symengine_wrapper.Mul:
                poly = sp.Poly(term.coeffs[0].sympify())
                to_remove = [abs(i) for i in poly.coeffs() if abs(i) < tol]
                for i in to_remove:
                    poly = poly.subs(i, 0)
                if poly == 0:
                    term_is_zero = True

        if not term_is_zero:
            new_operator += term

    return new_operator.simplify()


class BMNModel():
    def __init__(self, gauge_group_degree: int, bits_per_oscillator: int):
        self.num_matrices = 3 # hard-code this for now
        self.gauge_group_degree = gauge_group_degree
        self.num_generators = gauge_group_degree**2 - 1
        self.bits_per_oscillator = bits_per_oscillator
        self.num_states_per_oscillator = 2**bits_per_oscillator
        self.num_qubits = self.num_matrices * self.num_generators * bits_per_oscillator
        print(f'number_qubits {self.num_qubits}, Hilbert space dimension {2**self.num_qubits}')

        # Eq. 4.5 of https://arxiv.org/pdf/2011.06573
        self.single_qubit_state_map = {
            '00' : SparsePauliOp(data=["I", "Z"], coeffs=np.array([0.5, -0.5])),
            '01' : SparsePauliOp(data=["X", "Y"], coeffs=np.array([0.5, 0.5 * 1j])),
            '10' : SparsePauliOp(data=["X", "Y"], coeffs=np.array([0.5, -0.5 * 1j])),
            '11' : SparsePauliOp(data=["I", "Z"], coeffs=np.array([0.5, 0.5])),
            }

        self.build_index_map()

    def build_index_map(self):
        index_map = {}
        counter = 0
        for i in range(self.num_matrices):
            for a in range(self.num_generators):
                index_map[(i,a)] = counter
                counter += self.bits_per_oscillator
        #index_map_inverse = dict(map(reversed, index_map.items()))
        self.index_map = index_map

    def creation_operator(self, matrix_idx: int, generator_idx: int) -> SparsePauliOp:
        """
        Build the annihilation operators for the oscillator.

        Parameters
        ----------
        matrix_idx : int
            The SO(3) index enumerating the matrices (1, 2, 3)
        generator_idx : int
            The generator index (1, 2, ..., N^2)

        Returns
        -------
        SparsePauliOp
            The creation operator.
        """
        qubit_index_0 = self.index_map[(matrix_idx, generator_idx)] # first non-trivial qubit index
        operator = SparsePauliOp(data="I" * self.num_qubits, coeffs=np.asarray([0])) # initialize operator

        # loop over the truncated set of internal states for each oscillator
        for j in range(self.num_states_per_oscillator-1):

            # Eq. 4.3 of https://arxiv.org/pdf/2011.06573
            # a^\dagger = \sum_{j=0}^{\Lambda - 2} \sqrt{j+1} |j+1><j|
            ket_bitstring = f'{j+1:0{self.bits_per_oscillator}b}'
            bra_bitstring = f'{j:0{self.bits_per_oscillator}b}'

            # build the operator |j+1><j| bitwise
            new_term = SparsePauliOp(data="I" * self.num_qubits, coeffs=np.asarray([1])) # initialize operator
            ##print(f"j={j} |{ket_bitstring}><{bra_bitstring}|")
            for i in range(self.bits_per_oscillator):
                qubit_index = qubit_index_0 + i

                ##print(f"|{ket_bitstring[i]}><{bra_bitstring[i]}|")

                # extract the i-th bit term in |j+1><j|
                single_qubit_matrix_element = self.single_qubit_state_map[ket_bitstring[i] + bra_bitstring[i]]
                first_pauli = single_qubit_matrix_element.paulis[0].__str__()
                second_pauli = single_qubit_matrix_element.paulis[1].__str__()

                # expand the 1-bit expression to an n-bit expression
                first_pauli = ("I" * qubit_index) + first_pauli + (self.num_qubits - qubit_index - 1) * "I"
                second_pauli = ("I" * qubit_index) + second_pauli + (self.num_qubits - qubit_index - 1) * "I"
                ##print(f'bit = {i}', SparsePauliOp(data=[first_pauli, second_pauli], coeffs=single_qubit_matrix_element.coeffs))

                # compose with previously constructed terms
                new_term = new_term @ SparsePauliOp(data=[first_pauli, second_pauli], coeffs=single_qubit_matrix_element.coeffs)

            ##print(f"new term = ", new_term)
            # add to overall expression
            operator += np.sqrt(j + 1) * new_term

        return operator.simplify()


    def annihilation_operator(self, matrix_idx: int, generator_idx: int) -> SparsePauliOp:
        """
        Build the annihilation operators for the oscillator.

        Parameters
        ----------
        matrix_idx : int
            The SO(3) index enumerating the matrices (1, 2, 3)
        generator_idx : int
            The generator index (1, 2, ..., N^2)

        Returns
        -------
        SparsePauliOp
            The creation operator.
        """
        return self.creation_operator(matrix_idx, generator_idx).adjoint()


    def position_operator(self, matrix_idx: int, generator_idx: int) -> SparsePauliOp:
        """
        The position operator.

        Parameters
        ----------
        matrix_idx : int
            The SO(3) index enumerating the matrices (1, 2, 3)
        generator_idx : int
            The generator index (1, 2, ..., N^2)

        Returns
        -------
        SparsePauliOp
            The position operator.
        """
        creation = self.creation_operator(matrix_idx, generator_idx)
        annihilation = self.annihilation_operator(matrix_idx, generator_idx)
        coeff = complex(np.sqrt(1/2))
        operator = coeff * (creation + annihilation)
        return operator

    def momentum_operator(self, matrix_idx: int, generator_idx: int) -> SparsePauliOp:
        """
        The momentum operator.

        Parameters
        ----------
        matrix_idx : int
            The SO(3) index enumerating the matrices (1, 2, 3)
        generator_idx : int
            The generator index (1, 2, ..., N^2)

        Returns
        -------
        SparsePauliOp
            The momentum operator.
        """
        creation = self.creation_operator(matrix_idx, generator_idx)
        annihilation = self.annihilation_operator(matrix_idx, generator_idx)
        coeff = complex(-1j * np.sqrt(1/2))
        operator = coeff * (creation - annihilation)
        return operator

    def hamiltonian_bosonic_free(self, nu: Union[Parameter, float]) -> SparsePauliOp:
        """
        The free, non-interacting part of the Hamiltonian for the mini-BMN model,
        (1/2) Pi^i Pi^i + (nu^2/2) X^i X^i

        Parameters
        ----------
        nu : Union[Parameter, float]
            The mass deformation parameter.

        Returns
        -------
        SparsePauliOp
            The Hamiltonian term.
        """
        operator = SparsePauliOp(data="I" * self.num_qubits, coeffs=np.asarray([0])) # initialize operator
        for i in range(self.num_matrices):
            for a in range(self.num_generators):
                operator += (1/2) * (
                    self.momentum_operator(i, a) @ self.momentum_operator(i, a)
                    + nu**2 * self.position_operator(i, a) @ self.position_operator(i, a)
                    )
        return operator


    def hamiltonian_cubic_interaction(self, nu: Union[Parameter, float]) -> SparsePauliOp:
        """
        The cubic term in the Hamiltonian for the mini-BMN model,
        i nu epsilon^{ijk} X^i X^j X^k

        Parameters
        ----------
        nu : Union[Parameter, float]
            The mass deformation parameter.

        Returns
        -------
        SparsePauliOp
            The Hamiltonian term.
        """
        su_group = SpecialUnitaryGroup(self.gauge_group_degree)
        operator = SparsePauliOp(data="I" * self.num_qubits, coeffs=np.asarray([0])) # initialize operator
        for a in range(self.num_generators):
            for b in range(self.num_generators):
                for c in range(self.num_generators):
                    coeff = -6 * su_group.structure_constants.get((a+1,b+1,c+1), 0)
                    if coeff != 0:
                        operator += coeff*(self.position_operator(0, a)
                                @ self.position_operator(1, b)
                                @ self.position_operator(2, c)
                                )
        return operator * nu


    def hamiltonian_quartic_interaction(self) -> SparsePauliOp:
        """
        The quartic term in the Hamiltonian for the mini-BMN model,
        -(1/4) [X^i, X^j]^2

        Returns
        -------
        SparsePauliOp
            The Hamiltonian term.
        """
        operator = SparsePauliOp(data="I" * self.num_qubits, coeffs=np.asarray([0])) # initialize operator
        su_group = SpecialUnitaryGroup(self.gauge_group_degree)

        for i in range(self.num_matrices):
            for j in range(self.num_matrices):

                for a in range(self.num_generators):
                    for b in range(self.num_generators):
                        for c in range(self.num_generators):
                            for d in range(self.num_generators):

                                    coeff = (1/4) * sum(
                                        su_group.structure_constants.get((a+1,b+1,e+1), 0)
                                        * su_group.structure_constants.get((c+1,d+1,e+1), 0)
                                        for e in range(self.num_generators)
                                    )

                                    if coeff != 0:
                                        operator += coeff * (
                                            self.position_operator(i, a)
                                            @ self.position_operator(j, b)
                                            @ self.position_operator(i, c)
                                            @ self.position_operator(j, d)
                                            )

        return operator


    def hamiltonian(self, nu: Union[Parameter, float], free_only:bool=False) -> SparsePauliOp:
        """
        The bosonic (un-traced) Hamiltonian for the mini-BMN model.

        Parameters
        ----------
        nu : Union[Parameter, float]
            The mass deformation parameter.
        free_only : bool, optional
            Drop interaction terms so that the model is free, by default False

        Returns
        -------
        SparsePauliOp
            The Hamiltonian.
        """
        H = (self.hamiltonian_bosonic_free(nu)
            + int(not free_only) * self.hamiltonian_cubic_interaction(nu)
            + int(not free_only) * self.hamiltonian_quartic_interaction())
        H = H.simplify()
        H = drop_small_coeffs(H)
        return H
