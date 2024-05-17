import numpy as np
import sympy as sp


number_of_matrices = 3
gauge_group_degree = 2 # the N in SU(N)
number_of_generators = gauge_group_degree**2 - 1 #understand the diff b/w U(N) and SU(N)

bits_per_oscillator = 2
number_states_per_oscillator = 2**bits_per_oscillator

number_qubits = (
    number_of_matrices
    * number_of_generators
    * bits_per_oscillator
)

print(f'number_qubits {number_qubits}, Hilbert space dimension {2**number_qubits}')
index_map = {}
counter = 0
for i in range(number_of_matrices):
    for a in range(number_of_generators):
        index_map[(i,a)] = counter
        counter += bits_per_oscillator
index_map_inverse = dict(map(reversed, index_map.items()))


# Eq. 4.5 of https://arxiv.org/pdf/2011.06573
single_qubit_state_map = {
    '00':0.5*(sp.symbols("I") + sp.symbols('Z')),
    '01':0.5*(sp.symbols("X") + 1j*sp.symbols('Y')),
    '10':0.5*(sp.symbols("X") - 1j*sp.symbols('Y')),
    '11':0.5*(sp.symbols("I") - sp.symbols('Z'))
    }


def annihilation_operator_old(matrix_idx: int, generator_idx: int) -> sp.core.add.Add:
    """
    Build the annihilation operator for the oscillator.

    Parameters
    ----------
    matrix_idx : int
        The SO(3) index enumerating the matrices (1, 2, 3)
    generator_idx : int
        The generator index (1, 2, ..., N^2)

    Returns
    -------
    sp.core.add.Add
        A sympy representation of the operator, written in terms of Pauli's, e.g.
        1.0*I0 + 0.5*X0 + 1.5*X1 - 0.5*I*Y0 - 0.5*I*Y1
    """
    qubit_indx0 = index_map[(matrix_idx, generator_idx)]
    operator = 0
    for j in range(number_states_per_oscillator-1):
        bra_bitstring = f'{j:0{bits_per_oscillator}b}'
        ket_bitstring = f'{j+1:0{bits_per_oscillator}b}'
        for i in range(bits_per_oscillator):
            new_term = single_qubit_state_map[bra_bitstring[i] + ket_bitstring[i]]
            new_term = new_term.subs('I', f'I{qubit_indx0+i}')
            new_term = new_term.subs('X', f'X{qubit_indx0+i}')
            new_term = new_term.subs('Y', f'Y{qubit_indx0+i}')
            new_term = new_term.subs('Z', f'Z{qubit_indx0+i}')
            operator += new_term
    return operator


def creation_operator_old(matrix_idx: int, generator_idx: int) -> sp.core.add.Add:
    """
    Build the creation operator for the oscillator.

    Parameters
    ----------
    matrix_idx : int
        The SO(3) index enumerating the matrices (1, 2, 3)
    generator_idx : int
        The generator index (1, 2, ..., N^2)

    Returns
    -------
    sp.core.add.Add
        A sympy representation of the operator, written in terms of Pauli's, e.g.
        1.0*I0 + 0.5*X0 + 1.5*X1 - 0.5*I*Y0 - 0.5*I*Y1
    """
    qubit_indx0 = index_map[(matrix_idx, generator_idx)]
    operator = 0
    for j in range(number_states_per_oscillator-1):
        bra_bitstring = f'{j+1:0{bits_per_oscillator}b}'
        ket_bitstring = f'{j:0{bits_per_oscillator}b}'
        for i in range(bits_per_oscillator):
            new_term = single_qubit_state_map[bra_bitstring[i] + ket_bitstring[i]]
            new_term = new_term.subs('I', f'I{qubit_indx0+i}')
            new_term = new_term.subs('X', f'X{qubit_indx0+i}')
            new_term = new_term.subs('Y', f'Y{qubit_indx0+i}')
            new_term = new_term.subs('Z', f'Z{qubit_indx0+i}')
            operator += new_term
    return operator


class SpecialUnitaryGroup():
    '''
    https://en.wikipedia.org/wiki/Structure_constants
    https://arxiv.org/pdf/2108.07219.pdf
    '''
    def __init__(self, N):
        self.N = N
        self.structure_constants = self.generate_structure_constants()
        self.generators = self.generate_generators()
        self.check()

    def alpha(self, n, m):
        assert (1 <= m) and (m < n) and (n <= self.N)
        return n**2 + 2*(m-n) -1

    def beta(self, n, m):
        assert (1 <= m) and (m < n) and (n <= self.N)
        return n**2 + 2*(m-n)

    def gamma(self, n):
        assert (1 <= n) and (n <= self.N)
        return n**2 - 1

    def generate_generators(self):
        generators = {}
        for n in range(1, self.N+1):
            for m in range(1, self.N+1):
                if m < n:
                    matrix = np.zeros((self.N, self.N))
                    matrix[m-1,n-1] = 1
                    matrix[n-1,m-1] = 1
                    generators[self.alpha(n,m)] = matrix/2

                    matrix = np.zeros((self.N, self.N))
                    matrix[m-1,n-1] = 1
                    matrix[n-1,m-1] = -1
                    generators[self.beta(n,m)] = -1j*matrix/2

            if n > 1:
                matrix = np.zeros((self.N, self.N))
                matrix[n-1,n-1] = (1 - n)
                for l in range(1, n):
                    matrix[l-1,l-1] = 1
                generators[self.gamma(n)] = matrix / np.sqrt(2*n*(n-1))

        return generators

    def generate_structure_constants(self):

        structure_constants = {}
        '''
        for n in range(1, self.N+1):
            for m in range(1, self.N+1):
                if m < n:
                    print('alpha = %i, beta = %i' %(self.alpha(n,m), self.beta(n,m)))
            print('gamma = %i' % self.gamma(n))
        '''


        for n in range(1, self.N+1):
            for m in range(1, self.N+1):
                for k in range(1, self.N+1):

                    try:
                        structure_constants[(
                            self.alpha(n,m),
                            self.alpha(k,n),
                            self.beta(k,m)
                            )] = 1/2
                    except:
                        pass

                    try:
                        structure_constants[(
                            self.alpha(n,m),
                            self.alpha(n,k),
                            self.beta(k,m)
                            )] = 1/2
                    except:
                        pass

                    try:
                        structure_constants[(
                            self.alpha(n,m),
                            self.alpha(k,m),
                            self.beta(k,n)
                            )] = 1/2
                    except:
                        pass

                    try:
                        structure_constants[(
                            self.beta(n,m),
                            self.beta(k,m),
                            self.beta(k,n)
                            )] = 1/2
                    except:
                        pass

                    try:
                        structure_constants[(
                            self.alpha(n,m),
                            self.beta(n,m),
                            self.gamma(m)
                            )] = - np.sqrt( (m-1)/(2*m) )
                    except:
                        pass

                    try:
                        structure_constants[(
                            self.alpha(n,m),
                            self.beta(n,m),
                            self.gamma(n)
                            )] = np.sqrt(n / (2*(n-1)) )
                    except:
                        pass

                    if m < k and k < n:
                        try:
                            structure_constants[(
                                self.alpha(n,m),
                                self.beta(n,m),
                                self.gamma(k)
                                )] = np.sqrt(1 / (2*k*(k-1)) )
                        except:
                            pass

        for key, value in list(structure_constants.items()):
            if value == 0:
                structure_constants.pop(key)
            else:
                i, j, k = key
                structure_constants[(i,k,j)] = - value
                structure_constants[(j,i,k)] = - value
                structure_constants[(j,k,i)] = value
                structure_constants[(k,i,j)] = value
                structure_constants[(k,j,i)] = - value

        return structure_constants

    def check(self):
        for i in range(1, self.N**2-1):
            for j in range(i+1, self.N**2):

                term1 = (np.dot(
                    self.generators[i],
                    self.generators[j]
                    ) -
                np.dot(
                    self.generators[j],
                    self.generators[i]
                    ))

                term2 = sum(
                    1j*self.structure_constants.get((i,j,k), 0)
                    * self.generators[k]
                    for k in range(1, self.N**2))

                if not np.allclose(term1, term2):
                    print(i,j)
                    print(term1)
                    print(term2)
                    raise AssertionError