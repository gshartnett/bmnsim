import numpy as np


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
