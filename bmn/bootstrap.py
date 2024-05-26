from itertools import product
from typing import Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.linear_algebra import (
    create_sparse_matrix_from_dict,
    get_null_space,
)


class BootstrapSystem:
    """
    _summary_
    """

    def __init__(
        self,
        matrix_system: MatrixSystem,
        hamiltonian: SingleTraceOperator,
        gauge: MatrixOperator,
        half_max_degree: int,
        tol: float = 1e-10,
    ):
        self.matrix_system = matrix_system
        self.hamiltonian = hamiltonian
        self.gauge = gauge
        self.half_max_degree = half_max_degree
        self.operator_list = self.generate_operators(2 * half_max_degree)
        self.operator_dict = {op: idx for idx, op in enumerate(self.operator_list)}
        if 2 * self.half_max_degree < self.hamiltonian.max_degree:
            raise ValueError(
                "2 * half_max_degree must be >= max degree of Hamiltonian."
            )
        self.param_dim = len(self.operator_dict)
        self.tol = tol

    def generate_operators(self, max_degree: int) -> list[str]:
        """
        Generate the list of operators used in the bootstrap, i.e.
            I, X, P, XX, XP, PX, PP, ...,
        up to and including strings of max_degree degree.

        Parameters
        ----------
        max_degree : int
            Maximum degree of operators to consider.

        Returns
        -------
        list[str]
            A list of the operators.
        """
        operators = {
            deg: [x for x in product(self.matrix_system.operator_basis, repeat=deg)]
            for deg in range(0, max_degree + 1)
        }
        self.psd_matrix_dim = sum(
            len(value)
            for degree, value in operators.items()
            if degree <= self.half_max_degree
        )
        return [x for xs in operators.values() for x in xs]  # flatten

    def single_trace_to_coefficient_vector(
        self, st_operator: SingleTraceOperator, return_null_basis: bool=False
    ) -> np.ndarray:
        """
        TODO make sparse compatible
        Map a single trace operator to a vector of the coefficients.

        Parameters
        ----------
        st_operator : SingleTraceOperator
            The operator

        return_null_basis : bool, optional
            Controls whether the flag is returned in the original basis or the null basis.
            By default False.

        Returns
        -------
        np.ndarray
            The vector.
        """
        vec = [0] * self.param_dim
        for op, coeff in st_operator:
            idx = self.operator_dict[op]
            vec[idx] = coeff
        if not return_null_basis:
            return np.asarray(vec)
        return np.asarray(vec) @ self.null_space_matrix

    def generate_hamiltonian_constraints(self) -> list[SingleTraceOperator]:
        """
        Generate the Hamiltonian constraints <[H,O]>=0 for O single trace.

        Returns
        -------
        list[SingleTraceOperator]
            The list of constraint terms.
        """
        constraints = []
        for op in self.operator_list:
            constraints.append(
                self.matrix_system.single_trace_commutator(
                    st_operator1=self.hamiltonian,
                    st_operator2=SingleTraceOperator(data={op: 1}),
                )
            )
        return self.clean_constraints(constraints)

    def generate_gauge_constraints(self) -> list[SingleTraceOperator]:
        """
        Generate the Gauge constraints <tr(G O)>=0 for O a general matrix operator.
        Because G has max degree 2, this will create constraints involving terms
        with degree max_degree + 2. These will be discarded.

        Returns
        -------
        list[SingleTraceOperator]
            The list of constraint terms.
        """
        constraints = []
        for op in self.operator_list:
            constraints.append((self.gauge * MatrixOperator(data={op: 1})).trace())
        return self.clean_constraints(constraints)

    def generate_reality_constraints(self) -> list[SingleTraceOperator]:
        """
        Generate single trace constraints imposed by reality,
            <O^dagger> = <O>

        Returns
        -------
        list[SingleTraceOperator]
            The list of constraint terms.
        """
        constraints = []
        for op in self.operator_list:
            st_operator = SingleTraceOperator(data={op: 1})
            st_operator_dagger = self.matrix_system.hermitian_conjugate(
                operator=SingleTraceOperator(data={op: 1})
            )
            if len(st_operator - st_operator_dagger) > 0:
                constraints.append(st_operator - st_operator_dagger)
        return self.clean_constraints(constraints)

    def generate_cyclic_constraints(self) -> list[SingleTraceOperator]:
        # TODO it might be nice to implement this using a DoubleTraceOperator class
        # I definitely don't like how it is impelemented now
        constraints = {}
        for idx, op in enumerate(self.operator_list):
            if len(op) > 1:
                assert isinstance(op, tuple)
                # print(f'op = {op}, op[1:] + (op[0],) = {op[1:] + (op[0],)}') #GSH
                # the LHS corresponds to single trace operators
                # note that in Eq S37, S38 their RHS contains single trace operators (k=1, k=r) in their notation
                eq_lhs = SingleTraceOperator(data={op: 1}) - SingleTraceOperator(
                    data={op[1:] + (op[0],): 1}
                )
                # eq_lhs += self.matrix_system.commutation_rules[(op_str[0], op_str[1])] * SingleTraceOperator(data={tuple(op_str[2:]): 1})
                # eq_lhs += self.matrix_system.commutation_rules[(op_str[0], op_str[len(op_str)-1])] * SingleTraceOperator(data={tuple(op_str[1:len(op_str)-1]): 1})

                # rhe RHS corresponds to double trace operators
                eq_rhs = []
                for k in range(1, len(op)):
                    commutator = self.matrix_system.commutation_rules[(op[0], op[k])]
                    # if idx == 7:
                    #    print(f'[{op[0]}, {op[k]}] = ', commutator)
                    eq_rhs.append(
                        [
                            commutator,
                            SingleTraceOperator(data={tuple(op[1:k]): 1}),
                            SingleTraceOperator(data={tuple(op[k + 1 :]): 1}),
                        ]
                    )
                constraints[idx] = {"lhs": eq_lhs, "rhs": eq_rhs}
        return constraints

    def build_linear_constraints(
        self, return_matrix: bool = True
    ) -> Union[list[SingleTraceOperator, coo_matrix]]:
        """
        Build the linear constraints. Each linear constraint corresponds to a
        linear combination of single trace operators that must vanish. The set
        of linear constraints may be numerically represented as a matrix L_{ij},
        where the first index runs over the set of all such constraints, and the
        second index runs over the set of single trace operators considered at this
        bootstrap, i.e., the constraint equations are

        L_{ij} v_j = 0.

        Parameters
        ----------
        return_matrix : bool, optional
            Flag used to control whether a numerical matrix is returned (default) or
            a list of SingleTraceOperators, which can be helpful for debugging.
            By default True

        Returns
        -------
        Union[list[SingleTraceOperator, coo_matrix]]
            The set of linear constraints.
        """

        empty_operator = SingleTraceOperator(data={(): 0})
        constraints = []

        # Hamiltonian constraints
        for st_operator in self.generate_hamiltonian_constraints():
            if st_operator != empty_operator:
                constraints.append({op: coeff for op, coeff in st_operator})

        # gauge constraints
        for st_operator in self.generate_gauge_constraints():
            if st_operator != empty_operator:
                constraints.append({op: coeff for op, coeff in st_operator})

        # reality constraints
        #for st_operator in self.generate_reality_constraints():
        #    if st_operator != empty_operator:
        #        constraints.append({op: coeff for op, coeff in st_operator})

        # optionally return the constraints in a human-readable form
        if not return_matrix:
            return constraints

        # build the index-value dict
        index_value_dict = {}
        for idx_constraint, constraint_dict in enumerate(constraints):
            for op, coeff in constraint_dict.items():
                index_value_dict[(idx_constraint, self.operator_dict[op])] = coeff

        return create_sparse_matrix_from_dict(
            index_value_dict=index_value_dict,
            matrix_shape=(len(constraints), len(self.operator_list)),
        )

    def build_quadratic_constraints(self, null_space_matrix):
        """
        TODO return type
        TODO remove linearly dependent constraints
        Build the quadratic constraints. The quadratic constraints are exclusively due to
        the cyclic constraints. The constraints can be written as

        A_{ijk} v_j v_k + B_{ij} v_j = 0.

        After imposing the linear constraints by transforming to the null basis, these become

        A'_{iab} u_a u_b + B'_{ia} u_a = 0,

        where A'_{iab} = M_{ijk} K_{ja} K_{kb}, B'_{ia} = B_{ij} K_{ja}

        Returns
        -------
        _type_
            _description_
        """

        empty_operator = SingleTraceOperator(data={(): 0})
        constraints = self.generate_cyclic_constraints()

        linear_terms = []
        quadratic_terms = []

        for constraint_idx, (operator_idx, term) in enumerate(constraints.items()):
            if (term["lhs"] == empty_operator) != (
                sum(abs(x[0]) for x in term["rhs"]) < self.tol
            ):
                raise ValueError(
                    f"Warning, only one of (LHS, RHS) is trivial for quadratic constraint {constraint_idx}."
                )

            if term["lhs"] != empty_operator:
                # print(term['lhs'])
                linear_constraint_vector = self.single_trace_to_coefficient_vector(
                    term["lhs"]
                )

                quadratic_matrix = np.zeros(
                    (self.param_dim, self.param_dim)
                )  # TODO make real eventually
                for i in range(len(term["rhs"])):
                    coeff = term["rhs"][i][0]
                    if np.abs(coeff) > self.tol:
                        quadratic_constraint_vector_1 = (
                            self.single_trace_to_coefficient_vector(term["rhs"][i][1])
                        )
                        quadratic_constraint_vector_2 = (
                            self.single_trace_to_coefficient_vector(term["rhs"][i][2])
                        )
                        quadratic_matrix += coeff * np.outer(
                            quadratic_constraint_vector_2, quadratic_constraint_vector_2
                        )

                # transform to null basis
                linear_constraint_vector = np.dot(
                    linear_constraint_vector, null_space_matrix
                )
                quadratic_matrix = np.einsum(
                    "ia, ij, jb->ab",
                    null_space_matrix,
                    quadratic_matrix,
                    null_space_matrix,
                )
                # quadratic_constraint_vector_1 = np.dot(quadratic_constraint_vector_1, null_space_matrix)
                # quadratic_constraint_vector_2 = np.dot(quadratic_constraint_vector_2, null_space_matrix)

                # record
                if (np.max(np.abs(linear_constraint_vector)) > self.tol) and (
                    np.max(np.abs(quadratic_matrix)) > self.tol
                ):
                    linear_terms.append(linear_constraint_vector)
                    quadratic_terms.append(quadratic_matrix)

        return {"linear": np.asarray(linear_terms), "quadratic": np.asarray(quadratic_terms)}

    def clean_constraints(
        self, constraints: list[SingleTraceOperator]
    ) -> list[SingleTraceOperator]:
        """
        Remove constraints that involve operators outside the basis set.

        Parameters
        ----------
        constraints : list[SingleTraceOperator]
            The single trace constraints.

        Returns
        -------
        list[SingleTraceOperator]
            The cleaned constraints.
        """
        cleaned_constraints = []
        for st_operator in constraints:
            if all([op in self.operator_list for op in st_operator.data]):
                cleaned_constraints.append(st_operator)
        return cleaned_constraints

    def build_bootstrap_table(self, null_space_matrix) -> None:
        """
        TODO figure out return type

        Creates the bootstrap table.

        The bootstrap matrix is

        M_{ij} = < tr(O^dagger_i O_j)>.

        Representing the set of single trace operators considered at this
        boostrap level as a vector v, this may be written as

        M_{ij} = v_{I(i,j)},

        where I(i,j) is an index map.

        For example, in the one-matrix case,
          i = 1 corresponds to the string 'X'
          i = 4 corresponds to the string 'XP'
          M_{14} = < tr(XXP) > = v_8

        After imposing the linear constraints, the variable v becomes

        v_i = K_{ij} u_j

        and the bootstrap matrix becomes

        M_{ij} = K_{I(i,j)k} u_k

        Define the bootstrap table as the 3-index array K_{I(i,j)k}.
        Note that this is independent of the single trace variables.

        ------------------------------------------------------------------------
        NOTE
        In https://doi.org/10.1103/PhysRevLett.125.041601, M is simplified by imposing discrete
        symmetries. Each operator is assigned an integer-valued degree, and only expectations
        with zero total degree are non-zero:

        degree_total( O_1^{dagger} O_2 ) = -degree(O_1) + degree(O_2).

        The degree function depends on the symmetry. In the two-matrix example the degree is
        the charge of the operators (in the A, B, C, D basis) under the SO(2) generators. In
        the one-matrix example, it is the charge under reflection symmetry (X, P) -> (-X, -P).

        Imposing the condition that the total degree is zero leads to M being block diagonal,
        provided that the operators are sorted by degree, as in

        O = [(degree -d_min operators), (degree -d + 1 operators), ..., (degree d_max operators)]

        The blocks will then corresponds to the different ways of pairing the two operators
        O_i^dagger O_j to form a degree zero operator.

        No discrete symmetries will be imposed here
        ------------------------------------------------------------------------
        Returns
        -------
        X
            X
        """
        bootstrap_dict = {}
        for idx1, op_str1 in enumerate(self.operator_list[: self.psd_matrix_dim]):
            for idx2, op_str2 in enumerate(self.operator_list[: self.psd_matrix_dim]):
                index_map = self.operator_dict[op_str1 + op_str2]
                for k in range(null_space_matrix.shape[1]):
                    x = null_space_matrix[index_map, k]
                    if np.abs(x) > self.tol:
                        bootstrap_dict[(idx1, idx2, k)] = x
        return bootstrap_dict

    def build_optimization_problem(self):
        linear_constraint_matrix = self.build_linear_constraints().todense()
        null_space_matrix = get_null_space(linear_constraint_matrix)

        self.param_dim_null = null_space_matrix.shape[1]
        self.null_space_matrix = null_space_matrix

        bootstrap_dict = self.build_bootstrap_table(null_space_matrix=null_space_matrix)
        quadratic_constraints = self.build_quadratic_constraints(
            null_space_matrix=null_space_matrix
        )

        # map to a sparse array
        bootstrap_array = np.zeros((self.psd_matrix_dim, self.psd_matrix_dim, self.param_dim_null))
        for (i, j, k), value in bootstrap_dict.items():
            bootstrap_array[i,j,k] = value

        bootstrap_array_sparse = csr_matrix(bootstrap_array.reshape(bootstrap_array.shape[0] * bootstrap_array.shape[1], bootstrap_array.shape[2]))

        return bootstrap_array_sparse, quadratic_constraints
