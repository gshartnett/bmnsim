from collections import Counter
from itertools import product
from typing import (
    Optional,
    Union,
)
import os
import numpy as np
import pickle
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
    hstack,
    vstack,
    bmat,
    save_npz,
    load_npz
)

from bmn.algebra import (
    DoubleTraceOperator,
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.linear_algebra import (
    create_sparse_matrix_from_dict,
    get_null_space_sparse,
    get_row_space_sparse,
    get_real_coefficients_from_dict,
)
from bmn.debug_utils import debug


class BootstrapSystemComplex:
    """
    _summary_
    """

    def __init__(
        self,
        matrix_system: MatrixSystem,
        hamiltonian: SingleTraceOperator,
        gauge: MatrixOperator,
        max_degree_L: int,
        symmetry_generators: list[SingleTraceOperator] = None,
        tol: float = 1e-10,
        odd_degree_vanish=True,
        simplify_quadratic=True,
        save_path: Optional[str]=None,
        verbose: bool=False,
        fraction_operators_to_retain: float = 1.0,
    ):
        self.matrix_system = matrix_system
        self.hamiltonian = hamiltonian
        self.gauge = gauge
        self.max_degree_L = max_degree_L
        self.odd_degree_vanish = odd_degree_vanish
        self.fraction_operators_to_retain = fraction_operators_to_retain
        self.operator_list = self.generate_operators_truncated(
            L=max_degree_L,
            fraction_operators_to_retain=fraction_operators_to_retain
            )
        self.operator_dict = {op: idx for idx, op in enumerate(self.operator_list)}
        if 2 * self.max_degree_L < self.hamiltonian.max_degree:
            raise ValueError(
                "2 * max_degree_L must be >= max degree of Hamiltonian."
            )
        self.param_dim_complex = len(self.operator_dict)
        self.param_dim_real = 2 * len(self.operator_dict)
        self.tol = tol
        self.null_space_matrix = None
        self.linear_constraints = None
        self.quadratic_constraints = None
        self.simplify_quadratic = simplify_quadratic
        self.symmetry_generators = symmetry_generators
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        self.save_path = save_path
        self.verbose = verbose
        self._validate()

    def _validate(self):
        """
        Check that the operator basis used in matrix_system is consistent with gauge and hamiltonian.
        """
        self.validate_operator(operator=self.gauge)
        self.validate_operator(operator=self.hamiltonian)
        print(f"Bootstrap system instantiated for {len(self.operator_dict)} operators")
        print(f"Attribute: simplify_quadratic = {self.simplify_quadratic}")

    def validate_operator(self, operator: MatrixOperator):
        """
        Check that the operator only contains terms used in the matrix_system.
        """
        for op, coeff in operator:
            for op_str in op:
                if not (op_str in self.matrix_system.operator_basis):
                    raise ValueError(
                        f"Invalid operator: constrains term {op_str} which is not in operator_basis."
                    )

    def load_constraints(self):
        if self.save_path is None:
            raise ValueError("Error, no save path provided.")
        if not os.path.exists(self.save_path):
            raise ValueError(f"Error, save path {self.save_path} does not exist.")

        # load the linear constraints
        with open(self.save_path + "/linear_constraints_data.pkl", 'rb') as f:
            loaded_data = pickle.load(f)
        self.linear_constraints = [SingleTraceOperator(data=data) for data in loaded_data]

        # load the cyclic quadratic constraints
        with open(self.save_path + "/cyclic_quadratic.pkl", 'rb') as f:
            loaded_data = pickle.load(f)
        self.quadratic_constraints = {key: {'lhs': SingleTraceOperator(data=value['lhs']), 'rhs': DoubleTraceOperator(data=value['rhs'])} for key, value in loaded_data.items()}

        self.null_space_matrix = load_npz(self.save_path + "/null_space_matrix.npz")
        self.param_dim_null = self.null_space_matrix.shape[1]

    def scale_param_to_enforce_normalization(self, param: np.ndarray) -> np.ndarray:
        """
        Rescale the parameter vector to enforce the normalization condition that <1> = 1.

        Parameters
        ----------
        param : np.ndarray
            The input param.

        Returns
        -------
        np.ndarray
            The rescaled param.
        """
        raise NotImplementedError

    def build_null_space_matrix(
        self, additional_constraints: Optional[list[SingleTraceOperator]] = None
    ) -> np.ndarray:
        """
        Builds the null space matrix, K_{ia}.

        Note that K_{ia} K_{ib} = delta_{ab}.
        """
        linear_constraint_matrix = self.build_linear_constraints(additional_constraints)

        if self.verbose:
            debug(f"Building the null space matrix. The linear constraint matrix has dimensions {linear_constraint_matrix.shape}")

        self.null_space_matrix = get_null_space_sparse(linear_constraint_matrix)
        self.param_dim_null = self.null_space_matrix.shape[1]
        print(f"Null space dimension (number of parameters) = {self.param_dim_null}")

        if self.save_path is not None:
            save_npz(self.save_path + "/null_space_matrix.npz", self.null_space_matrix)

    def generate_operators_truncated(self, L, fraction_operators_to_retain=1.0):

        # generate all operators with length <= L
        operators = [[x for x in product(self.matrix_system.operator_basis, repeat=deg)] for deg in range(0, L + 1)]
        operators = [x for operators_by_degree in operators for x in operators_by_degree]
        print(f"Number of operators with length <= {L}: {len(operators)}")

        # truncate the set of operators considered
        if fraction_operators_to_retain < 1.0:

            # for reference find the number of operators in the multiplication table if we had not done the truncation
            untruncated_number_of_operators = len(list(set([op_str1[::-1] + op_str2 for op_str1 in operators for op_str2 in operators])))

            # perform the truncation
            num_operators_to_retain = int(fraction_operators_to_retain * len(operators))
            operators_retain = set(operators[0:num_operators_to_retain])
            operators = [op_str for op_str in operators_retain if op_str[::-1] in operators_retain]

            # add back any conjugates
            print(f"Number of operators with length <= {L} after truncation: {len(operators)}")

        # generate all matrices appearing in the LxL multiplication table
        operator_list = list(set([op_str1[::-1] + op_str2 for op_str1 in operators for op_str2 in operators]))

        if fraction_operators_to_retain < 1.0:
            print(f"Number of operators appearing in the L x L multiplication table (before truncation): {untruncated_number_of_operators}")
            print(f"Number of operators appearing in the L x L multiplication table (after truncation): {len(operator_list)}")
        else:
            print(f"Number of operators appearing in the L x L multiplication table: {len(operator_list)}")

        '''# arrange list in blocks of even/odd degree, i.e.,
        # operators_by_degree = {
        #   0: [(), ('X', 'X'), ..., ]
        #   1: [('X'), ('P'), ..., ]
        # }
        operators_by_degree = {}
        degree_func = lambda x: len(x)
        for op in operator_list:
            degree = degree_func(op)
            if degree not in operators_by_degree:
                operators_by_degree[degree] = [op]
            else:
                operators_by_degree[degree].append(op)

        print(len(operators_by_degree[0]), len(operators_by_degree[1]))
        print(operators_by_degree)

        # arrange the operators with degree <= L by degree mod 2
        # for building the bootstrap matrix
        bootstrap_basis_list = []
        for deg, op_list in operators_by_degree.items():
            if deg % 2 == 0 and deg <= self.max_degree_L:
                bootstrap_basis_list.extend(op_list)
        for deg, op_list in operators_by_degree.items():
            if deg % 2 != 0 and deg <= self.max_degree_L:
                bootstrap_basis_list.extend(op_list)
        '''
        self.bootstrap_basis_list = operators
        self.bootstrap_matrix_dim = len(operators)

        return operator_list


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
        operator_list = [x for xs in operators.values() for x in xs]

        if self.max_num_operators is not None:
            operator_list = operator_list[0: self.max_num_operators]
            for op in operator_list:
                op_reverse = op[::-1]
                if op_reverse not in operator_list:
                    operator_list.append(op_reverse)

        # arrange list in blocks of even/odd degree, i.e.,
        # operators_by_degree = {
        #   0: [(), ('X', 'X'), ..., ]
        #   1: [('X'), ('P'), ..., ]
        # }
        operators_by_degree = {}
        degree_func = lambda x: len(x)
        for op in operator_list:
            degree = degree_func(op)
            if degree not in operators_by_degree:
                operators_by_degree[degree] = [op]
            else:
                operators_by_degree[degree].append(op)

        # arrange the operators with degree <= L by degree mod 2
        # for building the bootstrap matrix
        bootstrap_basis_list = []
        for deg, op_list in operators.items():
            if deg % 2 == 0 and deg <= self.max_degree_L:
                bootstrap_basis_list.extend(op_list)
        for deg, op_list in operators.items():
            if deg % 2 != 0 and deg <= self.max_degree_L:
                bootstrap_basis_list.extend(op_list)
        self.bootstrap_basis_list = bootstrap_basis_list
        self.bootstrap_matrix_dim = len(bootstrap_basis_list)

        return operator_list

    def single_trace_to_coefficient_vector(
        self, st_operator: SingleTraceOperator, return_null_basis: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map a single trace operator to the (in general complex) vector of coefficients.
        Here, a real representation for the *operators* will be used, meaning that a
        general single trace operator will be written as <tr(O)> = sum_i a_i v_i.

        Optionally returns the vectors in the null basis. The null basis transformation
        acts on a real representation of the operator basis elements, i.e. the operator above
        is first written as sum_i z_i (vR_i + i vI_i), so that the coefficient vector is [z, z].

        Parameters
        ----------
        st_operator : SingleTraceOperator
            The operator

        return_null_basis : bool, optional
            Controls whether the vector is returned in the original basis or the null basis.
            By default False.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The real and imaginary parts of the coefficient vector.
        """

        # validate
        self.validate_operator(operator=st_operator)

        # build the complex-basis vector of coefficients
        vec = [0] * self.param_dim_complex
        for op, coeff in st_operator:
            idx = self.operator_dict[op]
            vec[idx] = coeff
        vec = np.asarray(vec)
        if not return_null_basis:
            return vec

        # return the real-basis vector of coefficients and convert to null space
        if self.null_space_matrix is None:
            raise ValueError("Error, must first build null space.")

        vec = np.concatenate((vec, vec))
        return vec @ self.null_space_matrix

    def double_trace_to_coefficient_matrix(self, dt_operator: DoubleTraceOperator):
        # use large-N factorization <tr(O1)tr(O2)> = <tr(O1)><tr(O2)>
        # to represent the double-trace operator as a quadratic expression of single trace operators,
        # \sum_{ij} M_{ij} v_i v_j, represented as an index-value dict.

        index_value_dict = {}
        for (op1, op2), coeff in dt_operator:
            idx1, idx2 = self.operator_dict[op1], self.operator_dict[op2]

            # symmetrize
            index_value_dict[(idx1, idx2)] = index_value_dict.get((idx1, idx2), 0) + coeff / 2
            index_value_dict[(idx2, idx1)] = index_value_dict.get((idx2, idx1), 0) + coeff / 2

        mat = create_sparse_matrix_from_dict(
            index_value_dict=index_value_dict,
            matrix_shape=(len(self.operator_list), len(self.operator_list)),
        )

        return mat

    def generate_symmetry_constraints(self, tol=1e-10) -> list[SingleTraceOperator]:
        """
        Generate any symmetry constraints <[g,O]>=0 for O single trace
        and g a symmetry generator.

        Returns
        -------
        list[SingleTraceOperator]
            The list of constraint terms.
        """
        # skip if no symmetry generators are provided
        if self.symmetry_generators is None:
            return []

        constraints = []
        n = len(self.matrix_system.operator_basis)

        # loop over symmetry generators
        for symmetry_generator in self.symmetry_generators:

            # initialize a matrix M which will implement the linear action of the generator g
            # M will obey [g, operators_vector] = M operators_vector
            M = np.zeros(shape=(n, n), dtype=np.complex128)
            for i, op in enumerate(self.matrix_system.operator_basis):
                commutator = self.matrix_system.single_trace_commutator(
                    symmetry_generator,
                    SingleTraceOperator(data={(op): 1})
                )
                for op, coeff in commutator:
                    if np.abs(coeff) > tol:
                        j = self.matrix_system.operator_basis.index(op[0])
                        M[i,j] = coeff

            # build the change of variables matrix
            eig_values, old_to_new_variables = np.linalg.eig(M)
            old_to_new_variables = old_to_new_variables.T

            # confirm that the eigenvector relationship holds
            assert np.all([np.allclose(np.zeros(n), M @ old_to_new_variables[i] - eig_values[i] * old_to_new_variables[i]) for i in range(n)])

            # build all monomials using the new operators
            if self.fraction_operators_to_retain != 1.0:
                raise ValueError("Warning, symmetry constraints and dropping a fraction of operators are not simultaneously supported.")

            new_ops_dict = {f"new_op_{i}":i for i in range(n)}
            all_new_operators = {deg: [x for x in product(new_ops_dict.keys(), repeat=deg)] for deg in range(1, 2*self.max_degree_L + 1)}
            all_new_operators = [x for xs in all_new_operators.values() for x in xs]  # flatten

            # loop over all operators in the eigenbasis
            for operator in all_new_operators:

                # compute the charge under the symmetry
                charge = sum([eig_values[new_ops_dict[basis_op]] for basis_op in operator])

                # if the charge is not zero, the resulting operator expectation value must vanish in a symmetric state
                if np.abs(charge) > tol:

                    operator2 = {}
                    for i in range(len(operator)):
                        operator2[i] = [(self.matrix_system.operator_basis[j], old_to_new_variables[new_ops_dict[operator[i]], j]) for j in range(n)]

                    # build the constraint single-trace operator
                    data = {}
                    for indices in list(product(range(n), repeat=len(operator))):
                        op = tuple([value[indices[idx]][0] for idx, value in enumerate(operator2.values())])
                        coeff = np.prod([value[indices[idx]][1] for idx, value in enumerate(operator2.values())])
                        if np.abs(coeff) > tol:
                            data[op] = data.get(op, 0) + coeff
                    constraints.append(SingleTraceOperator(data=data))

        return self.clean_constraints(constraints)

    def generate_hamiltonian_constraints(self) -> list[SingleTraceOperator]:
        """
        Generate the Hamiltonian constraints <[H,O]>=0 for O single trace.

        Returns
        -------
        list[SingleTraceOperator]
            The list of constraint terms.
        """
        constraints = []

        for op_idx, op in enumerate(self.operator_list):
            constraints.append(
                self.matrix_system.single_trace_commutator(
                    st_operator1=self.hamiltonian,
                    st_operator2=SingleTraceOperator(data={op: 1}),
                )
            )
            constraints.append(
                self.matrix_system.single_trace_commutator(
                    st_operator1=SingleTraceOperator(data={op: 1}),
                    st_operator2=self.hamiltonian,
                )
            )
            if self.verbose:
                debug(f"Generating Hamiltonian constraints, operator {op_idx+1}/{len(self.operator_list)}")

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
        for op_idx, op in enumerate(self.operator_list):
            constraints.append((self.gauge * MatrixOperator(data={op: 1})).trace())

            if self.verbose:
                debug(f"Generating gauge constraints, operator {op_idx+1}/{len(self.operator_list)}")

        return self.clean_constraints(constraints)

    def generate_odd_degree_vanish_constraints(self) -> list[SingleTraceOperator]:
        constraints = []
        for op in self.operator_list:
            if len(op) % 2 == 1:
                constraints.append(SingleTraceOperator(data={op: 1}))
        return constraints

    def generate_cyclic_constraints(
        self,
    ) -> tuple[int: SingleTraceOperator, dict[int, dict[str, SingleTraceOperator | DoubleTraceOperator]]]:
        """
        Generate cyclic constraints relating single trace operators to double
        trace operators. See S37 of
        https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.125.041601/supp.pdf

        Returns
        -------
        dict[int, dict[str, SingleTraceOperator | DoubleTraceOperator]]
            The linear and quadratic constraints.
        """
        identity = SingleTraceOperator(data={(): 1})
        quadratic_constraints = {}
        linear_constraints = {}
        for op_idx, op in enumerate(self.operator_list):
            if len(op) > 1:

                if not isinstance(op, tuple):
                    raise ValueError(f"op should be tuple, not {type(op)}")

                # the LHS corresponds to single trace operators
                eq_lhs = SingleTraceOperator(data={op: 1}) - SingleTraceOperator(
                    data={op[1:] + (op[0],): 1}
                )

                # rhe RHS corresponds to double trace operators
                # eq_rhs = []
                eq_rhs = DoubleTraceOperator(data={})
                for k in range(1, len(op)):
                    commutator = self.matrix_system.commutation_rules[(op[0], op[k])]
                    st_operator_1 = SingleTraceOperator(data={tuple(op[1:k]): 1})
                    st_operator_2 = SingleTraceOperator(data={tuple(op[k + 1 :]): 1})

                    # If the double trace term involves <tr(1)> simplify and add to the linear, LHS
                    if st_operator_1 == identity:
                        eq_lhs -= commutator * st_operator_2
                    elif st_operator_2 == identity:
                        eq_lhs -= commutator * st_operator_1
                    else:
                        eq_rhs += commutator * (st_operator_1 * st_operator_2)

                # if the quadratic term vanishes but the linear term is non-zero, record the constraint as being linear
                if not eq_lhs.is_zero() and eq_rhs.is_zero():
                    linear_constraints[op_idx] = eq_lhs

                # do not expect to find any constraints where the linear term vanishes but the quadratic term does not
                elif eq_lhs.is_zero() and not eq_rhs.is_zero():
                    raise ValueError(
                        f"Warning, for operator index {op_idx}, op={op}, the LHS is unexpectedly 0"
                    )

                # record proper quadratic constraints
                elif not eq_lhs.is_zero():
                    quadratic_constraints[op_idx] = {"lhs": eq_lhs, "rhs": eq_rhs}

            if self.verbose:
                debug(f"Generating cyclic constraints, operator {op_idx+1}/{len(self.operator_list)}")

        return linear_constraints, quadratic_constraints

    def generate_constraints(self) -> tuple[list[SingleTraceOperator]]:
        """
        Generate all constraints.

        Returns
        -------
        tuple[list[SingleTraceOperator]]
            The first entry in the tuple is the list of linear constraints.
            The second is the list of cyclic constraints.
        """
        linear_constraints = []

        # Hamiltonian constraints
        hamiltonian_constraints = self.generate_hamiltonian_constraints()
        print(f"Generated {len(hamiltonian_constraints)} Hamiltonian constraints")
        linear_constraints.extend(hamiltonian_constraints)

        # gauge constraints
        gauge_constraints = self.generate_gauge_constraints()
        print(f"Generated {len(gauge_constraints)} gauge constraints")
        linear_constraints.extend(gauge_constraints)

        # symmetry constraints
        if self.symmetry_generators is not None:
            symmetry_constraints = self.generate_symmetry_constraints()
            print(f"Generated {len(symmetry_constraints)} symmetry constraints")
            linear_constraints.extend(symmetry_constraints)

        # odd degree vanish
        if self.odd_degree_vanish:
            odd_degree_constraints = self.generate_odd_degree_vanish_constraints()
            print(
                f"Generated {len(odd_degree_constraints)} odd degree vanish constraints"
            )
            linear_constraints.extend(odd_degree_constraints)

        # cyclic constraints
        cyclic_linear, cyclic_quadratic = self.generate_cyclic_constraints()
        cyclic_linear = list(cyclic_linear.values())
        print(f"Generated {len(cyclic_linear)} linear cyclic constraints")
        print(f"Generated {len(cyclic_quadratic)} quadratic cyclic constraints")
        linear_constraints.extend(cyclic_linear)

        # NOTE pretty sure this is not necessary
        linear_constraints.extend(
            [self.matrix_system.hermitian_conjugate(op) for op in linear_constraints]
        )

        # XXX
        #linear_constraints = self.clean_constraints(linear_constraints)

        # save the constraints
        if self.save_path is not None:
            with open(self.save_path + "/linear_constraints_data.pkl", 'wb') as f:
                pickle.dump([constraint.data for constraint in linear_constraints], f)
            with open(self.save_path + "/cyclic_quadratic.pkl", 'wb') as f:
                cyclic_data_dict = {}
                for key, value in cyclic_quadratic.items():
                    cyclic_data_dict[key] = {'lhs': value['lhs'].data, 'rhs': value['rhs'].data}
                pickle.dump(cyclic_data_dict, f)

        return self.clean_constraints(linear_constraints), self.clean_constraints_quadratic(cyclic_quadratic)

    def build_linear_constraints(
        self, additional_constraints: Optional[list[SingleTraceOperator]] = None
    ) -> coo_matrix:
        """
        Build the linear constraints. Each linear constraint corresponds to a
        linear combination of single trace operators that must vanish. The set
        of linear constraints may be numerically represented as a matrix L_{ij},
        where the first index runs over the set of all such constraints, and the
        second index runs over the set of single trace operators considered at this
        bootstrap, i.e., the constraint equations are

        L_{ij} v_j = 0.

        NOTE: Not sure if it's a good idea to store the constraints, may become memory intensive.

        NOTE: In this implementation of the bootstrap, the constraints are complex-valued. However,
        the subsequent operations assume that the objects are real-valued.

        To address this, v = vR + i vI will be made real by stacking the real and imaginary parts,
        as in V = [vR, vI].

        The constraints L v = 0 can also be decomposed into real and imaginary parts, let L_ij = X_ij + i Y_ij
        be the real and imaginary components of the constraint coefficients. So then

        (L v)_i = L_ij (vR_j + i vI_j) = (X_ij + i Y_ij) (vR_j + i vI_j)
                = (X_ij vR_j - Y_ij vI_j) + i (Y_ij vR_j + X_ij vI_j)

        Note that the real and imaginary parts of the equation must separately hold.
        These can be jointly written as LL V = 0, where

        LL = [[X, -Y],
            [Y, X]]

        Note also that a permutation of the rows will not affect the content of the
        constraints.

        Returns
        -------
        coo_matrix
            The set of linear constraints.
        """
        if self.verbose:
            debug(f"Building the linear constraint matrix")

        # grab the constraints, building them if necessary
        if self.linear_constraints is None:
            constraints = self.generate_constraints()
            self.linear_constraints = constraints[0]
            self.quadratic_constraints = constraints[1]

        # add the additional constraints
        if additional_constraints is not None:
            self.linear_constraints += additional_constraints

        # build the index-value dict
        index_value_dict = {}
        constraint_idx = 0

        # loop over operators
        for st_operator in self.linear_constraints:
            for op_str, coeff in st_operator:
                index_value_dict[(constraint_idx, self.operator_dict[op_str])] = np.real(coeff)
                index_value_dict[(constraint_idx, self.operator_dict[op_str] + self.param_dim_complex)] = -np.imag(coeff)
            constraint_idx += 1

        # imaginary part (Y_ij vR_j + X_ij vI_j)
        for st_operator in self.linear_constraints:
            for op_str, coeff in st_operator:
                index_value_dict[(constraint_idx, self.operator_dict[op_str])] = np.imag(coeff)
                index_value_dict[(constraint_idx, self.operator_dict[op_str] + self.param_dim_complex)] = np.real(coeff)
            constraint_idx += 1

        # impose the reality constraints <tr(O^{dagger})> = <tr(O)>* (real part)
        # if O1^{dagger} = O2, and <tr(O1)> = vR_1 + i vI_1, <tr(O2)> = vR_2 + i vI_2
        # then the condition becomes
        # vR_1 = vR_2 and vI_1 = - vI_2
        # Note that Hermitian operators must be treated separately, else the dictionary encoding will be incorrect
        for op_str, op_idx in self.operator_dict.items():

            op_str_reversed = op_str[::-1]
            op_reversed_idx = self.operator_dict[op_str_reversed]

            # Hermitian operators: set imaginary part to zero
            if op_reversed_idx == op_idx:
                index_value_dict[(constraint_idx, op_idx + self.param_dim_complex)] = 1
                constraint_idx += 1

            # non-Hermitian operators
            else:
                # real part
                index_value_dict[(constraint_idx, op_idx)] = 1
                index_value_dict[(constraint_idx, op_reversed_idx)] = -1
                constraint_idx += 1

                # imaginary part
                index_value_dict[(constraint_idx, op_idx + self.param_dim_complex)] = 1
                index_value_dict[(constraint_idx, op_reversed_idx + self.param_dim_complex)] = 1
                constraint_idx += 1

        # assemble the constraint matrix
        linear_constraint_matrix = create_sparse_matrix_from_dict(
            index_value_dict=index_value_dict,
            matrix_shape=(constraint_idx, 2 * self.param_dim_complex),
        )

        return linear_constraint_matrix

    def build_quadratic_constraints(self) -> dict[str, np.ndarray]:
        """
        Build the quadratic constraints. The quadratic constraints are exclusively due to
        the cyclic constraints. The constraints can be written as

        A_{Iij} v_i v_j + B_{Ii} v_i = 0.

        After imposing the linear constraints by transforming to the null basis, these become

        A'_{Iab} u_a u_b + B'_{Ia} u_a = 0,

        where A'_{Iab} = A_{Iij} K_{ia} K_{jb}, B'_{Ia} = B_{Ii} K_{ia}

        The quadratic constraint tensor can be written as
            A_{Iij} = (1/2) sum_{k=0}^{K_I-1} (v_i^{(I,k)} v_j^{(I,k)} + v_j^{(I,k)} v_i^{(I,k)})
        for vectors v^{(I,k)}. For each I, the vectors correspond to the double trace terms, <tr()> <tr()>.

        Returns
        -------
        dict[str, np.nparray]
            The constraint arrays, contained in a dictionary like so
            {'quadratic': A, 'linear': B}
        """

        if self.quadratic_constraints is None:
            self.build_linear_constraints()
        quadratic_constraints = self.quadratic_constraints

        additional_constraints = []
        if self.null_space_matrix is None:
            self.build_null_space_matrix()
        null_space_matrix = self.null_space_matrix

        linear_terms = []
        quadratic_terms = []

        # add <1> = <1>^2
        normalization_constraint = {
            'lhs': SingleTraceOperator(data={(): 1}),
            'rhs': DoubleTraceOperator(data={((), ()): 1}),
        }
        quadratic_constraints[None] = normalization_constraint

        # loop over constraints
        for constraint_idx, constraint in enumerate(quadratic_constraints.values()):

            if self.verbose:
                debug(f"Generating quadratic constraints, operator {constraint_idx+1}/{len(quadratic_constraints)}")

            lhs = constraint["lhs"]
            rhs = constraint["rhs"]

            #print(f"constraint: lhs = {lhs}, rhs = {rhs}")

            # retrieve the vectorized form of the linear and quadratic terms
            # Note: these will be complex-valued and in the complex operator basis
            linear_constraint_vector = self.single_trace_to_coefficient_vector(lhs)
            quadratic_matrix = self.double_trace_to_coefficient_matrix(rhs)

            # convert to real operator basis, and split each constraint into a real and imaginary part
            # sum_k z_k v_k = sum_k (x_k vR_k - y_k vI_k) + i sum_k (y_k vR_k + x_k vI_k)
            # so that the real vector is [real(z), - imag(z)] and the imaginary term is [imag(z), real(z)]
            linear_constraint_vectorR = np.concatenate((linear_constraint_vector.real, -linear_constraint_vector.imag))
            linear_constraint_vectorI = np.concatenate((linear_constraint_vector.imag, linear_constraint_vector.real))

            # rewrite the quadratic constraints in terms of real variables
            # this entails two things:
            #   1. every constraint will become two (one real and one imaginary)
            #   2. each constraint will be naturally expressed as a (2d, 2d) matrix
            #      acting on the stacked parameter vector [vR, vI]
            qR = quadratic_matrix.real
            qI = quadratic_matrix.imag

            QR = bmat([[qR, -qI], [-qI, -qR]], format='coo')
            QI = bmat([[qI, qR], [qR, -qI]], format='coo')

            # transform to null basis
            # the minus sign is very important: (-RHS + LHS = 0)
            linear_constraint_vectorR = linear_constraint_vectorR @ null_space_matrix
            linear_constraint_vectorI = linear_constraint_vectorI @ null_space_matrix

            QR = (
                -null_space_matrix.T @ QR @ null_space_matrix
            )
            QI = (
                -null_space_matrix.T @ QI @ null_space_matrix
            )

            # reshape the (d, d) matrices to (1, d^2) matrices
            QR = QR.reshape((1, self.param_dim_null**2))
            QI = QI.reshape((1, self.param_dim_null**2))

            # process the real and imaginary constraints separately
            # real part
            linear_is_zero = np.max(np.abs(linear_constraint_vectorR)) < self.tol
            quadratic_is_zero = np.max(np.abs(QR)) < self.tol
            if self.simplify_quadratic:
                if not quadratic_is_zero:
                    linear_terms.append(csr_matrix(linear_constraint_vectorR))
                    quadratic_terms.append(QR)
                elif not linear_is_zero:
                    additional_constraints.append(lhs.get_real_part())
            else:
                if not quadratic_is_zero or not linear_is_zero:
                    linear_terms.append(csr_matrix(linear_constraint_vectorR))
                    quadratic_terms.append(QR)

            # imaginary part
            linear_is_zero = np.max(np.abs(linear_constraint_vectorI)) < self.tol
            quadratic_is_zero = np.max(np.abs(QI)) < self.tol
            if self.simplify_quadratic:
                if not quadratic_is_zero:
                    linear_terms.append(csr_matrix(linear_constraint_vectorI))
                    quadratic_terms.append(QI)
                elif not linear_is_zero:
                    additional_constraints.append(lhs.get_imag_part())
            else:
                if not quadratic_is_zero or not linear_is_zero:
                    linear_terms.append(csr_matrix(linear_constraint_vectorI))
                    quadratic_terms.append(QI)

        print(f"len(quadratic_terms) = {len(quadratic_terms)}")
        if len(quadratic_terms) == 0:
            raise ValueError

        if self.simplify_quadratic and len(additional_constraints) > 0:
            print(
                f"Building quadratic constraints: adding {len(additional_constraints)} new linear constraints and rebuilding null matrix"
            )
            self.build_null_space_matrix(additional_constraints=additional_constraints)
            return self.build_quadratic_constraints()

        # map to sparse matrices
        quadratic_terms = vstack(quadratic_terms)
        linear_terms = vstack(linear_terms)

        # apply reduction
        num_constraints = quadratic_terms.shape[0]

        print(
            f"Number of quadratic constraints before row reduction: {num_constraints}"
        )

        print(f"NOTE: not applying row reduction...")
        '''
        for some reason this is taking forever...
        stacked_matrix = hstack([quadratic_terms, linear_terms])
        stacked_matrix = get_row_space_sparse(stacked_matrix)
        num_constraints = stacked_matrix.shape[0]
        linear_terms = stacked_matrix[:, self.param_dim_null**2 :]
        quadratic_terms = stacked_matrix[:, : self.param_dim_null**2]
        print(f"Number of quadratic constraints after row reduction: {num_constraints}")
        '''

        return {
            "linear": linear_terms,
            "quadratic": quadratic_terms,
        }

    def clean_constraints(
        self, constraints: list[SingleTraceOperator]
    ) -> list[SingleTraceOperator]:
        """
        Remove constraints that involve operators outside the operator list.
        Also remove empty constraints of the form 0=0.

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

        # use a set to check membership as this operation is O(1) vs O(N) for lists
        set_of_all_operators = set(self.operator_list)
        for st_operator in constraints:
            if (
                all([op in set_of_all_operators for op in st_operator.data])
                and not st_operator.is_zero()
            ):
                cleaned_constraints.append(st_operator)
        return cleaned_constraints

    def clean_constraints_quadratic(
        self, constraints: dict[int, dict[str, SingleTraceOperator | DoubleTraceOperator]]
    ) -> dict[int, dict[str, SingleTraceOperator | DoubleTraceOperator]]:
        """
        Remove constraints that involve operators outside the operator list.
        Also remove empty constraints of the form 0=0.

        Parameters
        ----------
        constraints : list[SingleTraceOperator]
            The single trace constraints.

        Returns
        -------
        list[SingleTraceOperator]
            The cleaned constraints.
        """
        cleaned_constraints = {}

        # use a set to check membership as this operation is O(1) vs O(N) for lists
        set_of_all_operators = set(self.operator_list)

        for constraint_idx, quadratic_constraint in constraints.items():
            lhs = quadratic_constraint['lhs']
            rhs = quadratic_constraint['rhs']

            lhs_in_basis = all([op in set_of_all_operators for op in lhs.data])
            rhs_in_basis = all([op1 in set_of_all_operators and op2 in set_of_all_operators for (op1, op2) in rhs.data])

            if lhs_in_basis and rhs_in_basis:
                cleaned_constraints[constraint_idx] = {
                    'lhs': lhs,
                    'rhs': rhs,
                }

        return cleaned_constraints

    def build_bootstrap_table(self) -> csr_matrix:
        """
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
        csr_matrix
            The bootstrap array, with shape (self.bootstrap_matrix_dim**2, self.param_dim_null).
            It has been reshaped to be a matrix.
        """
        if self.null_space_matrix is None:
            raise ValueError("Error, null space matrix has not yet been built.")
        null_space_matrix = self.null_space_matrix

        bootstrap_dict = {}
        for idx1, op_str1 in enumerate(self.bootstrap_basis_list):
            op_str1 = op_str1[::-1]  # take the h.c. by reversing the elements
            for idx2, op_str2 in enumerate(self.bootstrap_basis_list):

                # tally up number of anti-hermitian operators, and add (-1) factor if odd
                num_antihermitian_ops = sum([not self.matrix_system.hermitian_dict[term] for term in op_str1])
                sign = (-1) ** num_antihermitian_ops

                # grab the index of the operator O_1^dag O_2
                index_map = self.operator_dict[op_str1 + op_str2]

                for k in range(null_space_matrix.shape[1]):
                    x = sign * null_space_matrix[index_map, k]
                    if np.abs(x) > self.tol:
                        #print(f"op1 = {op_str1[::-1]}, op2 = {op_str2}, op1* + op2 = {op_str1 + op_str2}, index = {index_map}, k = {k}, val={x}")
                        bootstrap_dict[(idx1, idx2, k)] = x

        #return bootstrap_dict

        # map to a sparse array
        bootstrap_array = np.zeros(
            (self.bootstrap_matrix_dim, self.bootstrap_matrix_dim, self.param_dim_null),
            dtype=np.float64,
        )
        for (i, j, k), value in bootstrap_dict.items():
            bootstrap_array[i, j, k] = value

        bootstrap_array_sparse = csr_matrix(
            bootstrap_array.reshape(
                bootstrap_array.shape[0] * bootstrap_array.shape[1],
                bootstrap_array.shape[2],
            )
        )

        return bootstrap_array_sparse

    def get_bootstrap_matrix(self, param: np.ndarray):
        dim = self.bootstrap_matrix_dim
        bootstrap_array_sparse = self.build_bootstrap_table()

        bootstrap_matrix = np.reshape(
            bootstrap_array_sparse.dot(param), (dim, dim)
        )

        # verify that matrix is symmetric
        # the general condition is that the matrix is Hermitian, but we have made it real
        # NOTE: this property only holds when the reality constraints are imposed - only then
        # do we have a relation b/w for example <tr(XP)> and <tr(XP)>.
        if not np.allclose(
            (bootstrap_matrix - bootstrap_matrix.T), np.zeros_like(bootstrap_matrix)
        ):
            violation = np.max((bootstrap_matrix - bootstrap_matrix.T))
            raise ValueError(f"Bootstrap matrix is not symmetric, violation = {violation}")

        return bootstrap_matrix

    def get_operator_expectation_value(
            self,
            st_operator: SingleTraceOperator,
            param: np.ndarray
            ) -> float:
        param_real = (self.null_space_matrix @ param)[:self.param_dim_complex]
        param_imag = (self.null_space_matrix @ param)[self.param_dim_complex:]
        param_complex = param_real + 1j * param_imag

        vec = self.single_trace_to_coefficient_vector(
                    st_operator=st_operator, return_null_basis=False
                )

        op_expectation_value = vec @ param_complex
        return op_expectation_value