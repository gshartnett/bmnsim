import os
import pickle
from itertools import product
from typing import Optional

import numpy as np
from scipy.linalg import ishermitian
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
    hstack,
    load_npz,
    save_npz,
    vstack,
)

from bmn.algebra import (
    DoubleTraceOperator,
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)
from bmn.debug_utils import debug
from bmn.linear_algebra import (
    create_sparse_matrix_from_dict,
    get_null_space_sparse,
    get_row_space_sparse,
)


class BootstrapSystem:
    """
    _summary_
    """

    def __init__(
        self,
        matrix_system: MatrixSystem,
        hamiltonian: SingleTraceOperator,
        gauge_generator: MatrixOperator,
        max_degree_L: int,
        symmetry_generators: list[SingleTraceOperator] = None,
        tol: float = 1e-10,
        odd_degree_vanish=True,
        simplify_quadratic=True,
        verbose: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        self.matrix_system = matrix_system
        self.hamiltonian = hamiltonian
        self.gauge_generator = gauge_generator
        self.max_degree_L = max_degree_L
        self.odd_degree_vanish = odd_degree_vanish
        self.operator_list = self.generate_operators(2 * max_degree_L)
        self.operator_dict = {op: idx for idx, op in enumerate(self.operator_list)}
        self.conversion_vec = np.asarray(
            [1 if len(op) % 2 == 0 else 1j for op in self.operator_list]
        )
        if 2 * self.max_degree_L < self.hamiltonian.max_degree:
            raise ValueError("2 * max_degree_L must be >= max degree of Hamiltonian.")
        self.param_dim = len(self.operator_dict)
        self.tol = tol
        self.null_space_matrix = None
        self.linear_constraints = None
        self.quadratic_constraints = None
        self.quadratic_constraints_numerical = None
        self.bootstrap_table_sparse = None
        self.simplify_quadratic = simplify_quadratic
        self.symmetry_generators = symmetry_generators
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path is not None:
            os.makedirs(self.checkpoint_path, exist_ok=True)
        self.verbose = verbose
        self._validate()

    def _validate(self):
        """
        Check that the operator basis used in matrix_system is consistent with gauge and hamiltonian.
        """
        self.validate_operator(operator=self.gauge_generator)
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

    def build_null_space_matrix(
        self, additional_constraints: Optional[list[SingleTraceOperator]] = None
    ) -> np.ndarray:
        """
        Builds the null space matrix, K_{ia}.

        Note that K_{ia} K_{ib} = delta_{ab}.
        """
        linear_constraint_matrix = self.build_linear_constraints(additional_constraints)

        if self.verbose:
            debug(
                f"Building the null space matrix. The linear constraint matrix has dimensions {linear_constraint_matrix.shape}"
            )

        self.null_space_matrix = get_null_space_sparse(linear_constraint_matrix)
        self.param_dim_null = self.null_space_matrix.shape[1]
        print(f"Null space dimension (number of parameters) = {self.param_dim_null}")

        if self.checkpoint_path is not None:
            save_npz(
                self.checkpoint_path + "/null_space_matrix.npz", self.null_space_matrix
            )

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

    def load_constraints(self, path):
        if not os.path.exists(path):
            raise ValueError(f"Error, save path {path} does not exist.")

        print(f"Attempting to load from checkpoints, checkpoint_path={path}")

        # load the linear constraints
        if os.path.exists(path + "/linear_constraints_data.pkl"):
            with open(path + "/linear_constraints_data.pkl", "rb") as f:
                loaded_data = pickle.load(f)
            self.linear_constraints = [
                SingleTraceOperator(data=data) for data in loaded_data
            ]
            print("  loaded previously computed linear constraints")

        # load the cyclic quadratic constraints
        if os.path.exists(path + "/cyclic_quadratic.pkl"):
            with open(path + "/cyclic_quadratic.pkl", "rb") as f:
                loaded_data = pickle.load(f)
            self.quadratic_constraints = {
                key: {
                    "lhs": SingleTraceOperator(data=value["lhs"]),
                    "rhs": DoubleTraceOperator(data=value["rhs"]),
                }
                for key, value in loaded_data.items()
            }
            print("  loaded previously computed cyclic constraints")

        # load the null space matrix
        if os.path.exists(path + "/null_space_matrix.npz"):
            self.null_space_matrix = load_npz(path + "/null_space_matrix.npz")
            self.param_dim_null = self.null_space_matrix.shape[1]
            print("  loaded previously computed null space matrix")

        # load the quadratic (numerical) constraints
        if os.path.exists(
            path + "/quadratic_constraints_numerical_linear_term.npz"
        ) and os.path.exists(
            path + "/quadratic_constraints_numerical_quadratic_term.npz"
        ):
            quadratic_constraints_numerical = {}
            quadratic_constraints_numerical["linear"] = load_npz(
                path + "/quadratic_constraints_numerical_linear_term.npz"
            )
            quadratic_constraints_numerical["quadratic"] = load_npz(
                path + "/quadratic_constraints_numerical_quadratic_term.npz"
            )
            self.quadratic_constraints_numerical = quadratic_constraints_numerical
            print("  loaded previously computed quadratic constraints (numerical)")

        # load the bootstrap table
        if os.path.exists(path + "/bootstrap_table_sparse.npz"):
            self.bootstrap_table_sparse = load_npz(path + "/bootstrap_table_sparse.npz")
            print("  loaded previously computed bootstrap table")

    def single_trace_to_coefficient_vector(
        self, st_operator: SingleTraceOperator, return_null_basis: bool = False
    ) -> np.ndarray:
        """
        Map a single trace operator to a vector of the coefficients, v_i.
        Optionally returns the vector in the null basis, u_a = v_i K_{ia}

        Parameters
        ----------
        st_operator : SingleTraceOperator
            The operator

        return_null_basis : bool, optional
            Controls whether the vector is returned in the original basis or the null basis.
            By default False.

        Returns
        -------
        np.ndarray
            The vector.
        """
        self.validate_operator(operator=st_operator)  # validate
        vec = [0] * self.param_dim
        for op, coeff in st_operator:
            idx = self.operator_dict[op]
            vec[idx] = coeff
        vec = np.asarray(vec)

        if not return_null_basis:
            return vec

        # TODO add note about this
        vec = self.conversion_vec * vec

        if np.all(vec.imag == 0):
            return vec.real @ self.null_space_matrix
        elif np.all(vec.real == 0):
            return vec.imag @ self.null_space_matrix
        else:
            raise ValueError("Vector should be purely real or purely imaginary.")

    def double_trace_to_coefficient_matrix(
        self, dt_operator: DoubleTraceOperator
    ) -> coo_matrix:
        """
        Use large-N factorization <tr(O1)tr(O2)> = <tr(O1)><tr(O2)> to
        represent the double-trace operator as a quadratic expression
        of single trace operators, \sum_{ij} M_{ij} v_i v_j.

        The matrix M is returned and stored as a COO matrix.

        Parameters
        ----------
        dt_operator : DoubleTraceOperator
            The double trace operator.

        Returns
        -------
        coo_matrix
            The matrix representation of the factorized double-trace operator.
        """
        index_value_dict = {}
        for (op1, op2), coeff in dt_operator:
            idx1, idx2 = self.operator_dict[op1], self.operator_dict[op2]

            # the odd-degree operators are assumed to be purely imaginary
            # the overall factor of i has been stripped off, but when
            # the double trace operator involves terms like <X><X^3> we need
            # to add back in the i^2 = -1 sign.
            both_odd = (len(op1) % 2 == 1) and (len(op2) % 2 == 1)
            sign = -1 if both_odd else 1

            # symmetrize
            index_value_dict[(idx1, idx2)] = (
                index_value_dict.get((idx1, idx2), 0) + sign * coeff / 2
            )
            index_value_dict[(idx2, idx1)] = (
                index_value_dict.get((idx2, idx1), 0) + sign * coeff / 2
            )

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

        # save path
        d = len(self.matrix_system.operator_basis) // 2  # dimension
        total_constraints_filepath = f"checkpoints/rotational_symmetry_constraints/dim_{d}_L_{self.max_degree_L}.pkl"

        # if symmetry constraints exist, load them
        if os.path.exists(total_constraints_filepath):
            with open(total_constraints_filepath, "rb") as f:
                constraints = pickle.load(f)
            debug(f"Loaded symmetry constraints from file {total_constraints_filepath}")

        # otherwise, generate them
        else:

            total_constraints = []
            counter = 0
            n = len(self.matrix_system.operator_basis)  # here, n = 2 * dim

            # loop over symmetry generators M
            for symmetry_idx, symmetry_generator in enumerate(self.symmetry_generators):

                constraints = []
                constraint_filepath = f"checkpoints/rotational_symmetry_constraints/dim_{d}_L_{self.max_degree_L}_sym_idx_{symmetry_idx}.pkl"

                # initialize a matrix M which will implement the linear action of the generator g
                # M will obey [g, operators_vector] = M operators_vector
                M = np.zeros(shape=(n, n), dtype=np.complex128)
                for i, op in enumerate(self.matrix_system.operator_basis):
                    commutator = self.matrix_system.single_trace_commutator(
                        symmetry_generator, SingleTraceOperator(data={(op): 1})
                    )
                    for op, coeff in commutator:
                        if np.abs(coeff) > tol:
                            j = self.matrix_system.operator_basis.index(op[0])
                            M[i, j] = coeff

                # build the change of variables matrix
                eig_values, old_to_new_variables = np.linalg.eig(M)
                old_to_new_variables = old_to_new_variables.T

                # confirm that the eigenvector relationship holds
                assert np.all(
                    [
                        np.allclose(
                            np.zeros(n),
                            M @ old_to_new_variables[i]
                            - eig_values[i] * old_to_new_variables[i],
                        )
                        for i in range(n)
                    ]
                )

                # build all monomials using the new operators with degree < 2*L
                new_ops_dict = {f"new_op_{i}": i for i in range(n)}
                all_new_operators = {
                    deg: [x for x in product(new_ops_dict.keys(), repeat=deg)]
                    for deg in range(1, 2 * self.max_degree_L + 1)
                }
                all_new_operators = [
                    x for xs in all_new_operators.values() for x in xs
                ]  # flatten

                # loop over all operators in the eigenbasis
                for idx, operator in enumerate(all_new_operators):
                    counter += 1

                    # compute the charge under the symmetry
                    charge = sum(
                        [eig_values[new_ops_dict[basis_op]] for basis_op in operator]
                    )

                    # if the charge is not zero, the resulting operator expectation value must vanish in a symmetric state
                    if np.abs(charge) > tol:

                        # intermediate data structure used to transform operator in eigen-basis to the original basis
                        # operator2 = {
                        #   0: ("X0", a0), ("X1", a1), ..., ("Pi0", b0), ("Pi1", b1), ...
                        #   1: ("X0", c0), ("X1", c1), ..., ("Pi0", d0), ("Pi1", d1), ...
                        #   ...
                        # }
                        # which means that the constraint operator is
                        # (a0 "X0" + ... + b0 "Pi0") x (c0 "X0" + ... d0 "Pi0" + ...) x ...
                        operator2 = {}
                        for i in range(len(operator)):
                            operator2[i] = [
                                (
                                    self.matrix_system.operator_basis[j],
                                    old_to_new_variables[new_ops_dict[operator[i]], j],
                                )
                                for j in range(n)
                            ]

                        # build the constraint single-trace operator using the above intermediate data structure
                        data = {}
                        for indices in list(product(range(n), repeat=len(operator))):
                            op = tuple(
                                [
                                    value[indices[idx]][0]
                                    for idx, value in enumerate(operator2.values())
                                ]
                            )
                            coeff = np.prod(
                                [
                                    value[indices[idx]][1]
                                    for idx, value in enumerate(operator2.values())
                                ]
                            )
                            if np.abs(coeff) > tol:
                                data[op] = data.get(op, 0) + coeff
                        constraint_op = SingleTraceOperator(data=data)

                        # if the constraint contains both real and imaginary terms, they must each hold separately
                        if constraint_op.is_real():
                            constraints.append(constraint_op)
                        else:
                            constraints.append(constraint_op.get_real_part())
                            constraints.append(constraint_op.get_imag_part())

                    if self.verbose:
                        debug(
                            f"Generating symmetry constraints, operator {counter}/{len(all_new_operators) * len(self.symmetry_generators)}, generator {symmetry_idx+1}/{len(self.symmetry_generators)}"
                        )

                # clean them
                constraints = self.clean_constraints(constraints)

                # save them
                with open(constraint_filepath, "wb") as f:
                    pickle.dump(constraints, f)

                # add to the total set of constraints
                total_constraints.extend(constraints)

            # save them
            with open(total_constraints_filepath, "wb") as f:
                pickle.dump(constraints, f)

        return constraints

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
                debug(
                    f"Generating Hamiltonian constraints, operator {op_idx+1}/{len(self.operator_list)}"
                )

        return self.clean_constraints(constraints)

    def generate_gauge_constraints(self) -> list[SingleTraceOperator]:
        """
        Generate the Gauge constraints <tr(G O)>=0 for O a general matrix operator.
        Because G has max degree 2, the gauge constraints for some operators will
        involve operators outside the bootstrap system. These will be discarded.

        Returns
        -------
        list[SingleTraceOperator]
            The list of constraint terms.
        """
        constraints = []
        for op_idx, op in enumerate(self.operator_list):
            constraints.append(
                (self.gauge_generator * MatrixOperator(data={op: 1})).trace()
            )

            if self.verbose:
                debug(
                    f"Generating gauge constraints, operator {op_idx+1}/{len(self.operator_list)}"
                )

        return self.clean_constraints(constraints)

    def generate_odd_degree_vanish_constraints(self) -> list[SingleTraceOperator]:
        """
        Generate the constraints that set all odd-degree basis operators to zero.

        Returns
        -------
        list[SingleTraceOperator]
            The constraints.
        """
        constraints = []
        for op in self.operator_list:
            if len(op) % 2 == 1:
                constraints.append(SingleTraceOperator(data={op: 1}))
        return constraints

    def generate_reality_constraints(self) -> list[SingleTraceOperator]:
        """
        Generate single trace constraints imposed by reality,
            <O^dagger> = <O>*

            NOTE: The current implementation assumes that
                - <O> is real for deg(O) even, so that <O>* = <O>.
                - <O> is imaginary for deg(O) odd, so that <O>* = -<O>.

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
                sign = 1 if len(op) % 2 == 0 else -1
                constraints.append(st_operator - sign * st_operator_dagger)
        return self.clean_constraints(constraints)

    def generate_cyclic_constraint(
        self, op: tuple
    ) -> tuple[SingleTraceOperator, DoubleTraceOperator]:
        """
        Generate the cyclic constraint for a given operator.

        The constraint generally consists of both single- and double-trace terms.
        The former are moved to the LHS of the constraint equation, and the latter
        are moved to the RHS, so that the constraint becomes
            LHS (single-trace operators) = RHS (double-trace operators)

        Parameters
        ----------
        op : tuple
            The operator.

        Returns
        -------
        tuple[SingleTraceOperator, DoubleTraceOperator]
            The single- and double-trace operators.

        Raises
        ------
        ValueError
            _description_
        """
        if len(op) <= 1:
            raise ValueError("Error, expect that the operator have degree >= 1.")

        eq_lhs = SingleTraceOperator(data={op: 1}) - SingleTraceOperator(
            data={op[1:] + (op[0],): 1}
        )

        # rhe RHS corresponds to double trace operators
        eq_rhs = DoubleTraceOperator(data={})
        for k in range(1, len(op)):
            commutator = self.matrix_system.commutation_rules[(op[0], op[k])]
            st_operator_1 = SingleTraceOperator(data={tuple(op[1:k]): 1})
            st_operator_2 = SingleTraceOperator(data={tuple(op[k + 1 :]): 1})

            # If the double trace term involves <tr(1)> simplify and add to the linear, LHS
            if st_operator_1 == SingleTraceOperator(data={(): 1}):
                eq_lhs -= commutator * st_operator_2
            elif st_operator_2 == SingleTraceOperator(data={(): 1}):
                eq_lhs -= commutator * st_operator_1
            else:
                eq_rhs += commutator * (st_operator_1 * st_operator_2)

        return eq_lhs, eq_rhs

    def generate_cyclic_constraints(
        self,
    ) -> dict[int, dict[str, SingleTraceOperator | DoubleTraceOperator]]:
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

                eq_lhs, eq_rhs = self.generate_cyclic_constraint(op=op)

                """
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
                """

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
                debug(
                    f"Generating cyclic constraints, operator {op_idx+1}/{len(self.operator_list)}"
                )

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
            linear_constraints.extend(symmetry_constraints)

        # reality constraints
        reality_constraints = self.generate_reality_constraints()
        print(f"Generated {len(reality_constraints)} reality constraints")
        linear_constraints.extend(reality_constraints)

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
        # linear_constraints.extend(
        #    [self.matrix_system.hermitian_conjugate(op) for op in linear_constraints]
        # )

        # save the constraints
        if self.checkpoint_path is not None:
            with open(self.checkpoint_path + "/linear_constraints_data.pkl", "wb") as f:
                pickle.dump([constraint.data for constraint in linear_constraints], f)
            with open(self.checkpoint_path + "/cyclic_quadratic.pkl", "wb") as f:
                cyclic_data_dict = {}
                for key, value in cyclic_quadratic.items():
                    cyclic_data_dict[key] = {
                        "lhs": value["lhs"].data,
                        "rhs": value["rhs"].data,
                    }
                pickle.dump(cyclic_data_dict, f)

        return linear_constraints, cyclic_quadratic

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

        Returns
        -------
        coo_matrix
            The set of linear constraints.
        """
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
        for idx_constraint, st_operator in enumerate(self.linear_constraints):
            vec = self.conversion_vec * self.single_trace_to_coefficient_vector(
                st_operator=st_operator
            )

            # vec is purely real
            if np.all(vec.imag == 0):
                for op_str, coeff in st_operator:
                    idx_operator = self.operator_dict[op_str]
                    index_value_dict[(idx_constraint, idx_operator)] = np.real(
                        vec[idx_operator]
                    )
            # vec is purely imag
            elif np.all(vec.real == 0):
                for op_str, coeff in st_operator:
                    idx_operator = self.operator_dict[op_str]
                    index_value_dict[(idx_constraint, idx_operator)] = np.imag(
                        vec[idx_operator]
                    )
            else:
                raise ValueError(
                    "Error, coefficient vector not purely real or purely imaginary."
                )

        # return the constraint matrix
        return create_sparse_matrix_from_dict(
            index_value_dict=index_value_dict,
            matrix_shape=(len(self.linear_constraints), len(self.operator_list)),
        )

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
            "lhs": SingleTraceOperator(data={(): 1}),
            "rhs": DoubleTraceOperator(data={((), ()): 1}),
        }
        quadratic_constraints[None] = normalization_constraint

        # loop over constraints
        print("Building quadratic constraints...")
        for constraint_idx, (operator_idx, constraint) in enumerate(
            quadratic_constraints.items()
        ):

            if self.verbose:
                debug(
                    f"Generating quadratic constraints, operator {constraint_idx+1}/{len(quadratic_constraints)}"
                )
            # debug(f"Memory usage: {psutil.Process().memory_info().rss / 1024 ** 2}")

            lhs = constraint["lhs"]
            rhs = constraint["rhs"]

            # initialize the quadratic constraint matrix
            # linear_constraint_vector = self.single_trace_to_coefficient_vector(lhs)
            linear_constraint_vector = self.single_trace_to_coefficient_vector(
                st_operator=lhs, return_null_basis=True
            )
            quadratic_matrix = self.double_trace_to_coefficient_matrix(rhs)

            # transform to null basis
            # the minus sign is very important: (-RHS + LHS = 0)
            # linear_constraint_vector = linear_constraint_vector @ null_space_matrix
            quadratic_matrix = (
                -null_space_matrix.T @ quadratic_matrix @ null_space_matrix
            )

            # reshape the (d,d) matrix to a (1,d^2) matrix
            quadratic_matrix = quadratic_matrix.reshape((1, self.param_dim_null**2))

            linear_is_zero = np.max(np.abs(linear_constraint_vector)) < self.tol
            quadratic_is_zero = np.max(np.abs(quadratic_matrix)) < self.tol

            if self.simplify_quadratic:
                if not quadratic_is_zero:
                    linear_terms.append(csr_matrix(linear_constraint_vector))
                    quadratic_terms.append(quadratic_matrix)
                elif not linear_is_zero:
                    additional_constraints.append(lhs)
            else:
                if not quadratic_is_zero or not linear_is_zero:
                    linear_terms.append(csr_matrix(linear_constraint_vector))
                    quadratic_terms.append(quadratic_matrix)

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
        stacked_matrix = hstack([quadratic_terms, linear_terms])
        stacked_matrix = get_row_space_sparse(stacked_matrix)
        num_constraints = stacked_matrix.shape[0]
        linear_terms = stacked_matrix[:, self.param_dim_null**2 :]
        quadratic_terms = stacked_matrix[:, : self.param_dim_null**2]
        print(f"Number of quadratic constraints after row reduction: {num_constraints}")

        if self.checkpoint_path is not None:
            save_npz(
                self.checkpoint_path
                + "/quadratic_constraints_numerical_linear_term.npz",
                linear_terms,
            )
            save_npz(
                self.checkpoint_path
                + "/quadratic_constraints_numerical_quadratic_term.npz",
                quadratic_terms,
            )

        self.quadratic_constraints_numerical = {
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
        if self.verbose:
            debug("Cleaning constraints...")

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
        print("Building the bootstrap table...")
        if self.null_space_matrix is None:
            raise ValueError("Error, null space matrix has not yet been built.")

        bootstrap_dict = {}
        for idx1, op_str1 in enumerate(self.bootstrap_basis_list):
            op_str1 = op_str1[::-1]  # take the h.c. by reversing the elements
            for idx2, op_str2 in enumerate(self.bootstrap_basis_list):

                # tally up number of anti-hermitian operators, and add (-1) factor if odd
                num_antihermitian_ops = sum(
                    [not self.matrix_system.hermitian_dict[term] for term in op_str1]
                )
                sign = (-1) ** num_antihermitian_ops

                # add a factor of i if the total degree is odd
                degree = len(op_str1 + op_str2)
                i_factor = 1 if (degree % 2 == 0) else 1j

                # build the index map for the matrix entry in question
                index_map = self.operator_dict[op_str1 + op_str2]
                for k in range(self.null_space_matrix.shape[1]):
                    x = sign * i_factor * self.null_space_matrix[index_map, k]
                    if np.abs(x) > self.tol:
                        bootstrap_dict[(idx1, idx2, k)] = x

        # map to a sparse array
        bootstrap_table = np.zeros(
            (self.bootstrap_matrix_dim, self.bootstrap_matrix_dim, self.param_dim_null),
        )
        for (i, j, k), value in bootstrap_dict.items():
            bootstrap_table[i, j, k] = value

        self.bootstrap_table_sparse = csr_matrix(
            bootstrap_table.reshape(
                bootstrap_table.shape[0] * bootstrap_table.shape[1],
                bootstrap_table.shape[2],
            )
        )

        # save
        if self.checkpoint_path is not None:
            save_npz(
                self.checkpoint_path + "/bootstrap_table_sparse.npz",
                self.bootstrap_table_sparse,
            )

    def get_bootstrap_matrix(self, param: np.ndarray) -> np.ndarray:
        """
        Build the bootstrap matrix for a given null parameter vector.

        Parameters
        ----------
        param : np.ndarray
            The parameter vector.

        Returns
        -------
        np.ndarray
            The bootstrap matrix.

        Raises
        ------
        ValueError
            Raise an error if the bootstrap matrix is not Hermitian.
        """

        dim = self.bootstrap_matrix_dim
        if self.bootstrap_table_sparse is None:
            self.buid_bootstrap_table()

        bootstrap_matrix = np.reshape(
            self.bootstrap_table_sparse.dot(param), (dim, dim)
        )

        # verify that matrix is Hermitian
        if not ishermitian(bootstrap_matrix):
            raise ValueError(f"Bootstrap matrix is not symmetric.")

        return bootstrap_matrix

    def get_operator_expectation_value(
        self, st_operator: SingleTraceOperator, param: np.ndarray
    ) -> float:
        """
        Return the expectation value of a single-trace operator given
        a null parameter vector.

        TODO revisit this for complex models

        Parameters
        ----------
        st_operator : SingleTraceOperator
            The single-trace operator.
        param : np.ndarray
            The null parameter vector.

        Returns
        -------
        float
            The expectation value.
        """
        vec = self.single_trace_to_coefficient_vector(
            st_operator=st_operator, return_null_basis=True
        )
        op_expectation_value = vec @ param
        return op_expectation_value
