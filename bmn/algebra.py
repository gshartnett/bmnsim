import numpy as np
from typing import Union, Self
from numbers import Number


class MatrixOperator:
    """
    Class for un-traced matrix operators.

    TODO
    What about case of constant or zero operator?
    build some unit tests to check the basic operation
    """

    def __init__(self, data: dict[tuple : list[Number]], tol: float = 1e-10):
        self.tol = tol
        self.data = {}
        for op, coeff in data.items():
            if np.abs(coeff) > self.tol:
                if isinstance(op, tuple):
                    self.data[op] = coeff
                elif isinstance(op, str):
                    self.data[(op,)] = coeff
                else:
                    raise ValueError(
                        "All operators must be tuples of strings, e.g. (X, Y, P)."
                    )
        self.operators = list(self.data.keys())
        self.coeffs = list(self.data.values())
        self.degrees = [len(op) for op in self.operators]
        if self.degrees == []:
            self.degree = 0
        else:
            self.max_degree = max(self.degrees)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(operators={self.operators!r}, coeff={self.coeffs!r})"

    def __str__(self) -> str:
        x = ""
        for idx, (coeff, op) in enumerate(zip(self.coeffs, self.operators)):
            x += f"{coeff}" + f" {op}"
            if idx != len(self.operators) - 1:
                x += " + "
        return x

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot add {type(other)} and {self.__class__.__name__}")
        new_data = self.data.copy()
        for op, coeff in other.data.items():
            new_data[op] = new_data.get(op, 0) + coeff
        return self.__class__(data=new_data)

    def __sub__(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            raise ValueError(
                f"Cannot subtract {type(other)} and {self.__class__.__name__}"
            )
        new_data = self.data.copy()
        for op, coeff in other.data.items():
            new_data[op] = new_data.get(op, 0) - coeff
        return self.__class__(data=new_data)

    def __rmul__(self, other: Number):
        if isinstance(other, Number):
            new_data = {op: other * coeff for op, coeff in self.data.items()}
            return self.__class__(data=new_data)
        else:
            raise ValueError("Warning, right multiplication only valid for numbers")

    def __mul__(self, other: Union[Self, Number]):
        if not isinstance(other, (Number, self.__class__)):
            raise ValueError(f"Cannot multiply {type(other)} and {self.__class__}")
        if isinstance(other, Number):
            return self.__rmul__(other)
        new_data = {}
        for op1, coeff1 in self.data.items():
            for op2, coeff2 in other.data.items():
                new_data[op1 + op2] = new_data.get(op1 + op2, 0) + coeff1 * coeff2
        return self.__class__(data=new_data)

    def __pow__(self, power: int):
        if not isinstance(power, int):
            raise ValueError(
                "Warning, exponentiation only defined for positive integer powers."
            )
        if power < 0:
            raise ValueError(
                "Warning, exponentiation only defined for positive integer powers."
            )
        if power == 0:
            raise NotImplementedError
        if power == 1:
            return self
        else:
            return self * self.__pow__(power - 1)

    def __eq__(self, other: Self) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.data == other.data

    def trace(self):
        return SingleTraceOperator(data=self.data)


class SingleTraceOperator(MatrixOperator):
    """
    Single trace operator.
    """

    def __str__(self) -> str:
        x = ""
        for idx, (coeff, op) in enumerate(zip(self.coeffs, self.operators)):
            x += f"{coeff}" + f" tr {op}"
            if idx != len(self.operators) - 1:
                x += " + "
        return x


class MatrixSystem:
    """
    Class for doing algebra.
    """

    def __init__(self, operator_basis: list[str], commutation_rules_concise: int):
        self.operator_basis = operator_basis
        self.commutation_rules = self.build_commutation_rules(commutation_rules_concise)

    def build_commutation_rules(self, commutation_rules_concise):
        '''
        Expand the supplied concise commutation rules to cover all
        possibilities, i.e.
        [P1, X1], [X1, P1], [X2, P1], etc
        '''
        commutation_rules = {}
        for op_str_1 in self.operator_basis:
            for op_str_2 in self.operator_basis:
                if (
                    commutation_rules_concise.get((op_str_1, op_str_2), None)
                    is not None
                ):
                    commutation_rules[(op_str_1, op_str_2)] = commutation_rules_concise[
                        (op_str_1, op_str_2)
                    ]
                    commutation_rules[(op_str_2, op_str_1)] = (
                        -commutation_rules_concise[(op_str_1, op_str_2)]
                    )
                elif (
                    commutation_rules_concise.get((op_str_2, op_str_1), None)
                    is not None
                ):
                    commutation_rules[(op_str_2, op_str_1)] = commutation_rules_concise[
                        (op_str_2, op_str_1)
                    ]
                    commutation_rules[(op_str_1, op_str_2)] = (
                        -commutation_rules_concise[(op_str_2, op_str_1)]
                    )
                else:
                    commutation_rules[(op_str_1, op_str_2)] = 0
                    commutation_rules[(op_str_2, op_str_1)] = 0
        return commutation_rules

    def single_trace_commutator(
        self, op1: SingleTraceOperator, op2: SingleTraceOperator
    ) -> SingleTraceOperator:
        '''
        Take the commutator of two single trace operators.
        '''
        if not (
            isinstance(op1, SingleTraceOperator)
            and isinstance(op2, SingleTraceOperator)
        ):
            raise ValueError("Arguments must be single trace operators.")

        # initialize data of commutator
        new_data = {}

        # loop over the terms in each single trace operator
        for op1_term, coeff1 in op1.data.items():
            for op2_term, coeff2 in op2.data.items():

                # loop over the variables in each term
                for variable1_idx, variable1 in enumerate(op1_term):
                    for variable2_idx, variable2 in enumerate(op2_term):

                        # TODO revisit this relation to better understand it/derive it
                        new_coeff = (
                            coeff1
                            * coeff2
                            * self.commutation_rules[(variable1, variable2)]
                        )
                        new_term = (
                            op2_term[:variable2_idx]
                            + op1_term[variable1_idx + 1 :]
                            + op1_term[:variable1_idx]
                            + op2_term[variable2_idx + 1 :]
                        )
                        new_data[new_term] = new_data.get(new_term, 0) + new_coeff

        return SingleTraceOperator(data=new_data)
