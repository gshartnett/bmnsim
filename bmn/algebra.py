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
            self.max_degree = 0
        else:
            self.max_degree = max(self.degrees)

    def __iter__(self):
        for key, value in self.data.items():
            yield key, value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(operators={self.operators!r}, coeff={self.coeffs!r})"

    def __str__(self) -> str:
        x = ""
        for idx, (coeff, op) in enumerate(zip(self.coeffs, self.operators)):
            x += f"{coeff}" + f" {op}"
            if idx != len(self.operators) - 1:
                x += " + "
        return x

    def copy(self):
        return self.__class__(data={k: v for k, v in self})

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot add {type(other)} and {self.__class__.__name__}")
        new_data = self.data.copy()
        for op, coeff in other:
            new_data[op] = new_data.get(op, 0) + coeff
        return self.__class__(data=new_data)

    def __sub__(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            raise ValueError(
                f"Cannot subtract {type(other)} and {self.__class__.__name__}"
            )
        new_data = self.data.copy()
        for op, coeff in other:
            new_data[op] = new_data.get(op, 0) - coeff
        return self.__class__(data=new_data)

    def __rmul__(self, other: Number):
        if isinstance(other, Number):
            new_data = {op: other * coeff for op, coeff in self}
            return self.__class__(data=new_data)
        else:
            raise ValueError("Warning, right multiplication only valid for numbers")

    def __mul__(self, other: Union[Self, Number]):
        if not isinstance(other, (Number, self.__class__)):
            raise ValueError(f"Cannot multiply {type(other)} and {self.__class__}")
        if isinstance(other, Number):
            return self.__rmul__(other)
        new_data = {}
        for op1, coeff1 in self:
            for op2, coeff2 in other:
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
        return self.data == {k: v for k, v in other}

    def __len__(self) -> int:
        return len(self.data)

    def trace(self):
        return SingleTraceOperator(data={k: v for k, v in self})


class SingleTraceOperator(MatrixOperator):
    """
    Single trace operator.
    TODO consider adding an __iter__/making it a generator
    """

    def __str__(self) -> str:
        x = ""
        for idx, (coeff, op) in enumerate(zip(self.coeffs, self.operators)):
            x += f"{coeff}" + f" tr {op}"
            if idx != len(self.operators) - 1:
                x += " + "
        return x

    def __mul__(self, other: Number):
        if not isinstance(other, Number):
            raise ValueError(f"Cannot multiply {type(other)} and {self.__class__}")
        if isinstance(other, Number):
            return self.__rmul__(other)
        '''
        # loop over all terms
        for op1, coeff1 in self.data.items():
            for op2, coeff2 in self.data.items():

        # special case when either self or other is proportional to identity
        if self.max_degree == 0:
            # zero operator
            if self.coeffs == []:
                return 0 * SingleTraceOperator(data = {k: v for k, v in other.data.items()})
            return self.coeffs[0] * SingleTraceOperator(data = {k: v for k, v in other.data.items()})
        if other.max_degree == 0:
            if other.coeffs == []:
                return 0 * SingleTraceOperator(data = {k: v for k, v in self.data.items()})
            return other.coeffs[0] * SingleTraceOperator(data = {k: v for k, v in self.data.items()})

        return DoubleTraceOperator(
            op1=SingleTraceOperator(data={k: v for k, v in self.data.items()}),
            op2=SingleTraceOperator(data={k: v for k, v in other.data.items()}),
        )
        '''

'''
class DoubleTraceOperator:
    def __init__(self, operator1: SingleTraceOperator, operator2: SingleTraceOperator):

        # zero
        if len(operator1) * len(operator2) == 0:
            return SingleTraceOperator(data={():0})

        #self.data = {}
        #for op1, coeff1 in operator1.data.items():
        #    for op2, coeff2 in operator2.data.items():
        #        self.data[(op1, op2)] = coeff1 * coeff2
        if not (len(op1) == 1 and len(op2) == 1):
            raise ValueError("Each SingleTraceOperator must have deg <=1.")

        self.coeff = op1.coeffs[0] * op2.coeffs[0]
        self.op1 = op1.operators[0]
        self.op2 = op2.operators[0]

    def __repr__(self) -> str:
         return f"{self.__class__.__name__}(op1={self.op1!r}, op2={self.op2!r}, coeff={self.coeff!r})"

    def __str__(self) -> str:
        x = "("
        for idx, (coeff, op) in enumerate(
            zip(self.data[0].coeffs, self.data[0].operators)
        ):
            x += f"{coeff}" + f" tr {op}"
            if idx != len(self.data[0].operators) - 1:
                x += " + "
            else:
                x += ")"

        x += " * ("

        for idx, (coeff, op) in enumerate(
            zip(self.data[1].coeffs, self.data[1].operators)
        ):
            x += f"{coeff}" + f" tr {op}"
            if idx != len(self.data[1].operators) - 1:
                x += " + "
            else:
                x += ")"
        return x
'''

class MatrixSystem:
    """
    Class for doing algebra.
    """

    def __init__(self, operator_basis: list[str], commutation_rules_concise: int, hermitian_dict: dict[str, str]):
        self.operator_basis = operator_basis

        print('Assuming all operators are either Hermitian or anti-Hermitna.')
        #self.hermitian_dict = {op_str: ('X' in op_str) for op_str in self.operator_basis}
        #self.hermitian_dict = {op_str: True for op_str in self.operator_basis}
        self.hermitian_dict = hermitian_dict
        if set(hermitian_dict.keys()) != set(operator_basis):
            raise ValueError("Warning, keys of hermitian_dict must match operator_basis elements.")
        self.commutation_rules = self.build_commutation_rules(commutation_rules_concise)


    def hermitian_conjugate(self, operator: MatrixOperator) -> Self:
        # assumes operator basis is Hermitian or anti-Hermitian
        data = {}
        for op, coeff in operator:
            reversed_op = op[::-1]
            num_antihermitian = sum(1 * (not self.hermitian_dict[op_str]) for op_str in op)
            data[reversed_op] = (-1)**num_antihermitian * np.conjugate(coeff)
        return operator.__class__(data=data)
        '''
        return self.__class__(
            data={
                op[::-1]: np.conjugate(coeff)
                for op, coeff in self
            }
        )
        '''

    def build_commutation_rules(self, commutation_rules_concise):
        """
        Expand the supplied concise commutation rules to cover all
        possibilities, i.e.
        [P1, X1], [X1, P1], [X2, P1], etc
        """
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
        self, st_operator1: SingleTraceOperator, st_operator2: SingleTraceOperator
    ) -> SingleTraceOperator:
        """
        Take the commutator of two single trace operators.
        """
        if not (
            isinstance(st_operator1, SingleTraceOperator)
            and isinstance(st_operator2, SingleTraceOperator)
        ):
            raise ValueError("Arguments must be single trace operators.")

        # initialize data of commutator
        new_data = {}

        # loop over the terms in each single trace operator
        for op1, coeff1 in st_operator1:
            for op2, coeff2 in st_operator2:

                # loop over the variables in each term
                for variable1_idx, variable1 in enumerate(op1):
                    for variable2_idx, variable2 in enumerate(op2):

                        # TODO revisit this relation to better understand it/derive it
                        new_coeff = (
                            coeff1
                            * coeff2
                            * self.commutation_rules[(variable1, variable2)]
                        )
                        new_term = (
                            op2[:variable2_idx]
                            + op1[variable1_idx + 1 :]
                            + op1[:variable1_idx]
                            + op2[variable2_idx + 1 :]
                        )
                        new_data[new_term] = new_data.get(new_term, 0) + new_coeff

        return SingleTraceOperator(data=new_data)
