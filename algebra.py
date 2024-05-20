import numpy as np
import sympy as sp
from typing import Union, Self
from numbers import Number


class MatrixOperator():
    '''
    Class for un-traced matrix operators.

    TODO
    consider allowing for symbolic coefficients
    build some unit tests to check the basic operation
    '''

    #def __init__(self, operators: Union[str, list[str]], coeffs: Union[Number, list[Number]], tol=1e-10):
    def __init__(self, data: dict[tuple: list[Number]], tol: float=1e-10):
        #if not isinstance(operators, list):
        #    operators = [operators]
        #if not isinstance(coeffs, list):
        #    coeffs = [coeffs]
        self.tol = tol
        for op in data.keys():
            if not isinstance(op, tuple):
                raise ValueError("All operators must be tuples of strings. For degree-1 operators, use e.g. '(X,)'")
        self.data = {op: coeff for op, coeff in data.items() if np.abs(coeff) > self.tol}
        self.operators = list(self.data.keys())
        self.coeffs = list(self.data.values())
        self._validate()
        self.degrees = [len(op) for op in self.operators]
        self.max_degree = max(self.degrees)

    def _validate(self):
        if len(self.coeffs) != len(self.operators):
            raise ValueError("Warning, unequal numbers of terms and coefficients were specified.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(operators={self.operators!r}, coeff={self.coeffs!r})"

    def __str__(self) -> str:
        x = ''
        for idx, (coeff, op) in enumerate(zip(self.coeffs, self.operators)):
            x += f'{coeff}' + f' {op}'
            if idx != len(self.operators) - 1:
                x += ' + '
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
            raise ValueError(f"Cannot subtract {type(other)} and {self.__class__.__name__}")
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
            raise ValueError("Warning, exponentiation only defined for positive integer powers.")
        if power < 0:
            raise ValueError("Warning, exponentiation only defined for positive integer powers.")
        if power == 0:
            raise NotImplementedError
        if power == 1:
            return self
        else:
            return self * self.__pow__(power-1)


class SingleTraceOperator(MatrixOperator):

    def __str__(self) -> str:
        x = ''
        for idx, (coeff, op) in enumerate(zip(self.coeffs, self.operators)):
            x += f'{coeff}' + f' tr {op}'
            if idx != len(self.operators) - 1:
                x += ' + '
        return x


class MatrixSystem:
    '''
    Class for doing algebra.
    '''
    def __init__(self, operator_basis: list[str], commutation_rules_concise: int):
        self.operator_basis = operator_basis
        self.commutation_rules = self.build_commutation_rules(commutation_rules_concise)
        self._validate()

    def build_commutation_rules(self, commutation_rules_concise):
        # expand the supplied concise commutation rules to cover all possibilities
        commutation_rules = {}
        for op_str_1 in self.operator_basis:
            for op_str_2 in self.operator_basis:
                if commutation_rules_concise.get((op_str_1, op_str_2), None) is not None:
                    commutation_rules[(op_str_1, op_str_2)] = commutation_rules_concise[(op_str_1, op_str_2)]
                    commutation_rules[(op_str_2, op_str_1)] = -commutation_rules_concise[(op_str_1, op_str_2)]
                elif commutation_rules_concise.get((op_str_2, op_str_1), None) is not None:
                    commutation_rules[(op_str_2, op_str_1)] = commutation_rules_concise[(op_str_2, op_str_1)]
                    commutation_rules[(op_str_1, op_str_2)] = -commutation_rules_concise[(op_str_2, op_str_1)]
                else:
                    commutation_rules[(op_str_1, op_str_2)] = 0
                    commutation_rules[(op_str_2, op_str_1)] = 0
        return commutation_rules

    def _get_single_trace_commutator_monomial(self, op1: SingleTraceOperator, op2: SingleTraceOperator):
        '''
        commutator restricted to monomial terms
        NOTE probably don't need a separate monomial class, but it's an option
        '''
        if not (isinstance(op1, SingleTraceOperator) and isinstance(op2, SingleTraceOperator)):
            raise ValueError("Arguments must be single trace operators.")
        if not (len(op1.data) == 1 and len(op2.data) == 1):
            raise ValueError("Arguments must be monomial operators, i.e. operators consisting of only a single term.")

        # extract the list of terms
        # in the case of 1 term only, op1.operators is a list of strs, like ['X1']
        # in the case of multiple terms, op1.operators is a list of tuples, like [('X1'), ('P1')]
        '''
        OLD, used before enforcing ops are tuples

        if isinstance(op1.operators[0], str):
            op1_list = op1.operators
        else:
            op1_list = list(op1.operators[0])

        if isinstance(op2.operators[0], str):
            op2_list = op2.operators
        else:
            op2_list = list(op2.operators[0])
        '''
        op1_list = op1.operators
        op2_list = op2.operators
        coeff1 = op1.coeffs[0]
        coeff2 = op2.coeffs[0]

        print(op1_list, op2_list)
        new_data = {}
        # consider all possible pairs of operators from each argument
        for idx1 in range(len(op1_list)):
            for idx2 in range(len(op2_list)):
                commutator = self.commutation_rules[(op1_list[idx1][0], op2_list[idx2][0])]
                op1_left, op1_right = split_list(op1_list, idx1)
                op2_left, op2_right = split_list(op2_list, idx2)
                new_data[tuple(op1_left + op2_left + op1_right + op2_right)] = coeff1 * coeff2 * commutator

        # how to handle case where they commute or return a constant...
        return SingleTraceOperator(data=new_data)

    def get_single_trace_commutator(op1: SingleTraceOperator, op2: SingleTraceOperator):
        pass

    def _validate(self):
        pass
