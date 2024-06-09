from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)


def test_instantiate_single_trace_operator():
    """
    Make sure that the single trace operator instantiation is
    insensitive to whether the input is a tuple or a string
    for the special case of a degree 1 term.
    """
    op1 = MatrixOperator(data={"P2": 0.5})
    op2 = MatrixOperator(data={("P2",): 0.5})
    assert op1 == op2


def test_single_trace_commutator_onematrix():
    """
    Test the Hamiltonian constraint <[H,O]> = 0 for the case
    of O = tr(XP) and H given  by the single-matrix QM model
    studied in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.041601.
    This is used to derive the constraint Eq. (8)
    """
    matrix_system = MatrixSystem(
        operator_basis=["X", "P"],
        commutation_rules_concise={
            ("P", "X"): -1j,
        },
        hermitian_dict={"P": True, "X": True},
    )
    OP1 = SingleTraceOperator(data={("X", "P"): 1})
    OP2 = SingleTraceOperator(
        data={("P", "P"): 1, ("X", "X"): 1, ("X", "X", "X", "X"): 7}
    )
    assert matrix_system.single_trace_commutator(OP1, OP2) == SingleTraceOperator(
        data={("P", "P"): 2j, ("X", "X"): -2j, ("X", "X", "X", "X"): -4 * 7j}
    )


def test_zero_single_trace_operator():
    """
    Test edge cases involving the zero operator
    """
    zero = SingleTraceOperator(data={(): 0})

    assert len(zero) == 0

    # 0 * <tr(O)>
    assert SingleTraceOperator(data={("P", "P"): 0}) == zero

    # alpha * zero
    # assert SingleTraceOperator(data={(): 3}) * SingleTraceOperator(data={("P", "P"): 0}) == zero

    # 0 * zero
    # assert SingleTraceOperator(data={(): 0}) * SingleTraceOperator(data={("P", "P"): 0}) == zero

    # zero * zero
    # assert SingleTraceOperator(data={("X"): 0}) * SingleTraceOperator(data={("P", "P"): 0}) == zero


def test_zero_double_trace_operator():
    """
    Test edge cases involving the zero operator
    """
    zero = SingleTraceOperator(data={(): 0})
    assert zero * zero == zero


def test_single_trace_component_of_double_trace():
    op = SingleTraceOperator(
        data={("P", "P"): 1, ("X", "X"): 1, ("X", "X", "X", "X"): 7}
    )
    one = SingleTraceOperator(data={(): 1})
    assert (one * op).get_single_trace_component() == op
    assert (op * one).get_single_trace_component() == op
