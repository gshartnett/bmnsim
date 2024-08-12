'''
TODO
- consider consolidating BFSS and ThreeMatrixModel
- consider consolidating 2- and 3- Matrix Models
'''
from bmn.algebra import MatrixSystem, SingleTraceOperator, MatrixOperator

class MatrixModel():
    def __init__(self, couplings):
        self.couplings = couplings
        self.build_matrix_system()
        self.build_gauge_generator()
        self.build_hamiltonian()
        self.build_operators_to_track()

    def build_matrix_system():
        raise NotImplementedError()

    def build_gauge_generator():
        raise NotImplementedError()

    def build_hamiltonian():
        raise NotImplementedError()

    def build_operators_to_track():
        raise NotImplementedError()


class OneMatrix(MatrixModel):
    def __init__(self, couplings):
        super().__init__(couplings)
        self.symmetry_generators = None

    def build_matrix_system(self):
        self.matrix_system = MatrixSystem(
            operator_basis=["X", "Pi"],
            commutation_rules_concise={
                ("Pi", "X"): 1,  # use Pi' = i P to ensure reality
                },
                hermitian_dict={"Pi": False, "X": True},
                )

    def build_gauge_generator(self):
        self.gauge_generator = MatrixOperator(data={("X", "Pi"): 1, ("Pi", "X"): -1, (): 1})

    def build_hamiltonian(self):
        self.hamiltonian = SingleTraceOperator(
            data={
                ("Pi", "Pi"): -0.5,
                ("X", "X"): 0.5 * self.couplings["g2"],
                ("X", "X", "X", "X"): self.couplings["g4"] / 4,
                ("X", "X", "X", "X", "X", "X"): self.couplings["g6"] / 6,
                })

    def build_operators_to_track(self):
        self.operators_to_track = {
            "tr(1)": SingleTraceOperator(data={(): 1}),
            "energy": self.hamiltonian,
            "x_2": SingleTraceOperator(data={("X", "X"): 1}),
            "x_4": SingleTraceOperator(data={("X", "X", "X", "X"): 1}),
            "p_2": SingleTraceOperator(data={("Pi", "Pi"): -1}),
            "p_4": SingleTraceOperator(data={("Pi", "Pi", "Pi", "Pi"): 1}),
        }


class TwoMatrix(MatrixModel):
    def __init__(self, couplings):
        super().__init__(couplings)
        self.build_symmetry_generators()

    def build_matrix_system(self):
        self.matrix_system = MatrixSystem(
            operator_basis=["X0", "Pi0", "X1", "Pi1"],
            commutation_rules_concise={
                ("Pi0", "X0"): 1,
                ("Pi1", "X1"): 1,
                },
                hermitian_dict={"Pi0": False, "X0": True, "Pi1": False, "X1": True},
                )

    def build_gauge_generator(self):
        self.gauge_generator = MatrixOperator(
            data={
                ("X0", "Pi0"): 1,
                ("Pi0", "X0"): -1,
                ("X1", "Pi1"): 1,
                ("Pi1", "X1"): -1,
                (): 2,
                }
                )

    def build_hamiltonian(self):
        self.hamiltonian = SingleTraceOperator(
            data={
                ("Pi0", "Pi0"): -1 / 2,
                ("Pi1", "Pi1"): -1 / 2,
                ("X0", "X0"): self.couplings["g2"] / 2,
                ("X1", "X1"): self.couplings["g2"] / 2,
                ("X0", "X1", "X0", "X1"): -self.couplings["g4"] / 4,
                ("X1", "X0", "X1", "X0"): -self.couplings["g4"] / 4,
                ("X0", "X1", "X1", "X0"): self.couplings["g4"] / 4,
                ("X1", "X0", "X0", "X1"): self.couplings["g4"] / 4,
                }
                )

    def build_operators_to_track(self):
        self.operators_to_track = {
            "tr(1)": SingleTraceOperator(data={(): 1}),
            "energy": self.hamiltonian,
            "x_2": SingleTraceOperator(data={("X0", "X0"): 1, ("X1", "X1"): 1}),
            "x_4": SingleTraceOperator(data={("X0", "X0", "X0", "X0"): 1, ("X1", "X1", "X1", "X1"): 1}),
            "p_2": SingleTraceOperator(data={("Pi0", "Pi0"): -1, ("Pi1", "Pi1"): -1}),
            "p_4": SingleTraceOperator(data={("Pi0", "Pi0", "Pi0", "Pi0"): 1, ("Pi1", "Pi1", "Pi1", "Pi1"): 1}),
            "neg_commutator_squared": SingleTraceOperator(data={
                ("X0", "X1", "X0", "X1"): -1,
                ("X1", "X0", "X1", "X0"): 1,
                ("X0", "X1", "X1", "X0"): 1,
                ("X1", "X0", "X0", "X1"): 1,
                }
                ),
                }

    def build_symmetry_generators(self):
        self.symmetry_generators = [SingleTraceOperator(data={("X0", "Pi1"): 1, ("X1", "Pi0"): -1})]


class ThreeMatrix(MatrixModel):
    def __init__(self, couplings):
        super().__init__(couplings)
        self.build_symmetry_generators()

    def build_matrix_system(self):
        self.matrix_system = MatrixSystem(
            operator_basis=["X0", "X1", "X2", "Pi0", "Pi1", "Pi2"],
            commutation_rules_concise={
                ("Pi0", "X0"): 1,
                ("Pi1", "X1"): 1,
                ("Pi2", "X2"): 1,
                },
                hermitian_dict={"Pi0": False, "X0": True, "Pi1": False, "X1": True, "Pi2": False, "X2": True},
                )

    def build_gauge_generator(self):
        self.gauge_generator = MatrixOperator(
            data={
                ("X0", "Pi0"): 1,
                ("Pi0", "X0"): -1,
                ("X1", "Pi1"): 1,
                ("Pi1", "X1"): -1,
                ("X2", "Pi2"): 1,
                ("Pi2", "X2"): -1,
                (): 3,
                }
                )

    def build_hamiltonian(self):
        self.hamiltonian = SingleTraceOperator(
            data={
                ("Pi0", "Pi0"): -0.5,
                ("Pi1", "Pi1"): -0.5,
                ("Pi2", "Pi2"): -0.5,
                # mass term
                ("X0", "X0"): self.couplings["g2"] / 2,
                ("X1", "X1"): self.couplings["g2"] / 2,
                ("X2", "X2"): self.couplings["g2"] / 2,
                # quartic term (XY)
                ("X0", "X1", "X0", "X1"): -self.couplings["g4"] / 4,
                ("X1", "X0", "X1", "X0"): -self.couplings["g4"] / 4,
                ("X0", "X1", "X1", "X0"): self.couplings["g4"] / 4,
                ("X1", "X0", "X0", "X1"): self.couplings["g4"] / 4,
                # quartic term (XZ)
                ("X0", "X2", "X0", "X2"): -self.couplings["g4"] / 4,
                ("X2", "X0", "X2", "X0"): -self.couplings["g4"] / 4,
                ("X0", "X2", "X2", "X0"): self.couplings["g4"] / 4,
                ("X2", "X0", "X0", "X2"): self.couplings["g4"] / 4,
                # quartic term (YZ)
                ("X1", "X2", "X1", "X2"): -self.couplings["g4"] / 4,
                ("X2", "X1", "X2", "X1"): -self.couplings["g4"] / 4,
                ("X1", "X2", "X2", "X1"): self.couplings["g4"] / 4,
                ("X2", "X1", "X1", "X2"): self.couplings["g4"] / 4,
                }
                )

    def build_operators_to_track(self):
        self.operators_to_track = {
            "tr(1)": SingleTraceOperator(data={(): 1}),
            "energy": self.hamiltonian,
            "x_2": SingleTraceOperator(data={("X0", "X0"): 1, ("X1", "X1"): 1, ("X2", "X2"): 1}),
            "x_4": SingleTraceOperator(data={("X0", "X0", "X0", "X0"): 1, ("X1", "X1", "X1", "X1"): 1, ("X2", "X2", "X2", "X2"): 1}),
            "p_2": SingleTraceOperator(data={("Pi0", "Pi0"): -1, ("Pi1", "Pi1"): -1, ("Pi2", "Pi2"): -1}),
            "p_4": SingleTraceOperator(data={("Pi0", "Pi0", "Pi0", "Pi0"): 1, ("Pi1", "Pi1", "Pi1", "Pi1"): 1, ("Pi2", "Pi2", "Pi2", "Pi2"): 1}),
            "neg_commutator_squared": SingleTraceOperator(data={
                # quartic term (XY)
                ("X0", "X1", "X0", "X1"): -1,
                ("X1", "X0", "X1", "X0"): 1,
                ("X0", "X1", "X1", "X0"): 1,
                ("X1", "X0", "X0", "X1"): 1,
                # quartic term (XZ)
                ("X0", "X2", "X0", "X2"): -1,
                ("X2", "X0", "X2", "X0"): -1,
                ("X0", "X2", "X2", "X0"): 1,
                ("X2", "X0", "X0", "X2"): 1,
                # quartic term (YZ)
                ("X1", "X2", "X1", "X2"): -1,
                ("X2", "X1", "X2", "X1"): -1,
                ("X1", "X2", "X2", "X1"): 1,
                ("X2", "X1", "X1", "X2"): 1,
                }
            ),
            }

    def build_symmetry_generators(self):
        self.symmetry_generators = [
            SingleTraceOperator(data={("X1", "Pi2"): 1, ("X2", "Pi1"): -1}),
            SingleTraceOperator(data={("X0", "Pi2"): 1, ("X2", "Pi0"): -1}),
            SingleTraceOperator(data={("X0", "Pi1"): 1, ("X1", "Pi0"): -1}),
            ]


class MiniBFSS(MatrixModel):
    def __init__(self, couplings):
        super().__init__(couplings)
        self.build_symmetry_generators()

    def build_matrix_system(self):
        self.matrix_system = MatrixSystem(
            operator_basis=["X0", "X1", "X2", "Pi0", "Pi1", "Pi2"],
            commutation_rules_concise={
                ("Pi0", "X0"): 1,
                ("Pi1", "X1"): 1,
                ("Pi2", "X2"): 1,
                },
                hermitian_dict={"Pi0": False, "X0": True, "Pi1": False, "X1": True, "Pi2": False, "X2": True},
                )

    def build_gauge_generator(self):
        self.gauge_generator = MatrixOperator(
            data={
                ("X0", "Pi0"): 1,
                ("Pi0", "X0"): -1,
                ("X1", "Pi1"): 1,
                ("Pi1", "X1"): -1,
                ("X2", "Pi2"): 1,
                ("Pi2", "X2"): -1,
                (): 3,
                }
                )

    def build_hamiltonian(self):
        self.hamiltonian = SingleTraceOperator(
            data={
                ("Pi0", "Pi0"): -0.5,
                ("Pi1", "Pi1"): -0.5,
                ("Pi2", "Pi2"): -0.5,
                # quartic term (XY)
                ("X0", "X1", "X0", "X1"): -self.couplings["lambda"] / 4,
                ("X1", "X0", "X1", "X0"): -self.couplings["lambda"] / 4,
                ("X0", "X1", "X1", "X0"): self.couplings["lambda"] / 4,
                ("X1", "X0", "X0", "X1"): self.couplings["lambda"] / 4,
                # quartic term (XZ)
                ("X0", "X2", "X0", "X2"): -self.couplings["lambda"] / 4,
                ("X2", "X0", "X2", "X0"): -self.couplings["lambda"] / 4,
                ("X0", "X2", "X2", "X0"): self.couplings["lambda"] / 4,
                ("X2", "X0", "X0", "X2"): self.couplings["lambda"] / 4,
                # quartic term (YZ)
                ("X1", "X2", "X1", "X2"): -self.couplings["lambda"] / 4,
                ("X2", "X1", "X2", "X1"): -self.couplings["lambda"] / 4,
                ("X1", "X2", "X2", "X1"): self.couplings["lambda"] / 4,
                ("X2", "X1", "X1", "X2"): self.couplings["lambda"] / 4,
                }
                )

    def build_operators_to_track(self):
        self.operators_to_track = {
            "tr(1)": SingleTraceOperator(data={(): 1}),
            "energy": self.hamiltonian,
            "x_2": SingleTraceOperator(data={("X0", "X0"): 1, ("X1", "X1"): 1, ("X2", "X2"): 1}),
            "x_4": SingleTraceOperator(data={("X0", "X0", "X0", "X0"): 1, ("X1", "X1", "X1", "X1"): 1, ("X2", "X2", "X2", "X2"): 1}),
            "p_2": SingleTraceOperator(data={("Pi0", "Pi0"): -1, ("Pi1", "Pi1"): -1, ("Pi2", "Pi2"): -1}),
            "p_4": SingleTraceOperator(data={("Pi0", "Pi0", "Pi0", "Pi0"): 1, ("Pi1", "Pi1", "Pi1", "Pi1"): 1, ("Pi2", "Pi2", "Pi2", "Pi2"): 1}),
            "neg_commutator_squared": SingleTraceOperator(data={
                # quartic term (XY)
                ("X0", "X1", "X0", "X1"): -1,
                ("X1", "X0", "X1", "X0"): 1,
                ("X0", "X1", "X1", "X0"): 1,
                ("X1", "X0", "X0", "X1"): 1,
                # quartic term (XZ)
                ("X0", "X2", "X0", "X2"): -1,
                ("X2", "X0", "X2", "X0"): -1,
                ("X0", "X2", "X2", "X0"): 1,
                ("X2", "X0", "X0", "X2"): 1,
                # quartic term (YZ)
                ("X1", "X2", "X1", "X2"): -1,
                ("X2", "X1", "X2", "X1"): -1,
                ("X1", "X2", "X2", "X1"): 1,
                ("X2", "X1", "X1", "X2"): 1,
                }
            ),
            }

    def build_symmetry_generators(self):
        self.symmetry_generators = [
            SingleTraceOperator(data={("X1", "Pi2"): 1, ("X2", "Pi1"): -1}),
            SingleTraceOperator(data={("X0", "Pi2"): 1, ("X2", "Pi0"): -1}),
            SingleTraceOperator(data={("X0", "Pi1"): 1, ("X1", "Pi0"): -1}),
            ]


class MiniBMN(MatrixModel):
    def __init__(self, couplings):
        super().__init__(couplings)
        self.build_symmetry_generators()

    def build_matrix_system(self):
        self.matrix_system = MatrixSystem(
            operator_basis=["X0", "X1", "X2", "P0", "P1", "P2"],
            commutation_rules_concise={
                ("P0", "X0"): -1j,
                ("P1", "X1"): -1j,
                ("P2", "X2"): -1j,
                },
                hermitian_dict={"P0": True, "X0": True, "P1": True, "X1": True, "P2": True, "X2": True},
                )

    def build_gauge_generator(self):
        self.gauge_generator = MatrixOperator(
            data={
                ("X0", "P0"): 1j,
                ("P0", "X0"): -1j,
                ("X1", "P1"): 1j,
                ("P1", "X1"): -1j,
                ("X2", "P2"): 1j,
                ("P2", "X2"): -1j,
                (): 3,}
                )

    def build_hamiltonian(self):
        self.hamiltonian = SingleTraceOperator(
            data={
                # kinetic terms
                ("P0", "P0"): 0.5,
                ("P1", "P1"): 0.5,
                ("P2", "P2"): 0.5,
                # quadratic term
                ("X0", "X0"): self.couplings["g2"] / 2,
                ("X1", "X1"): self.couplings["g2"] / 2,
                ("X2", "X2"): self.couplings["g2"] / 2,
                # quartic term (XY)
                ("X0", "X1", "X0", "X1"): -self.couplings["g4"] / 4,
                ("X1", "X0", "X1", "X0"): -self.couplings["g4"] / 4,
                ("X0", "X1", "X1", "X0"): self.couplings["g4"] / 4,
                ("X1", "X0", "X0", "X1"): self.couplings["g4"] / 4,
                # quartic term (XZ)
                ("X0", "X2", "X0", "X2"): -self.couplings["g4"] / 4,
                ("X2", "X0", "X2", "X0"): -self.couplings["g4"] / 4,
                ("X0", "X2", "X2", "X0"): self.couplings["g4"] / 4,
                ("X2", "X0", "X0", "X2"): self.couplings["g4"] / 4,
                # quartic term (YZ)
                ("X1", "X2", "X1", "X2"): -self.couplings["g4"] / 4,
                ("X2", "X1", "X2", "X1"): -self.couplings["g4"] / 4,
                ("X1", "X2", "X2", "X1"): self.couplings["g4"] / 4,
                ("X2", "X1", "X1", "X2"): self.couplings["g4"] / 4,
                }
                )

    def build_operators_to_track(self):
        self.operators_to_track = {
            "tr(1)": SingleTraceOperator(data={(): 1}),
            "energy": self.hamiltonian,
            "x_2": SingleTraceOperator(data={("X0", "X0"): 1, ("X1", "X1"): 1, ("X2", "X2"): 1}),
            "x_4": SingleTraceOperator(data={("X0", "X0", "X0", "X0"): 1, ("X1", "X1", "X1", "X1"): 1, ("X2", "X2", "X2", "X2"): 1}),
            "p_2": SingleTraceOperator(data={("P0", "P0"): -1, ("P1", "P1"): -1, ("P2", "P2"): -1}),
            "p_4": SingleTraceOperator(data={("P0", "P0", "P0", "P0"): 1, ("P1", "P1", "P1", "P1"): 1, ("P2", "P2", "P2", "P2"): 1}),
        }

    def build_symmetry_generators(self):
        self.symmetry_generators = [
            SingleTraceOperator(data={("X1", "P2"): 1, ("X2", "P1"): -1}),
            SingleTraceOperator(data={("X0", "P2"): 1, ("X2", "P0"): -1}),
            SingleTraceOperator(data={("X0", "P1"): 1, ("X1", "P0"): -1}),
            ]




