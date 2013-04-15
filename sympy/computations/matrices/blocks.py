from sympy.computations.core import Computation
from sympy.matrices.expressions import BlockMatrix
from sympy import Basic

class JoinBlocks(Computation):
    """ Join together matrices into a BlockMatrix.  Coalesce."""
    arg     = property(lambda self: self.args[0])
    inputs  = property(lambda self: tuple(self.arg.blocks))
    outputs = property(lambda self: (self.arg,))

    def fortran_function_interface(self):
        return ""

    def fortran_call(self, input_names, output_names):
        return ""

class SeparateBlocks(Computation):
    """ Separate matrix into a sub blocks.  Decompose."""
    arg     = property(lambda self: self.args[0])
    inputs  = property(lambda self: (self.arg,))
    outputs = property(lambda self: tuple(self.arg.blocks))

    def fortran_function_interface(self):
        return ""

    def fortran_call(self, input_names, output_names):
        return ""

class Slice(Computation):
    """ Slice sub-matrix """
    arg     = property(lambda self: self.args[0])
    inputs  = property(lambda self: (self.arg.arg,))
    outputs = property(lambda self: (self.arg,))

    def fortran_function_interface(self):
        return ""

    def fortran_call(self, input_names, output_names):
        return ""
