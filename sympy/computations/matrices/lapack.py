from sympy.computations.matrices.core import MatrixCall, remove_numbers
from sympy.computations.core import unique
from sympy.computations.matrices.shared import detranspose
from sympy.computations.matrices.shared import (alpha, beta, n, m, k, C,
        x, a, b)
from sympy import Q, S, Symbol, Basic
from sympy.matrices.expressions import MatrixSymbol

A = MatrixSymbol('A', n, n)
B = MatrixSymbol('B', n, m)
INFO = Symbol('INFO')

class IPIV(MatrixSymbol):
    def __new__(cls, A, token='None'):
        return Basic.__new__(cls, A, token)

    A = property(lambda self: self.args[0])
    token = property(lambda self: self.args[1])
    shape = property(lambda self: (self.A.shape[1], 1))
    name = property(lambda self: 'IPIV')

class LAPACK(MatrixCall):
    """ Linear Algebra PACKage - Dense Matrix computation """
    flags = ["-llapack"]

class GESV(LAPACK):
    """ General Matrix Vector Solve """
    _inputs   = (A, B)
    _outputs  = (A.I*B, IPIV(A), INFO)
    view_map = {0: 1}
    condition = True  # TODO: maybe require S to be invertible?

class POSV(LAPACK):
    """ Symmetric Positive Definite Matrix Solve """
    _inputs   = (A, B)
    _outputs  = (A.I*B, INFO)
    view_map = {0: 1}
    condition = Q.positive_definite(A) & Q.symmetric(A)



