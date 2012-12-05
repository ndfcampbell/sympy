from sympy.computations.matrices.core import MatrixCall
from sympy.computations.matrices.shared import (alpha, beta, n, m, k, C,
        x, a, b)
from sympy import Q, S, Symbol, Basic
from sympy.matrices.expressions import MatrixSymbol, PermutationMatrix

A = MatrixSymbol('A', n, n)
B = MatrixSymbol('B', n, m)
INFO = Symbol('INFO')

class IPIV(MatrixSymbol):
    def __new__(cls, A):
        return Basic.__new__(cls, A)

    A = property(lambda self: self.args[0])
    shape = property(lambda self: (1, self.A.shape[0]))
    name = property(lambda self: 'IPIV')

class LAPACK(MatrixCall):
    """ Linear Algebra PACKage - Dense Matrix computation """
    flags = ["-llapack"]

class GESV(LAPACK):
    """ General Matrix Vector Solve """
    _inputs   = (A, B)
    _outputs  = (PermutationMatrix(IPIV(A.I*B))*A.I*B, IPIV(A.I*B), INFO)
    view_map = {0: 1}
    condition = True  # TODO: maybe require S to be invertible?

class LASWP(LAPACK):
    """ Permute rows in a matrix """
    _inputs   = (PermutationMatrix(IPIV(A))*A, IPIV(A))
    _outputs  = (A,)
    view_map  = {0: 0}
    condition = True

class POSV(LAPACK):
    """ Symmetric Positive Definite Matrix Solve """
    _inputs   = (A, B)
    _outputs  = (A.I*B, INFO)
    view_map = {0: 1}
    condition = Q.positive_definite(A) & Q.symmetric(A)



