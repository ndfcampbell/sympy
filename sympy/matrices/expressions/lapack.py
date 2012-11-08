from sympy import Basic, Symbol, Q, symbols, ask
from sympy.matrices.expressions import MatrixSymbol, Transpose
from sympy.utilities.iterables import merge

from calls import MatrixCall, basetypes
from calls import alpha, beta, n, m, k, A, B, C, S, x, a, b
from calls import LD

INFO = Symbol('INFO', integer=True)

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
    _inputs   = (S, C)
    _outputs  = (S.I*C, IPIV(S), INFO)
    _out_types = (None, 'integer', 'integer')
    view_map = {0: 1}
    fortran_template = ("call %(fn)s(%(N)s, %(NRHS)s, %(A)s, "
                        "%(LDA)s, %(IPIV)s, %(B)s, %(LDB)s, %(INFO)s)")
    condition = True  # TODO: maybe require S to be invertible?
    def codemap(self, namefn, assumptions=True):
        varnames = 'A B IPIV INFO'.split()
        A, B = self.inputs
        _, IPIV, INFO = self.outputs
        names    = map(namefn, (A, B, IPIV, INFO))
        namemap  = dict(zip(varnames, names))
        other = {'LDA': LD(A),
                 'LDB': LD(B),
                 'N': str(A.shape[0]),
                 'NRHS': str(B.shape[1]),
                 'fn': self.fnname()}
        return merge(namemap, other)

class POSV(LAPACK):
    """ Symmetric Positive Definite Vector Solve """
    _inputs   = (S, C)
    _outputs  = (S.I*C, INFO)
    _out_types = (None, 'integer')
    view_map = {0: 1}
    fortran_template = ("call %(fn)s(%(UPLO)s, %(N)s %(NRHS)s, %(A)s, "
                        "%(LDA)s, %(B)s, %(LDB)s, %(INFO)s)")
    condition = Q.positive_definite(S) & Q.symmetric(S)

    def codemap(self, namefn, assumptions=True):
        varnames = 'A B INFO'.split()
        A, B = self.inputs
        _, INFO = self.outputs
        names    = map(namefn, (A, B, INFO))
        namemap  = dict(zip(varnames, names))
        other = {'LDA': LD(A),
                 'LDB': LD(B),
                 'N': str(A.shape[0]),
                 'NRHS': str(B.shape[1]),
                 'UPLO': 'U',
                 'fn': self.fnname()}
        return merge(namemap, other)

# TODO: Make these classes
class Lof(Basic):    pass
class Uof(Basic):    pass

class LU(LAPACK):
    _inputs = (S,)
    _outputs = (Lof(S), Uof(S), IPIV, INFO)
    view_map  = {0: 0, 1: 0}
    condition = True

    def codemap(self, namefn, assumptions=True):
        varnames = 'A IPIV INFO'.split()
        A = self.inputs
        _, _, IPIV, INFO = self.outputs
        names    = map(namefn, (A, IPIV, INFO))
        namemap  = dict(zip(varnames, names))
        other = {'LDA': LD(A),
                 'M': str(A.shape[0]),
                 'N': str(A.shape[1]),
                 'UPLO': 'L',
                 'fn': self.fnname()}
        return merge(namemap, other)

class GETRF(LU):
    """ General Triangular Factorization - Basic LU """
    fortran_template = ("call %(fn)s( %(M)s, %(N)s, %(A)s, %(LDA)s, "
                                     "%(IPIV)s, %(INFO)s )")

class POTRF(LU):
    """ Cholesky LU Decomposition """
    _outputs = (Lof(S), IPIV, INFO)
    view_map  = {0: 0}
    condition = Q.symmetric(S) & Q.positive_definite(S)

    fortran_template = ("call %(fn)s( %(UPLO)s, %(N)s, %(A)s, "
                                     "%(LDA)s, %(INFO)s )")
