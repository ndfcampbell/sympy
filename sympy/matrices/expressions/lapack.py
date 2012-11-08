from sympy import Basic, Symbol, Q, symbols, ask
from sympy.matrices.expressions import MatrixSymbol, Transpose
from sympy.utilities.iterables import merge

from calls import MatrixCall, basetypes
from calls import alpha, beta, n, m, k, A, B, C, S, x, a, b

IPIV = MatrixSymbol('IPIV', n, 1) # TODO: encode integer type
INFO = Symbol('INFO', integer=True)

class LAPACK(MatrixCall):
    """ Linear Algebra PACKage - Dense Matrix computation """
    flags = ["-llapack"]

class GESV(LAPACK):
    """ General Matrix Vector Solve """
    _inputs   = (S, C)
    _outputs  = (S.I*C, IPIV, INFO)
    fortran_template = ("call %(fn)s('%(N)s', '%(NRHS)s', '%(A)s', "
                        "%(LDA)s, %(IPIV)s, %(B)s, %(LDB)s, %(INFO)s)")
    condition = True  # TODO: maybe require S to be invertible?
    def codemap(self, namefn, assumptions=True):
        varnames = 'A B IPIV INFO'.split()
        A, B = self.inputs
        _, IPIV, INFO = self.outputs
        names    = map(namefn, (A, B, IPIV, INFO))
        namemap  = dict(zip(varnames, names))
        other = {'LDA': LD(A),
                 'N': str(A.shape[0]),
                 'NRHS': str(B.shape[1]),
                 'fn': self.fnname()}
        return merge(namemap, other)

