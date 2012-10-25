from sympy.computations import Computation
from sympy import Basic, Tuple, Symbol, Q, symbols, ask
from sympy.matrices.expressions import MatrixSymbol, Transpose
from sympy.rules.tools import subs
from sympy.utilities.iterables import merge

class BLAS(Computation):
    # TODO: metaclass magic for s/d/z prefixes
    def __new__(cls, *inputs):
        mapping = dict(zip(cls._inputs, inputs))
        outputs = subs(mapping)(Tuple(*cls._outputs))
        return Computation.__new__(cls, inputs, outputs)

    def print_Fortran(self, namefn, assumptions=True):
        return self.fortran_template % self.codemap(namefn, assumptions)

alpha = Symbol('alpha')
beta  = Symbol('beta')
n,m,k = symbols('n,m,k')
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, k)
C = MatrixSymbol('C', n, k)
S = MatrixSymbol('S', n, n)
x = MatrixSymbol('x', n, 1)
a = MatrixSymbol('a', m, 1)
b = MatrixSymbol('b', k, 1)

class MM(BLAS):
    _inputs   = (alpha, A, B, beta, C)
    _outputs  = (alpha*A*B + beta*C,)
    view_map  = {0: 4}
    condition = True

    def codemap(self, namefn, assumptions=True):
        varnames = 'alpha A B beta C'.split()
        alpha, A, B, beta, C = self.inputs
        names    = map(namefn, (alpha, detranspose(A), B, beta, C))
        namemap  = dict(zip(varnames, names))
        other = {'TRANSA': trans(A), 'TRANSB': trans(B),
                 'LDA': LD(A), 'LDB': LD(B), 'LDC': LD(C),
                 'M':str(C.shape[0]), 'K':str(B.shape[0]), 'N':str(C.shape[1]),
                 'fn': self.__class__.__name__,
                 'SIDE': left_or_right(A, B, Q.symmetric, assumptions),
                 'DIAG': diag(A, assumptions),
                 'UPLO': 'U'} # TODO: symmetric matrices might be stored low
        return merge(namemap, other)

class GEMM(MM):
    fortran_template = ("%(fn)s('%(TRANSA)s', '%(TRANSB)s', %(M)s, %(N)s, %(K)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, "
                        "%(B)s, %(LDB)s, %(beta)s, %(C)s, %(LDC)s)")

class SYMM(MM):
    condition = Q.symmetric(A) | Q.symmetric(B)
    fortran_template = ("%(fn)s('%(SIDE)s', '%(UPLO)s', %(M)s, %(N)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, %(B)s, %(LDB)s, "
                        "%(beta)s, %(C)s, %(LDC)s)")

class TRMM(MM):
    _inputs = (alpha, A, B)
    _outputs = (alpha*A*B,)
    view_map  = {0: 2}
    condition = Q.triangular(A) | Q.triangular(B)
    fortran_template = ("%(fn)s('%(SIDE)s', '%(UPLO)s', '%(TRANSA)s', "
                        "'%(DIAG)s', %(M)s, %(N)s, %(alpha)s, %(A)s, %(LDA)s, "
                        "%(B)s, %(LDB)s)")
    def codemap(self, namefn, assumptions=True):
        varnames = 'alpha A B'.split()
        alpha, A, B = self.inputs
        names    = map(namefn, (alpha, detranspose(A), B))
        namemap  = dict(zip(varnames, names))
        other = {'TRANSA': trans(A), 'TRANSB': trans(B),
                 'LDA': LD(A), 'LDB': LD(B),
                 'M':str(B.shape[0]), 'N':str(B.shape[1]),
                 'fn': self.__class__.__name__,
                 'SIDE': left_or_right(A, B, Q.triangular, assumptions),
                 'DIAG': diag(A, assumptions),
                 'UPLO': uplo(A, assumptions)}
        return merge(namemap, other)

class SM(BLAS):
    _inputs   = (alpha, S, A)
    _outputs  = (alpha*S.I*A,)
    view_map  = {0: 2}
    condition = True

class TRSM(SM):
    condition = Q.triangular(A)

class MV(BLAS):
    _inputs   = (alpha, A, a, beta, x)
    _outputs  = (alpha*A*a + beta*x,)
    view_map  = {0: 4}
    condition = True

class GEMV(MV):
    pass

class SYMV(MV):
    condition = Q.symmetric(A) | Q.symmetric(B)

class TRMV(MV):
    condition = Q.triangular(A) | Q.triangular(B)

class SV(BLAS):
    _inputs   = (S, x)
    _outputs  = (S.I*x,)
    view_map  = {0: 2}
    condition = True

class TRSV(SV):
    condition = Q.triangular(A)

# TODO: Make these classes
class Lof(Basic):    pass
class Uof(Basic):    pass
class LU(BLAS):
    _inputs   = (S,)
    _outputs  = (Lof(S), Uof(S))
    view_map  = {0: 0, 1: 0}
    condition = True

class Cholesky(LU):
    condition = Q.symmetric(S) & Q.positive_definite(S)

def trans(A):
    if isinstance(A, Transpose):
        return 'T'
    else:
        return 'N'

def uplo(A, assumptions):
    if ask(Q.upper_triangular(A), assumptions):
        return 'U'
    if ask(Q.lower_triangular(A), assumptions):
        return 'L'

def LD(A):
    # TODO make sure we don't use transposed matrices in untransposable slots
    return str(detranspose(A).shape[0])

def left_or_right(A, B, predicate, assumptions):
    if ask(predicate(A), assumptions):
        return 'L'
    if ask(predicate(B), assumptions):
        return 'R'

def diag(A, assumptions):
    if ask(Q.unit_triangular(A), assumptions):
        return 'U'
    else:
        return 'N'

def detranspose(A):
    if isinstance(A, Transpose):
        return A.arg
    else:
        return A
