""" Basic Linear Algebra Subroutines """

from sympy import Basic, Symbol, Q, symbols, ask
from sympy.matrices.expressions import MatrixSymbol, Transpose
from sympy.utilities.iterables import merge
from matcomp import MatrixCall

class BLAS(MatrixCall):
    """ Basic Linear Algebra Subroutine - Dense Matrix computation """
    flags = ["-lblas"]

# Pattern variables
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
    """ Matrix Multiply """
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
                 'fn': self.fnname(),
                 'SIDE': left_or_right(A, B, Q.symmetric, assumptions),
                 'DIAG': diag(A, assumptions),
                 'UPLO': 'U'} # TODO: symmetric matrices might be stored low
        return merge(namemap, other)

class GEMM(MM):
    """ General Matrix Multiply """
    fortran_template = ("call %(fn)s('%(TRANSA)s', '%(TRANSB)s', "
                        "%(M)s, %(N)s, %(K)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, "
                        "%(B)s, %(LDB)s, %(beta)s, %(C)s, %(LDC)s)")

class SYMM(MM):
    """ Symmetric Matrix Multiply """
    condition = Q.symmetric(A) | Q.symmetric(B)
    fortran_template = ("call %(fn)s('%(SIDE)s', '%(UPLO)s', %(M)s, %(N)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, %(B)s, %(LDB)s, "
                        "%(beta)s, %(C)s, %(LDC)s)")

class TRMM(MM):
    """ Triangular Matrix Multiply """
    _inputs = (alpha, A, B)
    _outputs = (alpha*A*B,)
    view_map  = {0: 2}
    condition = Q.triangular(A) | Q.triangular(B)
    fortran_template = ("call %(fn)s('%(SIDE)s', '%(UPLO)s', '%(TRANSA)s', "
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
                 'fn': self.fnname(),
                 'SIDE': left_or_right(A, B, Q.triangular, assumptions),
                 'DIAG': diag(A, assumptions),
                 'UPLO': uplo(A, assumptions)}
        return merge(namemap, other)

class SM(BLAS):
    """ Matrix Solve """
    _inputs   = (alpha, S, A)
    _outputs  = (alpha*S.I*A,)
    view_map  = {0: 2}
    condition = True

class TRSM(SM):
    """ Triangular Matrix Solve """
    condition = Q.triangular(A)

class MV(BLAS):
    """ Matrix Vector Multiply """
    _inputs   = (alpha, A, a, beta, x)
    _outputs  = (alpha*A*a + beta*x,)
    view_map  = {0: 4}
    condition = True

class GEMV(MV):
    """ General Matrix Vector Multiply """
    pass

class SYMV(MV):
    """ Symmetric Matrix Vector Multiply """
    condition = Q.symmetric(A) | Q.symmetric(B)

class TRMV(MV):
    """ Triangular Matrix Vector Multiply """
    condition = Q.triangular(A) | Q.triangular(B)

class SV(BLAS):
    """ Matrix Vector Solve """
    _inputs   = (S, x)
    _outputs  = (S.I*x,)
    view_map  = {0: 1}
    condition = True

class TRSV(SV):
    """ Triangular Matrix Vector Solve """
    fortran_template = ("call %(fn)s('%(UPLO)s', '%(TRANS)s', '%(DIAG)s', "
                        "%(N)s, %(A)s, %(LDA)s, %(x)s, %(INCX)s)")
    condition = Q.lower_triangular(S) | Q.upper_triangular(S)
    def codemap(self, namefn, assumptions=True):
        varnames = 'A x'.split()
        A, x = self.inputs
        names    = map(namefn, (detranspose(A), x))
        namemap  = dict(zip(varnames, names))
        other = {'TRANS': trans(A),
                 'LDA': LD(A),
                 'N':str(A.shape[0]),
                 'fn': self.fnname(),
                 'DIAG': diag(A, assumptions),
                 'UPLO': uplo(A, assumptions),
                 'INCX': '1'}
        return merge(namemap, other)

# TODO: Make these classes
class Lof(Basic):    pass
class Uof(Basic):    pass

class LU(BLAS):
    """ LU Decomposition """
    _inputs   = (S,)
    _outputs  = (Lof(S), Uof(S))
    view_map  = {0: 0, 1: 0}
    condition = True

class Cholesky(LU):
    """ Cholesky LU Decomposition """
    condition = Q.symmetric(S) & Q.positive_definite(S)

def trans(A):
    """ Return 'T' if A is a transpose, else 'N' """
    if isinstance(A, Transpose):
        return 'T'
    else:
        return 'N'

def uplo(A, assumptions):
    """ Return 'U' if A is stored in the upper Triangular 'U' if lower """
    if ask(Q.upper_triangular(A), assumptions):
        return 'U'
    if ask(Q.lower_triangular(A), assumptions):
        return 'L'

def LD(A):
    """ Leading dimension of matrix """
    # TODO make sure we don't use transposed matrices in untransposable slots
    return str(detranspose(A).shape[0])

def left_or_right(A, B, predicate, assumptions):
    """ Return 'L' if predicate is true of A, 'R' if predicate is true of B """
    if ask(predicate(A), assumptions):
        return 'L'
    if ask(predicate(B), assumptions):
        return 'R'

def diag(A, assumptions):
    """ Return 'U' if A is unit_triangular """
    if ask(Q.unit_triangular(A), assumptions):
        return 'U'
    else:
        return 'N'

def detranspose(A):
    """ Unpack a transposed matrix """
    if isinstance(A, Transpose):
        return A.arg
    else:
        return A
