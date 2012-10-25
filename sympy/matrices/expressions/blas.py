from sympy.computations import Computation
from sympy import Basic, Tuple

alpha = Symbol('alpha')
beta  = Symbol('beta')
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('A', m, k)
C = MatrixSymbol('C', n, k)
S = MatrixSymbol('S', n, n)
x = MatrixSymbol('x', n, 1)
a = MatrixSymbol('a', m, 1)
b = MatrixSymbol('b', k, 1)

class MM(BLAS):
    _inputs   = (alpha, A, B, beta, C)
    _outputs  = (alpha*A*B + beta*C)
    view_map  = {0: 4}
    condition = True

class GEMM(MM):
    pass

class SYMM(MM):
    condition = Q.symmetric(A) | Q.symmetric(B)

class TRMM(MM):
    condition = Q.triangular(A) | Q.triangular(B)

class SM(BLAS):
    _inputs   = (alpha, A, B)
    _outputs  = (alpha*A.I*B,)
    view_map  = {0: 2}
    condition = True

class TRSM(SM):
    condition = Q.triangular(A)

class MV(BLAS):
    _inputs   = (alpha, A, a, beta, b)
    _outputs  = (alpha*A*a + beta*b,)
    view_map  = {0: 4}
    condition = True

class GEMV(MV):
    pass

class SYMV(MV):
    condition = Q.symmetric(A) | Q.symmetric(B)

class TRMV(MV):
    condition = Q.triangular(A) | Q.triangular(B)

class SV(BLAS):
    _inputs   = (alpha, A, x)
    _outputs  = (alpha*A.I*x)
    view_map  = {0: 2}
    condition = True

class TRSV(SV):
    condition = Q.triangular(A)

class LU(BLAS):
    _inputs   = (S,)
    _outputs  = (Lof(S), Uof(S))
    view_map  = {0: 0, 1: 0}
    condition = True

class Cholesky(LU):
    condition = Q.symmetric(S) & Q.positive_definite(S)

class BLAS(Computation):
    # TODO: metaclass magic for s/d/z prefixes
    pass
