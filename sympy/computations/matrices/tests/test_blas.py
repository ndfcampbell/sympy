from sympy.computations.matrices.blas import GEMM, SYMM
from sympy.matrices.expressions import MatrixSymbol
from sympy.core import Symbol

a, b, c, d, x, y, z, n, m, l, k = map(Symbol, 'abcdxyznmlk')

def test_GEMM():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    Z = MatrixSymbol('Z', n, n)
    assert GEMM(a, X, Y, b, Z).inputs == (a, X, Y, b, Z)
    assert GEMM(a, X, Y, b, Z).outputs == (a*X*Y+b*Z, )
    assert GEMM(1, X, Y, 0, Y).inputs == (X, Y)


