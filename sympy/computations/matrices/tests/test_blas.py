from sympy.computations.matrices.blas import GEMM, SYMM
from sympy.matrices.expressions import MatrixSymbol
from sympy.core import Symbol
from sympy import Q

a, b, c, d, x, y, z, n, m, l, k = map(Symbol, 'abcdxyznmlk')

def test_GEMM():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    Z = MatrixSymbol('Z', n, n)
    assert GEMM(a, X, Y, b, Z).inputs == (a, X, Y, b, Z)
    assert GEMM(a, X, Y, b, Z).outputs == (a*X*Y+b*Z, )
    assert GEMM(1, X, Y, 0, Y).inputs == (X, Y)

def test_valid():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    C = MatrixSymbol('C', n, n)
    assert GEMM.valid((1, A, B, 2, C), True)
    assert not SYMM.valid((1, A, B, 2, C), True)
    assert SYMM.valid((1, A, B, 2, C), Q.symmetric(A))
    assert SYMM.valid((1, A, B, 2, C), Q.symmetric(B))
