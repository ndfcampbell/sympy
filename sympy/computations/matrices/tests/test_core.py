from sympy.computations.matrices.core import remove_numbers
from sympy.computations.matrices.blas import GEMM
from sympy.matrices import MatrixSymbol
from sympy.core import Symbol, S

def test_remove_numbers():
    X = MatrixSymbol('X', 1, 3)
    x = Symbol('x')
    assert remove_numbers([x, X, 1, 1.0, S.One]) == [x, X]

def test_inplace():
    a = Symbol('a')
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    g = GEMM(a, X, Y, S.Zero, Y)
    assert g.inplace == {0: 2}
