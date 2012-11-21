from sympy.computations.matrices.lapack import GESV, POSV
from sympy.matrices.expressions import MatrixSymbol
from sympy.core import Symbol
from sympy import Q

a, b, c, d, x, y, z, n, m, l, k = map(Symbol, 'abcdxyznmlk')

def test_GESV():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, m)
    assert GESV(X, Y).inputs  == (X, Y)
    assert GESV(X, Y).outputs[0] == X.I*Y

def test_POSV():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, m)
    posv = POSV(X, Y)
    assert posv.outputs[0] == X.I*Y
    assert not POSV.valid(posv.inputs, True)
    assert POSV.valid(posv.inputs, Q.symmetric(X) & Q.positive_definite(X))
