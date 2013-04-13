from sympy.matrices.expressions.eig import EigenVectors, EigenValues
from sympy.matrices.expressions import MatrixSymbol
from sympy.core import Symbol

n = Symbol('n')
X = MatrixSymbol('X', n, n)

def test_EigenVectors():
    ev = EigenVectors(X)
    assert ev.shape == X.shape

def test_EigenValues():
    ev = EigenValues(X)
    assert ev.shape == (X.rows, 1)
