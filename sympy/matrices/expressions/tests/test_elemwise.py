from sympy.matrices.expressions.elemwise import ElemwiseMatrix
from sympy import Symbol, MatrixSymbol, Lambda

x = Symbol('x')
y = Symbol('y')
X = MatrixSymbol('X', 3, 3)
Y = MatrixSymbol('Y', 3, 3)

def test_simple():
    X2 = ElemwiseMatrix(Lambda(x, x**2), X)
    assert X2[0, 1] == X[0, 1]**2
    assert X2.shape == X.shape

def test_compound():
    f = Lambda((x, y), x**y)
    M = ElemwiseMatrix(f, X, Y)
    assert M[0, 1] == X[0, 1]**Y[0, 1]
    assert M.shape == Y.shape
