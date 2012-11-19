from sympy.computations.matrices.shared import detranspose
from sympy.matrices.expressions import MatrixSymbol

def test_detranspose():
    X = MatrixSymbol('X', 2, 3)
    assert detranspose(X) is X
    assert detranspose(X.T) is X
