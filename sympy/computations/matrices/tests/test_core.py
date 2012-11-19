from sympy.computations.matrices.core import remove_numbers
from sympy.matrices import MatrixSymbol
from sympy.core import Symbol, S

def test_remove_numbers():
    X = MatrixSymbol('X', 1, 3)
    x = Symbol('x')
    assert remove_numbers([x, X, 1, 1.0, S.One]) == [x, X]
