from sympy.matrices.expressions.matcomp import basic_names, remove_numbers
from sympy import Symbol, MatrixSymbol, S

def test_basic_names():
    X = Symbol('X')
    x = Symbol('x')
    basic_names._cache = {}
    assert basic_names(X) == 'X'
    assert basic_names(x) == 'x_'

def test_remove_numbers():
    X = MatrixSymbol('X', 1, 3)
    x = Symbol('x')
    assert remove_numbers([x, X, 1, 1.0, S.One]) == [x, X]
