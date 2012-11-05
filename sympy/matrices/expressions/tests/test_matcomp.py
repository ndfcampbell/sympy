from sympy.matrices.expressions.matcomp import basic_names
from sympy import Symbol

def test_basic_names():
    X = Symbol('X')
    x = Symbol('x')
    assert basic_names(X) == 'X'
    assert basic_names(x) == 'x_'
