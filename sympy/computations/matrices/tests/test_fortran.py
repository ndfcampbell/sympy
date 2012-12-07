from sympy.computations.inplace import ExprToken
from sympy.computations.matrices.fortran import (nameof,
        unique_tokened_variables)
from sympy.core import Symbol

def test_nameof():
    assert nameof(ExprToken(1, 'hello')) == 1
    assert nameof(ExprToken(Symbol('x'), 'y')) == 'y'

def test_unique_tokened_variables():
    x,y,z = map(Symbol, 'xyz')
    vars = ExprToken(x, 'x'), ExprToken(y, 'y'), ExprToken(z, 'x')
    result = unique_tokened_variables(vars)
    assert len(result) == 2
