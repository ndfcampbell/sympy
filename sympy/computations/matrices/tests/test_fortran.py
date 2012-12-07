from sympy.computations.inplace import ExprToken
from sympy.computations.matrices.fortran import nameof
from sympy.core import Symbol

def test_nameof():
    assert nameof(ExprToken(1, 'hello')) == 1
    assert nameof(ExprToken(Symbol('x'), 'y')) == 'y'
