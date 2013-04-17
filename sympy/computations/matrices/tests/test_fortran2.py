from sympy.computations.matrices.fortran2 import *
from sympy import MatrixSymbol, Symbol
from sympy.computations.inplace import inplace_compile
from sympy.computations.matrices.compile import compile

n = Symbol('n')
X = MatrixSymbol('X', n, n)
y = MatrixSymbol('y', n, 1)
inputs = [X, y]
outputs = [X*y]
mathcomp = next(compile(inputs, outputs))
ic = inplace_compile(mathcomp)
types = {q: 'real*8' for q in [X, y, X*y]}
s = generate_fortran(ic, inputs, outputs, types, 'f')
def test_simple():

    print '\n' + s
    assert isinstance(s, str)
    assert "call dgemm('N', 'N', n, 1, n" in s
    assert "1.0" in s
    assert "X, n, y, n,"  in s

def test_dimensions():
    assert set(dimensions(ic)) == set((n, ))
    assert 'integer :: n' in s

def test_dimension_initialization():
    assert dimension_initialization(n, ExprToken(y, 'yvar')) == 'n = size(yvar, 1)'
    assert 'n = size(X, 1)' in s or 'n = size(y, 1)' in s
