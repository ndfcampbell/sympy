from sympy.computations.matrices.fortran2 import *
from sympy import MatrixSymbol, Symbol
from sympy.computations.inplace import inplace_compile
from sympy.computations.matrices.compile import compile

def test_simple():
    n = Symbol('n')
    X = MatrixSymbol('X', n, n)
    y = MatrixSymbol('y', n, 1)
    inputs = [X, y]
    outputs = [X*y]
    types = {q: 'real*8' for q in [X, y, X*y]}

    mathcomp = next(compile(inputs, outputs))
    ic = inplace_compile(mathcomp)
    s = generate_fortran(ic, inputs, outputs, types, [], 'f')

    print s
    assert isinstance(s, str)

