from sympy.computations.matrices.fortran2 import *
from sympy import MatrixSymbol, Symbol
from sympy.computations.inplace import inplace_compile
from sympy.matrices.expressions.fourier import DFT
from sympy.computations.matrices.compile import compile

def test_simple():
    n = Symbol('n')
    x = MatrixSymbol('X', n, 1)
    y = MatrixSymbol('y', n, 1)
    inputs = [x]
    outputs = [DFT(n)*x]
    types = {q: 'complex(kind=8)' for q in [x, DFT(n), DFT(n)*x]}
    types[Symbol('plan')] = 'type(C_PTR)'

    mathcomp = next(compile(inputs, outputs))
    ic = inplace_compile(mathcomp)
    s = generate_fortran(ic, inputs, outputs, types, 'f')

    print '\n' + s
    assert isinstance(s, str)


