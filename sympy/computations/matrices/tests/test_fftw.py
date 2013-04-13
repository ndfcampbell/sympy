from sympy.computations.matrices.fortran2 import *
from sympy import MatrixSymbol, Symbol
from sympy.computations.inplace import inplace_compile
from sympy.matrices.expressions.fourier import DFT
from sympy.computations.matrices.compile import compile
from sympy.computations.matrices.fftw import FFTW, Plan

n = Symbol('n')
x = MatrixSymbol('X', n, 1)
c = FFTW(x)

types = {q: 'complex(kind=8)' for q in [x, DFT(n), DFT(n)*x]}
types[Plan()] = 'type(C_PTR)'

def test_DAG_search():
    assert next(compile([x], [DFT(n)*x])) == FFTW(x)


def test_code_generation():
    ic = inplace_compile(c)
    s = generate_fortran(ic, c.inputs, c.outputs, types, 'f')

    print '\n' + s
    assert isinstance(s, str)

def test_DAG_search_in_context():
    A = MatrixSymbol('A', n, n)
    c = next(compile([A, x], [A*DFT(n)*x]))
    assert FFTW(x) in c.computations
    cs = compile([A, x], [DFT(n)*A*x])
    assert any(FFTW(A*x) in c.computations for c in cs)
