from sympy.matrices.expressions.blas import *
from sympy.computations import CompositeComputation
from sympy import *

n = Symbol('n', integer=True)
A, B, C, X, Y = [MatrixSymbol(s, n, n) for s in 'ABCXY']
x,y,a,b = [MatrixSymbol(s, n, 1) for s in 'xyab']
alpha, beta, gamma = symbols('alpha, beta, gamma')

def test_MM():
    mm = MM(alpha, A, B, beta, C)
    assert mm.inputs  == set((alpha, A, B, beta, C))
    assert mm.outputs == set((alpha*A*B + beta*C,))

def test_SV():
    sv = SV(A, y)
    assert sv.outputs == set((A.I*y,))

def test_composite():
    mm = MM(alpha, A, B, beta, C)
    sv = SV(alpha*A*B + beta*C, y)
    cc = CompositeComputation(mm, sv)

    assert cc.inputs  == set((alpha, A, B, beta, C, y))
    assert cc.outputs == set(((alpha*A*B + beta*C).I*y,))
    assert cc.dag_io() == {mm: set([sv]), sv: set([])}
    assert cc.dag_oi() == {sv: set([mm]), mm: set([])}
