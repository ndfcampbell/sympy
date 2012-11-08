from sympy.matrices.expressions.matcomp import *
from sympy import Q, Integer
from sympy.utilities.pytest import XFAIL
from sympy.matrices.expressions.lapack import *


n,m,k = symbols('n,m,k')
C = MatrixSymbol('C', n, k)
S = MatrixSymbol('S', n, n)

def test_GESV():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, k)
    gesv = GESV(A, B)
    assert gesv.inputs  == (A, B)
    assert gesv.outputs[0] == A.I*B
    print gesv.in_types
    assert gesv.in_types == ('real*8', 'real*8')
    assert gesv.out_types == ('real*8', 'integer', 'integer')
    assert gesv.types()[gesv.outputs[1]] == 'integer'

def test_GETRF():
    A = MatrixSymbol('A', n, n)
    getrf = GETRF(A)
