from sympy.matrices.expressions.matcomp import *
from sympy import Q, Integer
from sympy.utilities.pytest import XFAIL
from sympy.matrices.expressions.lapack import *


n,m,k = symbols('n,m,k')
C = MatrixSymbol('C', n, k)
S = MatrixSymbol('S', n, n)
A = MatrixSymbol('A', n, n)
B = MatrixSymbol('B', n, k)
gesv = GESV(A, B)
posv = POSV(A, B)

def test_GESV():
    assert gesv.inputs  == (A, B)
    assert gesv.outputs[0] == A.I*B

def test_in_out_types():
    assert gesv.in_types == ('real*8', 'real*8')
    assert gesv.out_types == ('real*8', 'integer', 'integer')
    assert gesv.types()[gesv.outputs[1]] == 'integer'

def test_ipiv_header_declaration():
    assert 'integer, intent(out) :: IPIV(n)' in gesv.print_Fortran(str)
    assert 'IPIV' in gesv.header(str)

def test_gesv_compiles():
    assert callable(gesv.build(str))

def test_POSV():
    assert posv.outputs[0] == A.I*B
    assert not POSV.valid(posv.inputs, True)
    assert POSV.valid(posv.inputs, Q.symmetric(A) & Q.positive_definite(A))

def test_posv_compiles():
    assert callable(posv.build(str))

def test_types():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, k)
    gesv = GESV(A, B, 'Z')
    assert gesv.in_types == ('complex*16', 'complex*16')
    assert gesv.out_types == ('complex*16', 'integer', 'integer')

def test_GETRF():
    A = MatrixSymbol('A', n, n)
    getrf = GETRF(A)
