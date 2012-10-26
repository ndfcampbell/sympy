from sympy.matrices.expressions.blas import *
from sympy.computations import CompositeComputation
from sympy import *

n,m,k = symbols('n,m,k')
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, k)
C = MatrixSymbol('C', n, k)
S = MatrixSymbol('S', n, n)

x,y,a,b = [MatrixSymbol(s, n, 1) for s in 'xyab']
alpha, beta, gamma = symbols('alpha, beta, gamma')

def test_MM():
    mm = MM(alpha, A, B, beta, C)
    assert mm.inputs  == (alpha, A, B, beta, C)
    assert mm.outputs == (alpha*A*B + beta*C,)

def test_SV():
    sv = SV(S, y)
    assert sv.outputs == (S.I*y,)

def test_composite():
    A,B,C = [MatrixSymbol(s, n, n) for s in 'ABC']
    mm = MM(alpha, A, B, beta, C)
    sv = SV(alpha*A*B + beta*C, y)
    cc = CompositeComputation((mm, sv))

    assert set(cc.inputs)  == set((alpha, A, B, beta, C, y))
    assert set(cc.outputs) == set(((alpha*A*B + beta*C).I*y,))
    assert cc.dag_io() == {mm: set([sv]), sv: set([])}
    assert cc.dag_oi() == {sv: set([mm]), mm: set([])}
    assert cc.toposort() == [mm, sv]

def test_GEMM():
    A = MatrixSymbol('A', m, k)
    B = MatrixSymbol('B', k, n)
    C = MatrixSymbol('C', m, n)
    assert GEMM(alpha, A,   B, beta, C).print_Fortran(str) == \
            "GEMM('N', 'N', m, n, k, alpha, A, m, B, k, beta, C, m)"

    D = MatrixSymbol('D', k, m)
    assert GEMM(alpha, D.T, B, beta, C).print_Fortran(str) == \
            "GEMM('T', 'N', m, n, k, alpha, D, k, B, k, beta, C, m)"

def test_SYMM():
    A = MatrixSymbol('A', m, m)
    B = MatrixSymbol('B', m, n)
    C = MatrixSymbol('C', m, n)
    assert SYMM(alpha, A, B, beta, C).print_Fortran(str, Q.symmetric(A)) == \
            "SYMM('L', 'U', m, n, alpha, A, m, B, m, beta, C, m)"

def test_TRMM():
    A = MatrixSymbol('A', m, m)
    B = MatrixSymbol('B', m, n)
    assert TRMM(alpha, A, B).print_Fortran(str, Q.upper_triangular(A)) == \
            "TRMM('L', 'U', 'N', 'N', m, n, alpha, A, m, B, m)"

def test_TRSV():
    A = MatrixSymbol('A', m, m)
    x = MatrixSymbol('x', m, 1)
    assert TRSV(A, x).print_Fortran(str, Q.upper_triangular(A)) == \
            "TRSV('U', 'N', 'N', m, A, m, x, 1)"
