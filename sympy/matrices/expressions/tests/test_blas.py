from sympy.matrices.expressions.blas import *
from sympy import *
from sympy.utilities.pytest import XFAIL

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
    assert GEMM(alpha, A,   B, beta, C, typecode='D').calls(str) == \
            ["call dgemm('N', 'N', m, n, k, alpha, A, m, B, k, beta, C, m)"]

    D = MatrixSymbol('D', k, m)
    assert GEMM(alpha, D.T, B, beta, C, typecode='S').calls(str) == \
            ["call sgemm('T', 'N', m, n, k, alpha, D, k, B, k, beta, C, m)"]

def test_SYMM():
    A = MatrixSymbol('A', m, m)
    B = MatrixSymbol('B', m, n)
    C = MatrixSymbol('C', m, n)
    assert SYMM(alpha, A, B, beta, C).calls(str, Q.symmetric(A)) == \
            ["call dsymm('L', 'U', m, n, alpha, A, m, B, m, beta, C, m)"]

def test_TRMM():
    A = MatrixSymbol('A', m, m)
    B = MatrixSymbol('B', m, n)
    assert TRMM(alpha, A, B).calls(str, Q.upper_triangular(A)) == \
            ["call dtrmm('L', 'U', 'N', 'N', m, n, alpha, A, m, B, m)"]

def test_TRSV():
    A = MatrixSymbol('A', m, m)
    x = MatrixSymbol('x', m, 1)
    assert TRSV(A, x).calls(str, Q.upper_triangular(A)) == \
            ["call dtrsv('U', 'N', 'N', m, A, m, x, 1)"]

def test_inplace_fn():
    A = MatrixSymbol('A', m, k)
    B = MatrixSymbol('B', k, n)
    C = MatrixSymbol('C', m, n)
    gemm = GEMM(alpha, A,   B, beta, C)
    assert gemm.outputs == (alpha*A*B + beta*C,)
    assert gemm.inplace_fn()(gemm).outputs == (C,)

def test_declarations():
    A = MatrixSymbol('A', m, k)
    B = MatrixSymbol('B', k, n)
    C = MatrixSymbol('C', m, n)
    gemm = GEMM(alpha, A, B, beta, C)
    expected = ["real*8, intent(in) :: A(m, k)",
                "real*8, intent(in) :: B(k, n)",
                "real*8, intent(inout) :: C(m, n)",
                "real*8, intent(in) :: alpha",
                "real*8, intent(in) :: beta",
                "integer, intent(in) :: k",
                "integer, intent(in) :: m",
                "integer, intent(in) :: n",]
    assert set(expected) == set(gemm.declarations(str))

def test_gemm_trsv():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    C = MatrixSymbol('C', n, n)
    x = MatrixSymbol('x', n, 1)
    context = (Q.lower_triangular(A) &
               Q.lower_triangular(B) &
               Q.lower_triangular(C))
    expr = (alpha*A*B + beta*C).I*x
    gemm = GEMM(alpha, A, B, beta, C)
    trsv = TRSV(alpha*A*B + beta*C, x)
    comp = MatrixRoutine((gemm, trsv), (alpha, A, B, beta, C, x), (expr,))
    comp.declarations(str)

    calls = ["call dgemm('N', 'N', n, n, n, alpha, A, n, B, n, beta, C, n)",
             "call dtrsv('L', 'N', 'N', n, C, n, x, 1)"]
    assert comp.outputs == (expr,)
    assert comp.shapes()[x] == x.shape
    assert all(q in comp.shapes() for q in (A,B,C,x,expr,))
    assert not any(q in comp.shapes() for q in (alpha, beta, n))
    assert comp.types()[n] == 'integer'
    assert comp.types()[A] == basetypes[gemm.typecode]
    assert comp.intents()[x] == comp.intents()[C] == 'inout'
    assert 'real*8, intent(in) :: alpha' in comp.declarations(str)
    assert ':: A(n, n)' in '\n'.join(comp.declarations(str))
    assert 'intent(inout) :: x(n, 1)' in '\n'.join(comp.declarations(str))
    assert set(calls) == set(comp.calls(str, context))
    assert set(comp.dimensions()) == set([n])
    assert 'integer, intent(in) :: n' in comp.declarations(str)
