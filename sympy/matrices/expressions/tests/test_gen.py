from sympy.matrices.expressions.gen import build_rule, top_down
# from sympy.matrices.expressions.gen import blas_rule
from sympy.matrices.expressions.blas import TRSV, GEMM, SYMM
from sympy.matrices.expressions import MatrixSymbol
from sympy import Q

X = MatrixSymbol('X', 3, 3)
Y = MatrixSymbol('Y', 3, 3)
Z = MatrixSymbol('Z', 3, 3)
x = MatrixSymbol('x', 3, 1)

def test_rr_from_blas():
    assumptions = (Q.lower_triangular(X) & Q.lower_triangular(Y) &
                   Q.lower_triangular(Z))
    expr = (3*X*Y + 2*Z).I*x
    rr = rr_from_blas(TRSV, assumptions)
    assert rr(expr).next() == TRSV(3*X*Y + 2*Z, x)

    rr = rr_from_blas(TRSV)
    assert len(rr(expr)) == 0

def test_blas_rule():
    assumptions = (Q.lower_triangular(X) & Q.lower_triangular(Y) &
                   Q.lower_triangular(Z))
    translate = build_rule(assumptions)

    assert list(translate(2*Z+3*X*Y)) == [GEMM(3, X, Y, 2, Z)]


    expr = (3*X*Y + 2*Z).I*x
    comp = blas_rule(expr).next()
    assert isinstance(comp, TRSV)
    comp2 = blas_rule(comp.inputs[0]).next()
    assert isinstance(comp2, GEMM)

def test_multiple_outs():
    expr = (3*X*Y + 2*Z)
    assumptions = Q.symmetric(X)
    blas_rule = build_rule(assumptions)
    assert set(map(type, blas_rule(expr))) == {SYMM, GEMM}

    assumptions = True
    blas_rule = build_rule(assumptions)
    assert set(map(type, blas_rule(expr))) == {GEMM}

def test_traverse():
    assumptions = (Q.lower_triangular(3*X*Y+2*Z) & Q.symmetric(Y))
    expr = (3*X*Y + 2*Z).I*x
    blas_rule = top_down(build_rule(assumptions))
    return blas_rule(expr)
    comp = blas_rule(expr).next()
    return comp
    assert False