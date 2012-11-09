from sympy.matrices.expressions.gen import build_rule, top_down, rr_from_blas
# from sympy.matrices.expressions.gen import blas_rule
from sympy.matrices.expressions.blas import TRSV, GEMM, SYMM
from sympy.matrices.expressions.matcomp import basic_names
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
    assert len(list(rr(expr))) == 0

def test_blas_rule():
    assumptions = (Q.lower_triangular(X) & Q.lower_triangular(Y) &
                   Q.lower_triangular(Z))
    translate = build_rule(assumptions)

    assert list(translate(2*Z+3*X*Y)) == [GEMM(3, X, Y, 2, Z)]


    expr = (3*X*Y + 2*Z).I*x
    comp = translate(expr).next()
    assert isinstance(comp, TRSV)
    comp2 = translate(comp.inputs[0]).next()
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
    comps = list(blas_rule(expr))
    assert len(comps) > 1
    assert all('X, Y, Z, x' in comp.header(str) for comp in comps)
    assert all(expr in comp.outputs for comp in comps)
    assert all(callable(c.build(basic_names, assumptions)) for c in comps)

def test_build_many():
    assumptions = (Q.lower_triangular(3*X*Y+2*Z) & Q.symmetric(Y))
    expr = (3*X*Y + 2*Z).I*x
    blas_rule = top_down(build_rule(assumptions))
    comps = list(blas_rule(expr))

    import numpy as np
    A,B,C = [np.asarray([[1,0,0],[4,5,0],[7,8,9]], order='F', dtype='float64')
            for i in range(3)]
    xx     = np.asarray([1,2,3], order='F', dtype='float64')
    for comp in comps:
        f = comp.build(basic_names, assumptions)
        assert callable(f)
        f(A, B, C, xx) # Ensure that this works

def test_transpose_posv():
    assumptions = (Q.positive_definite(X) & Q.positive_definite(Z) &
                   Q.symmetric(Z))
    target_expression = (3*X*X.T + 2*Z).I*X
    blas_rule = top_down(build_rule(assumptions))
    computations = list(top_down(build_rule(assumptions))(target_expression))
    c = computations[0]
    assert c.inputs == (X, Z)
    assert target_expression in c.outputs
    src = c.print_Fortran(str, assumptions)
    assert '_1' not in src
    print src
    assert "call dgemm('N', 'T', 3, 3, 3, 3, X, 3, X, 3, 2, Z, 3)" in src
    assert "call dposv(U, 3, 3, Z, 3, X, 3, INFO)" in src
