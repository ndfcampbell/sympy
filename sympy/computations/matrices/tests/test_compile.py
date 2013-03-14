from sympy.computations.matrices.compile import (patterns, basetype,
        typecheck)
from sympy.computations.matrices.compile import compile
from sympy.computations.compile import multi_output_rule
from sympy.computations.matrices.lapack import GESV, POSV, IPIV, LASWP
from sympy.computations.matrices.blas import GEMM
from sympy import Symbol, symbols, S, Q, Expr
from sympy.matrices.expressions import (MatrixSymbol, MatrixExpr,
        PermutationMatrix)
from sympy.utilities.pytest import XFAIL, slow, skip
from sympy.assumptions import assuming


a,b,c,d,e,x,y,z,m,n,l,k = map(Symbol, 'abcdexyzmnlk')

def rule(*exprs):
    return compile(Identity(*exprs))

def _reduces(expr, inputs, assumptions=()):
    with assuming(*assumptions):
        assert any(set(c.inputs).issubset(set(inputs))
                for c in compile(inputs, [expr]))

def _reduces_set(exprs, inputs, assumptions=()):
    with assuming(assumptions):
        assert any(set(c.inputs).issubset(set(inputs)) for c in rule(*exprs))

def test_typecheck():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    check = typecheck([a, X, Y])
    assert check(b, Y, Z)
    assert not check(X, a, Z)

def test_GEMM():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    expr = a*X*Y + b*Z
    _reduces(expr, (X, Y, Z, a, b))

def test_basetype():
    x = Symbol('x')
    X = MatrixSymbol('X', 3, 3)
    assert basetype(2*X) == MatrixExpr
    assert basetype(x + 3) == Expr

def test_types():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    expr = X*Y*Z

    # We can't do this with a single GEMM
    assert not any(isinstance(r, GEMM) for r in compile([X, Y, Z], [expr]))

def test_alternative_patterns():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    expr = a*X*Y
    _reduces(expr, (a, X, Y))

def test_SV():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    expr = X.I * Y
    results = list(compile((X, Y), (expr,)))
    assert len(results) != 0
    assert any(result.has(GESV) for result in results)
    assert not any(result.has(POSV) for result in results)

    with assuming(Q.symmetric(X), Q.positive_definite(X)):
        results = list(compile((X, Y), (expr,)))
    assert any(result.has(GESV) for result in results)
    assert any(result.has(POSV) for result in results)

def test_GESV():
    X = MatrixSymbol('X', 3, 2)
    y = MatrixSymbol('Y', 3, 1)

    expr = (X*X.T).I * y

    assert next(compile((X, y), (expr,)))


def test_non_trivial():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    expr = (a*X*Y + b*Z).I*Z
    assumptions = (Q.positive_definite(a*X*Y + b*Z), Q.symmetric(a*X*Y + b*Z))
    _reduces(expr, (a, b, X, Y, Z), assumptions)

def test_XYZ():
    W = MatrixSymbol('W', 3, 3)
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    _reduces(X*Y, (X, Y))
    _reduces(X*Y*Z, (X, Y, Z))
    _reduces(W*X*Y*Z, (W, X, Y, Z))

def test_XYinvZ():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    _reduces(X*Y.I*Z, (X, Y, Z))

def _test_large():
    W = MatrixSymbol('X', 3, 3)
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    expr = (a*X*Y*Z*Y.I*Z + b*Z*Y + c*W*W).I*Z*W
    _reduces(expr, (X, Y, Z, W))

def test_transpose_inputs():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    expr = X*Y.T
    _reduces(expr, (X, Y))

def test_GEMM_coefficients():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    exprs = (3*X*Y + 2*Z, X*Y + 2*Z, 3*X*Y + Z, X*Y + Z, X*Y)
    assert all(isinstance(next(compile((X,Y,Z), [expr])), GEMM) for expr in exprs)

def dont_test_multi_output_rule():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    assert len(list(compile([X, Y], [IPIV(X), PermutationMatrix(IPIV(X))*X]))) != 0

def test_LASWP():
    X = MatrixSymbol('X', 3, 3)
    exprs = IPIV(X), PermutationMatrix(IPIV(X))*X
    outputs = (X,)
    return any(set(c.outputs).issubset(set(outputs))
                for c in compile([X], exprs))

def test_XinvY():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    expr = X.I*Y
    _reduces(expr, (X, Y))

def test_alphaABC_GEMM():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    from sympy.computations.matrices.compile import (alpha, A, B, C, GEMM, S,
                makerule)

    pattern = (alpha*A*B + C, GEMM(alpha, A, B, S.One, C), (alpha, A, B, C), True)
    rule = makerule(pattern)
    expr = -X * Y + Z
    assert list(rule(expr))
