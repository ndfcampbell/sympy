from sympy.computations.matrices.compile import (patterns, make_rule, basetype,
        typecheck)
from sympy.computations.compile import multi_output_rule
from sympy.computations.matrices.lapack import GESV, POSV, IPIV, LASWP
from sympy.computations.matrices.blas import GEMM
from sympy.computations.core import Identity
from sympy import Symbol, symbols, S, Q, Expr
from sympy.matrices.expressions import (MatrixSymbol, MatrixExpr,
        PermutationMatrix)
from sympy.utilities.pytest import XFAIL, slow, skip


a,b,c,d,e,x,y,z,m,n,l,k = map(Symbol, 'abcdexyzmnlk')

def _reduces(expr, inputs, assumptions=True, patterns=patterns):
    rule = make_rule(patterns, assumptions)
    comp = Identity(expr)
    assert any(set(c.inputs).issubset(set(inputs)) for c in rule(comp))

def _reduces_set(exprs, inputs, assumptions=True, patterns=patterns):
    rule = make_rule(patterns, assumptions)
    comp = Identity(*exprs)
    assert any(set(c.inputs).issubset(set(inputs)) for c in rule(comp))

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
    rule = make_rule(patterns, True)
    expr = X*Y*Z
    comp = Identity(expr)
    results = list(rule(comp))
    # We can't do this with a single GEMM
    assert not any(isinstance(r, GEMM) for r in results)

def test_alternative_patterns():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    expr = a*X*Y
    _reduces(expr, (a, X, Y))

def test_SV():
    rule = make_rule(patterns, True)
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    expr = X.I * Y
    comp = Identity(expr)
    results = list(rule(comp))
    assert len(results) != 0
    assert any(result.has(GESV) for result in results)
    assert not any(result.has(POSV) for result in results)

    rule2 = make_rule(patterns, Q.symmetric(X) & Q.positive_definite(X))
    results = list(rule2(comp))
    assert any(result.has(GESV) for result in results)
    assert any(result.has(POSV) for result in results)

def test_non_trivial():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    expr = (a*X*Y + b*Z).I*Z
    assumptions = Q.positive_definite(a*X*Y + b*Z) & Q.symmetric(a*X*Y + b*Z)
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
    _reduces(expr, (X, Y, Z, W), True)

def test_transpose_inputs():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    expr = X*Y.T
    _reduces(expr, (X, Y), True)

def test_GEMM_coefficients():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    exprs = (3*X*Y + 2*Z, X*Y + 2*Z, 3*X*Y + Z, X*Y + Z, X*Y)
    rule = make_rule(patterns, True)
    assert all(isinstance(next(rule(Identity(expr))), GEMM) for expr in exprs)

def test_multi_output_rule():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    rule = multi_output_rule((IPIV(Y), PermutationMatrix(IPIV(Y))*Y),
            LASWP(PermutationMatrix(IPIV(Y))*Y, IPIV(Y)), Y)
    comp = Identity(IPIV(X), PermutationMatrix(IPIV(X))*X)
    assert len(list(rule(comp))) != 0

def test_LASWP():
    X = MatrixSymbol('X', 3, 3)
    exprs = IPIV(X), PermutationMatrix(IPIV(X))*X
    outputs = (X,)
    rule = make_rule(patterns, True)
    comp = Identity(*exprs)
    return any(set(c.outputs).issubset(set(outputs)) for c in rule(comp))

def test_XinvY():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    expr = X.I*Y
    _reduces(expr, (X, Y), True)
