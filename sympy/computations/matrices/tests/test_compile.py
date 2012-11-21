from sympy.computations.matrices.compile import patterns, make_rule
from sympy.computations.matrices.lapack import GESV, POSV
from sympy.computations.core import Identity
from sympy import Symbol, symbols, S, Q
from sympy.matrices.expressions import MatrixSymbol


a,b,c,d,e,x,y,z,m,n,l,k = map(Symbol, 'abcdexyzmnlk')

def test_GEMM():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    expr = a*X*Y + b*Z
    comp = Identity(expr)
    rule = make_rule(patterns, True)
    results = list(rule(comp))
    assert len(results) != 0

def test_alternative_patterns():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    expr = a*X*Y
    comp = Identity(expr)
    rule = make_rule(patterns, True)
    results = list(rule(comp))
    assert len(results) != 0

def test_SV():
    rule = make_rule(patterns, True)
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    expr = X.I * Y
    comp = Identity(expr)
    results = list(rule(comp))
    assert len(results) != 0
    comptypes = map(type, results)
    assert GESV in comptypes
    assert POSV not in comptypes

    rule2 = make_rule(patterns, Q.symmetric(X) & Q.positive_definite(X))
    results = list(rule2(comp))
    comptypes = map(type, results)
    assert GESV in comptypes
    assert POSV in comptypes

def test_non_trivial():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    expr = (a*X*Y + b*Z).I*Z
    comp = Identity(expr)
    rule = make_rule(patterns, True)
    results = list(rule(comp))
    assert len(results) != 0

def _reduces(expr, inputs, assumptions=True, patterns=patterns):
    rule = make_rule(patterns, assumptions)
    comp = Identity(expr)
    return any(set(c.inputs).issubset(set(inputs)) for c in rule(comp))

def test_XYZ():
    W = MatrixSymbol('W', 3, 3)
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    assert _reduces(X*Y, (X, Y))
    assert _reduces(X*Y*Z, (X, Y, Z))
    assert _reduces(W*X*Y*Z, (W, X, Y, Z))

def test_XYinvZ():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    print _reduces(X*Y.I*Z, (X, Y, Z))
    assert _reduces(X*Y.I*Z, (X, Y, Z))


def test_more_non_trivial():
    W = MatrixSymbol('X', 3, 3)
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    expr = (a*X*Y*Z*Y.I + b*Z*Y + c*W*W).I*(Z*W)
    comp = Identity(expr)
    rule = make_rule(patterns, True)
    results = list(rule(comp))

    assert len(results) != 0
