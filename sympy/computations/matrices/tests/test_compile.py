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
    print results
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
