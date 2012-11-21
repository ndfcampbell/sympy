from sympy.computations.matrices.compile import patterns
from sympy.computations.compile import input_crunch, brulify
from sympy.rules.branch import multiplex, exhaust, debug
from sympy.computations.core import Identity
from sympy import Symbol, symbols, S
from sympy.unify import patternify, unify, rewriterule, rebuild
from sympy.matrices.expressions import MatrixSymbol

rules = [brulify(source, target, *wilds) for source, target, wilds, _ in
        patterns]
rule = exhaust(multiplex(*map(input_crunch, rules)))

a,b,c,d,e,x,y,z,m,n,l,k = map(Symbol, 'abcdexyzmnlk')

def test_GEMM():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    expr = a*X*Y + b*Z
    comp = Identity(expr)
    print list(rule(comp))
    assert len(list(rule(comp))) != 0
