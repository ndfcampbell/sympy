from sympy.unify.rewrite import rewriterule
from sympy.unify.unify_sympy import rebuild, patternify
from sympy.rules.branch import canon
from sympy import sin, cos
from sympy.abc import x, y, z

def test_sincos_canon():
    rr = rewriterule(patternify(sin(x)**2 + cos(x)**2 + y, x, y), 1 + y)
    brl = canon(rr)

    expr = 4*(sin(x+y)**2 + cos(x+y)**2 + z*x)**3
    assert rebuild(brl(expr).next()) == 4*(1 + z*x)**3
