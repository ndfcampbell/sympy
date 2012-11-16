from sympy.rules.tools import subs
from sympy import Basic

def test_subs():
    from sympy import symbols
    a,b,c,d,e,f = symbols('a,b,c,d,e,f')
    mapping = {a: d, d: a, Basic(e): Basic(f)}
    expr   = Basic(a, Basic(b, c), Basic(d, Basic(e)))
    result = Basic(d, Basic(b, c), Basic(a, Basic(f)))
    assert subs(mapping)(expr) == result

def test_subs_no_repeat():
    expr     = Basic(1, Basic(2, 3))
    d = {Basic(2, 3): Basic(4, 5), 5: 6}
    expected = Basic(1, Basic(4, 5))  # note that the 5 is not a 6
    assert subs(d)(expr) == expected

def test_subs_empty():
    assert subs({})(2) is 2
