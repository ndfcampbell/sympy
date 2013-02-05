from sympy.unify.rewrites import crl, rewriterules
from sympy import *
from sympy.abc import x, y, z
from sympy.unify.usympy import construct, deconstruct
from sympy.unify.core import Variable

def test_crl():

    assert list(crl(0, 1, [])(0)) == [1]
    assert list(crl(0, 1, [])(1)) == []
    assert list(crl(Variable('x'), 1, [Variable('x')])(5)) == [1]

def test_crl_sympy():
    src, tgt, vs, cond = Basic(x), Basic(True), [x], lambda x: x < 10
    pat2 = deconstruct(src, vs), deconstruct(tgt, vs), map(Variable, vs), cond
    expr = deconstruct(Basic(3))

    assert list(crl(*pat2)(expr)) == [deconstruct(Basic(True))]

    expr = deconstruct(Basic(13))
    assert not list(crl(*pat2)(expr))

def test_rewriterules():
    pat1 = x, 'foo', [x], lambda x: isinstance(x, int) and x < 10
    pat2 = x, 'bar', [x], lambda x: isinstance(x, int) and x > 10
    f = rewriterules(*zip(pat1, pat2))

    assert list(f('a')) == ['a']
    assert list(f(3)) == ['foo']
    assert list(f(14)) == ['bar']

def test_rewriterules_sympy():
    pat1 = Basic(x), Basic(y), [x], lambda x: x.is_integer and x < 10
    pat2 = Basic(x), Basic(z), [x], lambda x: x.is_integer and x > 10
    f = rewriterules(*zip(pat1, pat2))

    assert list(f('a')) == ['a']

    assert list(f(Basic(S(3)))) == [Basic(y)]
    assert list(f(Basic(S(14)))) == [Basic(z)]

def test_rewriterules_sympy_nested():
    pat1 = Basic(1, Basic(x)), Basic(2, Basic(x)), [x], None
    pat2 = Basic(2, Basic(x)), Basic(3, Basic(x)), [x], None
    f = rewriterules(*zip(pat1, pat2))

    assert list(f('a')) == ['a']
    assert list(f(Basic(1, Basic(4)))) == [Basic(3, Basic(4))]

def test_strategy():
    from sympy.rules.branch import multiplex
    pat1 = Basic(1, Basic(x)), Basic(2, Basic(x)), [x], None
    pat2 = Basic(2, Basic(x)), Basic(3, Basic(x)), [x], None
    strategy = lambda rules: multiplex(*rules)
    f = rewriterules(*zip(pat1, pat2), strategy=strategy)

    assert list(f(Basic(1, Basic(4)))) == [Basic(2, Basic(4))]

def test_lopgy():
    try:
        import logpy
    except ImportError:
        return
    def construct(x):
        if logpy.core.isvar(x):
            return x.token
        if isinstance(x, tuple):
            return x[0](*map(construct, x[1:]))
        else:
            return x

    def deconstruct(expr, variables):
        if expr in variables:
            return logpy.var(expr)
        if not isinstance(expr, Basic) or expr.is_Atom:
            return expr
        return (type(expr),) + tuple(deconstruct(arg, variables)
                                     for arg in expr.args)
    def unify(a, b, s):
        return logpy.core.eq(a, b)(s)

    def is_leaf(x):
        return not isinstance(x, tuple)
    def children(x):
        return x[1:]
    def new(op, *args):
        return (op,) + tuple(args)
    def op(x):
        return x[0]

    pat1 = Basic(1, Basic(x)), Basic(2, Basic(x)), [x], None
    pat2 = Basic(2, Basic(x)), Basic(3, Basic(x)), [x], None
    f = rewriterules(*zip(pat1, pat2), construct=construct,
                                       deconstruct=deconstruct,
                                       unify=unify,
                                       reify=logpy.core.reify)

    assert list(f('a')) == ['a']
    assert list(f(Basic(1, Basic(4)))) == [Basic(3, Basic(4))]
