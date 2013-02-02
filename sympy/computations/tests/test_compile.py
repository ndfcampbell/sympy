from sympy.computations.example import patterns, inc, add, double, incdec
from sympy.computations.compile import (input_crunch, output_crunch,
        multi_input_rule, multi_output_rule)
from sympy.rules.branch import multiplex, exhaust, debug
from sympy.computations.core import Identity
from sympy import Symbol, symbols, S, Dummy
from sympy.unify import unify, rewriterule, rebuild
from sympy.abc import a, b, c, x, y, z

rules = map(rewriterule, *zip(*patterns))
rule = exhaust(multiplex(*map(input_crunch, rules)))

def test_compile():
    expr = y + 1
    rule = rewriterule(x + 1, inc(x), [x])
    assert list(rule(expr)) == [inc(y)]

def test_input_crunch():
    comp = Identity(y + 1)
    rule = input_crunch(rewriterule(x + 1, inc(x), [x]))
    assert len(list(rule(comp))) == 1

    comp = Identity(y + 3,)
    rule = input_crunch(rewriterule(x + y, add(x, y), [x, y]))
    assert len(list(rule(comp))) == 2

def test_add():
    expr = z + 3
    rule = rewriterule(x + y, add(x, y), [x, y])
    assert set(rule(expr)) == set([add(S(3), z), add(z, S(3))])
    rule = multiplex(*map(input_crunch, rules))
    comp = Identity(expr)
    assert set(rule(comp)) == set([add(S(3), z), add(z, S(3))])

def test_rule():
    z = Symbol('z')
    expr = 2*z + 1
    comp = Identity(expr)
    assert set(rule(comp)) == set([add(2*z, S(1)) + double(z),
                                   add(S(1), 2*z) + double(z),
                                   inc(2*z) + double(z)])

def _test_multi_rule(multi_rule):
    expr = z + 1, z - 1
    comp = Identity(*expr)
    rule = multi_rule((x + 1, x - 1), incdec(x), x)
    assert list(rule(comp)) == [incdec(z)]

def _test_multi_with_extra_inputs(multi_rule):
    expr = a + 1, a - 1, b + 1
    comp = Identity(*expr)
    rule = multi_rule((c + 1, c - 1), incdec(c), c)
    assert list(rule(comp)) == [incdec(a) + Identity(b + 1) ]

def test_multi_input():
    _test_multi_rule(multi_input_rule)
    _test_multi_with_extra_inputs(multi_input_rule)

def test_multi_output():
    _test_multi_rule(multi_output_rule)
    _test_multi_with_extra_inputs(multi_output_rule)
