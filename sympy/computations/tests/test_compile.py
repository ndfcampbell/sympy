from sympy.computations.example import patterns, inc, add, double
from sympy.computations.compile import input_crunch, output_crunch, brulify
from sympy.rules.branch import multiplex, exhaust, debug
from sympy.computations.core import Identity
from sympy import Symbol, symbols, S
from sympy.unify import patternify, unify, rewriterule, rebuild

rules = [brulify(*pattern) for pattern in patterns]
rule = exhaust(multiplex(*map(input_crunch, rules)))

x, y, z = symbols('x,y,z')

def test_compile():
    expr = y + 1
    rule = brulify(x + 1, inc(x), x)
    assert list(rule(expr)) == [inc(y)]

def test_input_crunch():
    comp = Identity(y + 1)
    rule = input_crunch(brulify(x + 1, inc(x), x))
    assert len(list(rule(comp))) == 1

    comp = Identity(y + 3,)
    rule = input_crunch(brulify(x + y, add(x, y), x, y))
    assert len(list(rule(comp))) == 2

def test_add():
    expr = y + 3
    rule = brulify(x + y, add(x, y), x, y)
    assert set(rule(expr)) == set([add(S(3), y), add(y, S(3))])
    rule = multiplex(*map(input_crunch, rules))
    comp = Identity(expr)
    assert set(rule(comp)) == set([comp + add(S(3), y), comp + add(y, S(3))])

def test_rule():
    y = Symbol('y')
    expr = 2*y + 1
    comp = Identity(expr)
    assert set(rule(comp)) == set([comp + add(2*y, S(1)) + double(y),
                                   comp + add(S(1), 2*y) + double(y),
                                   comp + inc(2*y) + double(y)])
