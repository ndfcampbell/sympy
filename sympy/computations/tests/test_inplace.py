from sympy.computations.inplace import (make_getname, Copy, inplace,
        purify_one, tokenize_one, ExprToken, tokenize,
        copies_one, purify, inplace_tokenize, remove_single_copies,
        inplace_compile)
from sympy.computations.core import CompositeComputation, OpComp

from sympy import Symbol, symbols
from sympy.computations.example import inc, minmax, flipflop
from sympy.rules.strat_pure import debug

a,b,c,x,y,z = symbols('a,b,c,x,y,z')

class inci(inc):
    inplace = {0: 0}

class minmax_inplace(minmax):
    inplace = {0: 0, 1: 1}

class flipflopi(flipflop):
    inplace = {0: 0, 1: 1}

def test_getname():
    getname = make_getname()
    assert getname(Symbol('x')) == 'x'
    assert getname(Symbol('y')) == 'y'
    assert getname(Symbol('x', real=True)) == 'x_2'
    assert getname((Symbol('x'), 'foo'), 'x') != 'x'
    assert len(set(map(getname, (1, 2, 2, 2, 3, 3, 4)))) == 4

def test_inplace():
    assert inplace(inc(3)) == {}
    assert inplace(inci(3)) == {0: 0}

def test_tokenize_one():
    comp = tokenize_one(inc(3), make_getname())
    assert comp.op == inc
    assert comp.inputs[0].expr == 3
    assert comp.outputs[0].expr == 4

def test_tokenize():
    comp = tokenize(inc(3), make_getname())
    assert comp.op == inc
    assert comp.inputs[0].expr == 3
    assert comp.outputs[0].expr == 4

    comp2 = tokenize(inc(3) + inc(4), make_getname())
    assert len(comp2.computations) == 2
    assert comp2.inputs[0].expr == 3
    assert comp2.outputs[0].expr == 5

def test_copies_one():
    tokenizer = make_getname()
    comp = tokenize(inc(3), tokenizer)
    assert copies_one(comp, tokenizer) == []

    comp = tokenize(inci(3), tokenizer)
    copy = copies_one(comp, tokenizer)[0]
    assert copy.op == Copy
    assert copy.inputs[0].expr == comp.inputs[0].expr
    assert copy.inputs[0].token == comp.inputs[0].token
    assert copy.inputs[0].token != copy.outputs[0].token
    assert copy.outputs[0].token != comp.inputs[0].token

    comp = tokenize(minmax_inplace(x, y), tokenizer)
    assert len(copies_one(comp, tokenizer)) == 2

def test_purify_one():
    tokenizer = make_getname()
    comp = tokenize(inc(3), tokenizer)
    assert purify_one(comp, tokenizer) == comp

    comp = tokenize(inci(3), tokenizer)
    purecomp = purify_one(comp, tokenizer)
    assert len(purecomp.computations) == 2
    a, b = purecomp.computations
    cp, incinpl = (a, b) if a.op==Copy else (b, a)
    assert cp.outputs == incinpl.inputs
    assert cp.inputs == comp.inputs
    assert incinpl.outputs == comp.outputs
    assert purecomp.inputs == comp.inputs
    assert purecomp.outputs == comp.outputs

    comp = tokenize(minmax_inplace(x, y), tokenizer)
    assert len(purify_one(comp, tokenizer).computations) == 3

def test_purify():
    tokenizer = make_getname()
    assert purify(tokenize(inc(3), tokenizer), tokenizer) == \
            tokenize(inc(3), tokenizer)
    assert purify(tokenize(inci(3), tokenizer), tokenizer) == \
            purify_one(tokenize(inci(3), tokenizer), tokenizer)

    tokenizer = make_getname()
    comp = tokenize(inc(3) + inci(4) + inci(5), tokenizer)
    purecomp = purify(comp, tokenizer)

    assert len(purecomp.computations) == 5
    assert purecomp.inputs == comp.inputs
    assert purecomp.outputs == comp.outputs

def test_inplace_tokenize():
    comp     = OpComp(inci, (ExprToken(1, 1),), (ExprToken(2, 2),))
    expected = OpComp(inci, (ExprToken(1, 1),), (ExprToken(2, 1),))
    assert inplace_tokenize(comp) == expected

    comp     = (OpComp(inci, (ExprToken(1, 1),), (ExprToken(2, 2),)) +
                OpComp(inci, (ExprToken(2, 2),), (ExprToken(3, 3),)))
    expected = (OpComp(inci, (ExprToken(1, 1),), (ExprToken(2, 1),)) +
                OpComp(inci, (ExprToken(2, 1),), (ExprToken(3, 1),)))
    assert inplace_tokenize(comp) == expected

def test_remove_single_copies():
    comp     = (OpComp(inci, (ExprToken(1, '1'),), (ExprToken(2, '2'),)) +
                OpComp(Copy, (ExprToken(1, '0'),), (ExprToken(1, '1'),)))
    expected =  OpComp(inci, (ExprToken(1, '0'),), (ExprToken(2, '2'),))
    assert remove_single_copies(comp) == expected

def test_integrative():
    from sympy import Basic
    from sympy.unify import patternify, unify
    comp = inci(x) + flipflopi(x+1, y)

    expected = (OpComp(inci, (ExprToken(x, 'x'),), (ExprToken(x+1, 'x'),)) +
                OpComp(flipflopi, (ExprToken(x+1, 'x'), ExprToken(y, 'y')),
                                  (ExprToken(Basic(x+1, y), 'x'),
                                   ExprToken(Basic(y, x+1), 'y'))))

    assert inplace_compile(comp) == expected

    comp = inci(x) + flipflopi(x+1, y) + inc(y)

    # We don't care about the variable names used. Let them be anything.
    expected = patternify(
                OpComp(inci, (ExprToken(x, 'x'),), (ExprToken(x+1, 'x'),)) +
                OpComp(inc , (ExprToken(y, 'y'),), (ExprToken(y+1, '_1'),)) +
                OpComp(Copy, (ExprToken(y, 'y'),), (ExprToken(y, '_2'),)) +
                OpComp(flipflopi, (ExprToken(x+1, 'x'), ExprToken(y, '_2')),
                                  (ExprToken(Basic(x+1, y), 'x'),
                                   ExprToken(Basic(y, x+1), '_2'))),
                '_1', '_2')

    assert len(list(unify(inplace_compile(comp), expected))) > 0