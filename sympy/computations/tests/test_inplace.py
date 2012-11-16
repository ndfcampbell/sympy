from sympy.computations.inplace import (make_getname, Copy, inplace,
        purify_one, make_idinc, tokenize_one, ExprToken, tokenize,
        copies_one, purify, OpComp, inplace_tokenize)
from sympy.computations.core import CompositeComputation

from sympy import Symbol, symbols
from sympy.computations.example import inc, minmax

a,b,c,x,y,z = symbols('a,b,c,x,y,z')

def test_getname():
    getname = make_getname()
    assert getname(Symbol('x')) == 'x'
    assert getname(Symbol('y')) == 'y'
    assert getname(Symbol('x', real=True)) == 'x_2'
    assert getname((Symbol('x'), 'foo'), 'x') != 'x'
    assert len(set(map(getname, (1, 2, 2, 2, 3, 3, 4)))) == 4

class inc_inplace(inc):
    inplace = {0: 0}

class minmax_inplace(minmax):
    inplace = {0: 0, 1: 1}

def test_inplace():
    assert inplace(inc(3)) == {}
    assert inplace(inc_inplace(3)) == {0: 0}


def test_idinc():
    idinc = make_idinc()
    assert idinc(1) == 1
    assert idinc(1) == 2
    assert idinc(1) == 3
    assert idinc(2) == 1

def test_tokenize_one():
    comp = tokenize_one(inc(3))
    assert comp.op == inc
    assert comp.inputs[0].expr == 3
    assert comp.outputs[0].expr == 4

def test_tokenize():
    comp = tokenize(inc(3))
    assert comp.op == inc
    assert comp.inputs[0].expr == 3
    assert comp.outputs[0].expr == 4

    comp2 = tokenize(inc(3) + inc(4))
    assert len(comp2.computations) == 2
    assert comp2.inputs[0].expr == 3
    assert comp2.outputs[0].expr == 5

def test_copies_one():
    tokenizer = make_getname()
    comp = tokenize(inc(3), tokenizer)
    assert copies_one(comp, tokenizer) == []

    comp = tokenize(inc_inplace(3), tokenizer)
    copy = copies_one(comp, tokenizer)[0]
    assert copy.op == Copy
    assert copy.inputs[0].expr == comp.inputs[0].expr
    assert copy.inputs[0].token == comp.inputs[0].token
    assert copy.outputs[0].token != comp.inputs[0].token

    comp = tokenize(minmax_inplace(x, y), tokenizer)
    assert len(copies_one(comp)) == 2

def test_purify_one():
    tokenizer = make_getname()
    comp = tokenize(inc(3), tokenizer)
    assert purify_one(comp, tokenizer) == comp

    comp = tokenize(inc_inplace(3), tokenizer)
    purecomp = purify_one(comp, tokenizer)
    assert len(purecomp.computations) == 2
    a, b = purecomp.computations
    cp, inci = (a, b) if a.op==Copy else (b, a)
    assert cp.outputs == inci.inputs
    assert cp.inputs == comp.inputs
    assert inci.outputs == comp.outputs
    assert purecomp.inputs == comp.inputs
    assert purecomp.outputs == comp.outputs

    comp = tokenize(minmax_inplace(x, y), tokenizer)
    assert len(purify_one(comp, tokenizer).computations) == 3

def test_purify():
    assert purify(tokenize(inc(3))) == tokenize(inc(3))
    assert purify(tokenize(inc_inplace(3))) == \
            purify_one(tokenize(inc_inplace(3)))

    tokenizer = make_getname()
    comp = tokenize(inc(3) + inc_inplace(4) + inc_inplace(5), tokenizer)
    purecomp = purify(comp, tokenizer)

    assert len(purecomp.computations) == 5
    assert purecomp.inputs == comp.inputs
    assert purecomp.outputs == comp.outputs

def test_inplace_tokenize():
    comp     = OpComp(inc_inplace, (ExprToken(1, 1),), (ExprToken(2, 2),))
    expected = OpComp(inc_inplace, (ExprToken(1, 1),), (ExprToken(2, 1),))
    assert inplace_tokenize(comp) == expected

    comp     = (OpComp(inc_inplace, (ExprToken(1, 1),), (ExprToken(2, 2),)) +
                OpComp(inc_inplace, (ExprToken(2, 2),), (ExprToken(3, 3),)))
    expected = (OpComp(inc_inplace, (ExprToken(1, 1),), (ExprToken(2, 1),)) +
                OpComp(inc_inplace, (ExprToken(2, 1),), (ExprToken(3, 1),)))
    assert inplace_tokenize(comp) == expected
