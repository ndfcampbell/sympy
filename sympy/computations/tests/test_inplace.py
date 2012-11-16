from sympy.computations.inplace import (make_getname, Copy, inplace,
        purify_one, CopyComp, make_idinc, tokenize_one, ExprToken, tokenize)

from sympy import Symbol, symbols
from sympy.computations.example import inc, minmax

a,b,c,x,y,z = symbols('a,b,c,x,y,z')

def test_getname():
    getname = make_getname()
    assert getname(Symbol('x')) == 'x'
    assert getname(Symbol('y')) == 'y'
    assert getname(Symbol('x', real=True)) == 'x_2'
    assert len(set(map(getname, (1, 2, 2, 2, 3, 3, 4)))) == 4

def test_copy():
    c = Copy(Symbol('x'))
    assert c.name == 'x'

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
