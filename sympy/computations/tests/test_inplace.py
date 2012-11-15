from sympy.computations.inplace import (make_getname, Copy, inplace,
        purify_one, CopyComp, make_idinc)

from sympy import Symbol, symbols
from sympy.computations.example import inc, minmax

a,b,c,x,y,z = symbols('a,b,c,x,y,z')

def test_getname():
    getname = make_getname()
    print getname(Symbol('x'))
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

def test_purify_one():
    assert purify_one(inc(3)) == inc(3)
    assert purify_one(inc_inplace(3)) == inc_inplace(3) + CopyComp(3, 1)

    assert purify_one(minmax(x, y)) == minmax(x, y)
    assert purify_one(minmax_inplace(x, y)) == (minmax_inplace(x, y) +
            CopyComp(x, 1) + CopyComp(y, 1))
