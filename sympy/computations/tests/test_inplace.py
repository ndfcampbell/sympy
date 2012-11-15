from sympy.computations.inplace import make_getname, Copy, inplace
from sympy import Symbol
from sympy.computations.example import inc

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

def test_inplace():
    assert inplace(inc(3)) == {}
    assert inplace(inc_inplace(3)) == {0: 0}
