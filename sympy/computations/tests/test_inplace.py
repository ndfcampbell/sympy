from sympy.computations.inplace import make_getname
from sympy import Symbol

def test_getname():
    getname = make_getname()
    print getname(Symbol('x'))
    assert getname(Symbol('x')) == 'x'
    assert getname(Symbol('y')) == 'y'
    assert getname(Symbol('x', real=True)) == 'x_2'
    assert len(set(map(getname, (1, 2, 2, 2, 3, 3, 4)))) == 4
