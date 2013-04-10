from sympy.computations.dot import gen_dot
from sympy.computations.core import CompositeComputation
a,b,c,d,e,f,g,h = 'abcdefgh'

def test_dot():
    from sympy.computations.core import OpComp
    MM = OpComp('minmax', (a, b), (d, e))
    A =  OpComp('foo', (d,), (f,))
    B =  OpComp('bar', (a, f), (g, h))
    C =  CompositeComputation(MM, A, B)

    assert isinstance(gen_dot(MM), str)
    assert isinstance(gen_dot(C), str)
