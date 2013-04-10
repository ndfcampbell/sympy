from sympy.computations.core import Computation, unique, CompositeComputation
from sympy.strategies.rl import flatten, unpack

a,b,c,d,e,f,g,h = 'abcdefgh'

class TComp(Computation):
    """ Test Computation class """
    op      = property(lambda self: self.args[0])
    inputs  = property(lambda self: tuple(self.args[1]))
    outputs = property(lambda self: tuple(self.args[2]))

    def __str__(self):
        ins  = "["+', '.join(self.inputs) +"]"
        outs = "["+', '.join(self.outputs)+"]"
        return "%s -> %s -> %s"%(ins, str(self.op), outs)


def test_flatten():
    MM = TComp('minmax', (a, b), (d, e))
    A =  TComp('foo', (d,), (f,))
    B =  TComp('bar', (a, f), (g, h))
    C =  CompositeComputation(MM, A, B)
    C2 = CompositeComputation(CompositeComputation(MM, A), B)
    assert len(flatten(C2).computations) == 3
    assert C.inputs == C2.inputs
    assert C.outputs == C2.outputs

def test_unpack():
    from sympy import Basic
    A =  TComp('foo', (d,), (f,))
    C =  Basic.__new__(CompositeComputation, A)
    assert unpack(C) == A
