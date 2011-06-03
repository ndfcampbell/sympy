from sympy import EmptySet, FiniteSet, S
from sympy.statistics.randomvariable import ProbabilitySpace, ProbabilityMeasure, Event, Die

def test_finite_events():
    d1,d2,d3 = Die(), Die(), Die()
    # Events for a die being even
    e1,e2,e3 = [Event(d, FiniteSet(2,4,6)) for d in (d1,d2,d3)]
    # Events for a die being even
    o1,o2,o3 = [Event(d, FiniteSet(1,3,5)) for d in (d1,d2,d3)]

    assert (e1 & e2).measure == S(1)/4
    assert (e1 & e2 & e3).measure == S(1)/8
    assert (e1 | e2).measure == S(3)/4
    assert (o1 | o2).measure == S(3)/4
    # assert (e1 & e2) == (o1 | o2).complement
    assert (e1 | o1) == d1.sample_space_event
    assert (e1 & e1) == e1
    assert (e2 | e2) == e2
    assert (o1 & e1) == Event(d1, S.EmptySet)


