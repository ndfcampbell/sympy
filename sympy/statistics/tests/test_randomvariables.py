from sympy import EmptySet, FiniteSet, S
from sympy.statistics.randomvariable import (ProbabilitySpace,
        Event, Die, Bernoulli)

def test_finite_events():
    # Tests with dice

    d1, d2, d3 = Die(), Die(), Die()
    # Events for a die being even
    e1, e2, e3 = [Event(d, FiniteSet(2,4,6)) for d in (d1,d2,d3)]
    # Events for a die being even
    o1, o2, o3 = [Event(d, FiniteSet(1,3,5)) for d in (d1,d2,d3)]

    assert (e1 & e2).measure == S(1)/4
    assert (e1 & e2 & e3).measure == S(1)/8
    assert (e1 | e2).measure == S(3)/4
    assert (o1 | o2).measure == S(3)/4
    # assert (e1 & e2) == (o1 | o2).complement
    assert (e1 | o1) == d1.sample_space_event
    assert (e1 & e1) == e1
    assert (e2 | e2) == e2
    assert (o1 & e1) == Event(d1, S.EmptySet)

    # Tests with coins

    coin1 = Bernoulli(a='H', b='T')
    coin2 = Bernoulli(a='H', b='T')

    coin1_is_heads = Event(coin1, FiniteSet('H'))
    coin2_is_anything = Event(coin2, FiniteSet('H', 'T'))
    assert coin1_is_heads.measure == S.Half
    assert coin2_is_anything.measure == 1
    assert (coin1_is_heads & coin2_is_anything).measure == S.Half
    assert (coin1_is_heads | coin2_is_anything).measure == 1

    assert coin1_is_heads.complement == Event(coin1, FiniteSet('T'))
    assert (coin1_is_heads & coin1_is_heads.complement ==
            Event(coin1, S.EmptySet))

