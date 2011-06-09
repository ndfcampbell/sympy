from sympy import EmptySet, FiniteSet, S, Symbol, Interval, exp, erf, sqrt
from sympy.statistics.randomvariables import (ProbabilitySpace,
        ContinuousProbabilitySpace, Event, Die, Bernoulli)

oo = S.Infinity

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

def test_continuous_events():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    pdf = exp(-x**2/2) / sqrt(2*S.Pi)
    C = ContinuousProbabilitySpace(x, pdf = pdf)
    e1, e2 = Event(C,Interval(-1,1)), Event(C,Interval(-2,2))

    assert e1.measure == erf(sqrt(2)/2)
    assert e2.measure == erf(sqrt(2))

    e1 & e2 == e1

    pdf = exp(-y**2/2) / sqrt(2*S.Pi)
    D = ContinuousProbabilitySpace(y, pdf = pdf)
    f1, f2 = Event(D,Interval(-1,1)), Event(D,Interval(-2,2))

    assert C != D
    assert f1 != e1
    assert (f1 & e1).measure == erf(sqrt(2)/2)**2

    U = ContinuousProbabilitySpace(x, pdf = 1, sample_space = Interval(0,1))
    ue1 = Event(U, Interval(0, S.Half))
    ue2 = Event(U, Interval(S(1)/4, S(3)/4))

    assert ue1.measure == ue2.measure == S.Half
    assert (ue1 & ue2).measure == S(1)/4
    assert (ue1 | ue2).measure == S(3)/4

    assert (ue1 & e1).measure == e1.measure / 2


def test_complement():
    x = Symbol('x', real=True)
    U = ContinuousProbabilitySpace(x, pdf = S(1)/10, sample_space = Interval(0,10))
    center = Event(U, Interval(4,6))
    d = Die(6)

    deven = Event(d, FiniteSet(2,4,6))
    done = Event(d, FiniteSet(1))

    assert deven.complement == Event(d, FiniteSet(1,3,5))
    assert center.complement == Event(U, Interval(0,4,False, True)+Interval(6,10, True, False))

    assert all(event.measure + event.complement.measure == 1
            for event in [deven, done, center] )

    assert (deven & center).complement.complement == (deven & center)

    elements = [Event(d*U, FiniteSet(a)*FiniteSet(b))
            for a,b in [(3,5), (2,8), (2,5)]]
    assert elements[0] in (deven & center).complement
    assert elements[1] in (deven & center).complement
    assert elements[2] not in (deven & center).complement




