from sympy import (EmptySet, FiniteSet, S, Symbol, Interval, exp, erf, sqrt,
        symbols, simplify, Eq, cos, And)
from sympy.statistics.randomvariables import (ProbabilitySpace,
        NormalProbabilitySpace, ContinuousProbabilitySpace, Event, Die,
        Bernoulli, PDF, E, _rel_to_event, var, covar, independent, P, dependent)

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

def test_finite_pdf():
    a,b,c = symbols('a,b,c')
    B = Bernoulli(a,b,c, symbol='B')
    assert PDF(B.value)[c] == 1-a
    assert PDF(B.value)[b] == a

    p = Symbol('p')
    q = 1-p
    B = Bernoulli(p,1,0, symbol='B')
    X = B.value
    assert E(X) == p
    assert simplify(p*q - var(X)) == 0

    d1,d2,d3 = Die(), Die(), Die()
    X,Y,Z = [d.value for d in [d1,d2,d3]]

    assert E(X) == 3+S.Half
    assert var(X) == S(35)/12
    assert E(X+Y) == 7
    assert E(X+X) == 7
    assert E(a*X+b) == a*E(X)+b
    assert var(X+Y) == var(X) + var(Y)
    assert var(X+X) == 4 * var(X)
    assert covar(X,Y) == S.Zero
    assert covar(X, X+Y) == var(X)
    assert PDF(Eq(cos(X*S.Pi),1))[True] == S.Half

def test_normal_properties():
    a = Symbol('alpha', bounded=True)
    b = Symbol('beta', real=True, bounded=True)
    # mu = Symbol('mu', real=True, bounded=True)
    mu = S(0) # Until integration of gaussians get better
    sigmasquared = Symbol('sigma^2', real=True, bounded=True, positive=True)
    C,D = NormalProbabilitySpace(mu,sigmasquared), NormalProbabilitySpace(0,1)
    X,Y = C.value, D.value
    assert E(X) == mu
    assert var(X) == sigmasquared
    assert var(Y) == 1
    assert E(a*X + b) == a*E(X) + b
    assert simplify(var(a*X + b)) == a**2 * var(X)
    assert covar(X,Y) == S.Zero

def test_mixed():
    a = Symbol('alpha', bounded=True)
    b = Symbol('beta', real=True, bounded=True)
    # mu = Symbol('mu', real=True, bounded=True)
    mu = S(0) # Until integration of gaussians get better
    sigmasquared = Symbol('sigma^2', real=True, bounded=True, positive=True)
    A,B = NormalProbabilitySpace(mu,sigmasquared), NormalProbabilitySpace(0,1)
    X,Y = A.value, B.value

    D = Die().value

    assert E(X+D) == E(X)+E(D)
    assert simplify(var(X+D)) == var(X) + var(D)
    assert independent(X,D)
    assert dependent(X+D, D)

def test_event_generation():
    # mu = Symbol('mu', real=True, bounded=True)
    mu = S(0) # Until integration of gaussians get better
    sigmasquared = Symbol('sigma^2', real=True, bounded=True, positive=True)
    A,B = NormalProbabilitySpace(mu,sigmasquared), NormalProbabilitySpace(0,1)
    X,Y = A.value, B.value

    d1, d2 = Die(), Die()
    D1, D2 = d1.value, d2.value

    assert _rel_to_event(D1>4).set == FiniteSet(5,6)
    assert _rel_to_event(And(D1>3, D1+D2<6)).equals(
            Event(d1, FiniteSet(4)) & Event(d2, FiniteSet(1)) )

    assert P(Eq(D1,1)) == S(1)/6
    assert P(D1>3) == S.Half
    assert P(D1+D2 > 7) == S(5)/12

    assert _rel_to_event(X**2<1).set == Interval(-1,1, True,True)
    assert P(X**2<1) == erf(sqrt(2)/(2*sqrt(sigmasquared)))

