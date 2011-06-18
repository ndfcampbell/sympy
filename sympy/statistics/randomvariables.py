from sympy import (Basic, S, Rational, Expr, integrate, cacheit, Symbol, Add,
        Tuple, sqrt, Mul, solve, simplify, powsimp, Dummy, exp, pi, Piecewise,
        And, Or, gamma)
from sympy.core.sympify import sympify
from sympy.core.sets import (Set, ProductSet, FiniteSet, Union, Interval,
        is_flattenable)
from sympy.core.relational import Lt, Gt, Eq, Relational, Inequality
from sympy.solvers.inequalities import reduce_poly_inequalities
oo = S.Infinity

#from sympy.core import sympify, Integer, Rational, oo, Float, pi, Set, Basic
#from sympy.functions import sqrt, exp, erf

class ProbabilitySpace(Basic):
    """
    Represents the outcomes of a random process.
    Contains:
        A Sample Space of possible outcomes
        A Probability Measure on subsets of the sample space
    """

    def __new__(cls, symbol, sample_space, probability_measure):
        symbol = sympify(symbol)
        return Basic.__new__(cls, symbol, sample_space, probability_measure)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def sample_space(self):
        return self.args[1]

    @property
    def sample_space_event(self):
        return Event(self, self.sample_space)

    @property
    def probability_measure(self):
        return self.args[2]

    @property
    def value(self):
        return RandomSymbol(self)

    def __mul__(self, other):
        return ProductProbabilitySpace(self, other)

    @property
    def is_product(self):
        return False

    @property
    def is_finite(self):
        return self.sample_space.is_finite

    @property
    def is_bounded(self):
        if self.sample_space.is_bounded:
            return True
        # Even if sample_space is unbounded the value may be bounded if
        # Probability is zero on that part of the space

    _count = 0
    _name = 'space'
    @classmethod
    def create_symbol(cls):
        cls._count += 1
        return Symbol('%s%d'%(cls._name, cls._count), real=True)

class ProbabilityMeasure(Basic):
    """
    A Probability Measure is a function from Events to Real numbers.
    For events A,B in the SampleSpace the measure has the following properties
    P(A) >= 0 , P(SampleSpace) == 1 , P(A + B) == P(A) + P(B) - P(A intersect B)
    """
    def __call__(self, event):
        if not event.is_union and event.set.is_union:
            return UnionEvent(Event(event.pspace, set)
                    for set in event.set.args).measure
        return self._call(event)

class Event(Basic):
    """
    A subset of the possible outcomes of a random process.
    """
    def __new__(cls, pspace, set):
        if pspace.is_product and set.is_product and len(pspace)==len(set.sets):
            obj = ProductEvent(Event(space, s)
                    for space, s in zip(pspace.spaces, set.sets))
        elif set.is_union:
            obj = UnionEvent(Event(pspace, s) for s in set.args)
        else:
            obj = Basic.__new__(cls, pspace, set)

        return obj

    @property
    @cacheit
    def measure(self):
        return self.pspace.probability_measure(self)

    @property
    def pspace(self):
        return self.args[0]

    @property
    def set(self):
        return self.args[1]

    def intersect(self, other):
        """
        Returns the event that occurs if both the events occur.
        May be a ProductEvent if the events come from different ProbSpaces
        """
        if other.pspace == self.pspace:
            return Event(self.pspace, self.set.intersect(other.set))
        else:
            return ProductEvent(self).intersect(ProductEvent(other))

    def union(self, other):
        """
        Returns the event that occurs if either of the events occur.
        May be a UnionEvent if the events come from different ProbabilitySpaces
        """

        if other.pspace == self.pspace and not self.is_union:
            return Event(self.pspace, self.set + other.set)
        else:
            return UnionEvent(self, other)

    @property
    def complement(self):
        return Event(self.pspace, self.pspace.sample_space - self.set)

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersect(other)

    def __iter__(self):
        return self._iter_events()

    def __sub__(self, other):
        return self.intersect(other.complement)

    def __neg__(self):
        return self.complement

    def __invert__(self):
        return self.complement

    def _iter_events(self):
        if self.is_iterable:
            return (AtomicEvent(self.pspace, FiniteSet(item)) for item in self.set)
        else:
            raise TypeError("This Event is not iterable")

    @property
    def is_product(self):
        return False

    @property
    def is_union(self):
        return False

    @property
    def is_iterable(self):
        return self.set.is_iterable

    @property
    def is_real(self):
        return self.set.is_real

    def __contains__(self, other):
        if isinstance(other, Event):
            return other.pspace == self.pspace and self.set.subset(other.set)
        else:
            return other in self.set

class AtomicEvent(Event):
    @property
    def value(self):
        return tuple(self.set)[0]
    def dict(self):
        return {self.pspace.symbol:self.value}

def is_random(x):
    return isinstance(x, RandomSymbol)
def is_random_expr(expr):
    return any(is_random(sym) for sym in expr.free_symbols)
def random_symbols(expr):
    if is_random(expr):
        return [expr]
    try:
        return [s for s in expr.free_symbols if is_random(s)]
    except:
        return []

class RandomSymbol(Symbol):
    """
    Represents a Random Variable.
    A function on a ProbabilitySpace's Sample Space
    """

    def __new__(cls, pspace):
        return Basic.__new__(cls, pspace)

    @property
    def pspace(self):
        return self.args[0]

    @property
    def symbol(self):
        return self.pspace.symbol

    @property
    def pdf(self):
        return self.pspace.probability_measure.pdf

    @property
    def name(self):
        return self.symbol.name

    @property
    def is_commutative(self):
        return self.symbol.is_commutative
    @property
    def is_finite(self):
        return self.pspace.is_finite
    @property
    def is_real(self):
        return self.pspace.sample_space.is_real
    @property
    def is_positive(self):
        return self.pspace.sample_space.is_positive
    @property
    def is_bounded(self):
        return self.pspace.is_bounded

    def _hashable_content(self):
        return self._args


#====================================================================
#=== Product Spaces =================================================
#====================================================================

class ProductProbabilitySpace(ProbabilitySpace):
    """
    Cartesian Product of ProbabilitySpaces

    >>> from sympy.statistics.randomvariables import Coin
    >>> print (Coin(symbol='coin1') * Coin(symbol='coin2')).sample_space_event
    (coin1, coin2) in {H, T} x {H, T}

    """

    def __new__(cls, *args):
        args = list(args)
        def flatten(arg):
            if isinstance(arg, ProbabilitySpace) and not arg.is_product:
                return [arg]
            if isinstance(arg, ProbabilitySpace) and arg.is_product:
                return sum(map(flatten, arg.spaces), [])
            if is_flattenable(arg):
                return sum(map(flatten, arg), [])
            raise TypeError("Inputs must be (iterables of) ProductSpaces")
        spaces = flatten(args)

        # If there are repeat spaces remove them while preserving order
        s = []
        for space in spaces:
            if space not in s:
                s.append(space)

        return Basic.__new__(cls, *s)

    @property
    def spaces(self):
        return self.args

    @property
    def symbol(self):
        return tuple(pspace.symbol for pspace in self.spaces)

    @property
    def sample_space(self):
        return ProductSet(pspace.sample_space for pspace in self.spaces)

    @property
    def sample_space_event(self):
        return ProductEvent(pspace.sample_space_event for pspace in self.spaces)

    @property
    def probability_measure(self):
        return ProductProbabilityMeasure()

    def __iter__(self):
        return self.spaces.__iter__()

    def __contains__(self, other):
        return other in self.spaces

    @property
    def is_product(self):
        return True

    def __len__(self):
        return len(self.spaces)

class ProductProbabilityMeasure(ProbabilityMeasure):
    """
    ProductProbabilityMeasures measure the probability of a ProductEvent.
    In essense they are very simple. It simply calls and multiplies the
    measures of the constituent events.

    For internal use only.
    """
    def __new__(cls):
        return Basic.__new__(cls)

    def _call(self, productevent):
        # Assuming independence
        prob = 1
        for event in productevent.events:
            prob *= event.measure
        return prob

class ProductEvent(Event):
    """
    An intersection of events across different probability spaces
    The event that die1 is even and die2 is odd is a ProductEvent
    """

    def __new__(cls, *events):
        if len(events)==1 and isinstance(events[0], ProductEvent):
            return events[0]
        events = list(events)
        def flatten(arg):
            if isinstance(arg, Event) and not arg.is_product:
                return [arg]
            if isinstance(arg, Event) and arg.is_product:
                return list(arg.events)
            if is_flattenable(arg):
                return sum(map(flatten, arg), [])
            raise Exception("unhandled input %s"%arg)
        events = flatten(events)

        # If any of the events are union events
        # Break apart that union into subevents,
        # Construct a product event with each subevent
        # Return a Union of ProductEvents

        if any(event.is_union for event in events):
            for i, unionevent in enumerate(events):
                if unionevent.is_union: # Find first union event
                    break
            prodevents = []
            for subevent in unionevent.events:
                prodevents.append(
                        ProductEvent(*(events[:i]+[subevent]+events[i+1:])) )
            return UnionEvent(*prodevents)


        assert not any(event.is_union for event in events)

        return Basic.__new__(cls, *events)

    @property
    def events(self):
        return self.args

    @property
    def pspace(self):
        return ProductProbabilitySpace(e.pspace for e in self.events)

    @property
    def set(self):
        return ProductSet(e.set for e in self.events)

    def intersect(self, other):
        """
        Intersect this ProductEvent with another (possibly Product) event
        Returns a ProductEvent in all cases
        """

        # Consider all probability spaces in either of the events
        pspace = self.pspace * other.pspace
        # Cast each event up to the product subspace
        A = cast_event(self, pspace)
        B = cast_event(other, pspace)
        # Clear out any unions if they exist
        if A.is_union:
            return A.intersect(B)
        if B.is_union:
            return B.intersect(A)
        # intersect all of the constituent events
        return ProductEvent(a.intersect(b)
                for a,b in zip(A.events, B.events))

    def getevent(self, pspace):
        """Get event associated with a particular probability space"""
        for event in self.events:
            if event.pspace == pspace:
                return event
        raise ValueError("Probability Space not found in Event")

    def _iter_events(self):
        if self.is_iterable:
            import itertools
            event_iterators = [event._iter_events() for event in self.events]
            return (ProductEvent(*atomic_events)
                    for atomic_events in itertools.product(*event_iterators))
        else:
            raise TypeError("Not all constituent sets are iterable")

    def subset(self, other):
        self.intersect(other) == other

    @property
    def is_product(self):
        return True

class AtomicProductEvent(ProductEvent, AtomicEvent):
    def dict(self):
        return {ev.pspace.symbol:ev.value for ev in self.events}

def cast_event(event, productspace):
    """
    Casts an event into a particular ProductProbabilitySpace
    If ProbabilitySpaces are requested which are not in the event then the
    entire sample space is taken instead

    For internal use only.
    """
    if event.is_union:
        return UnionEvent(cast_event(e, productspace) for e in event.events)
    if not event.is_product:
        event = ProductEvent(event)
    events = []
    for pspace in productspace:
        if pspace in event.pspace:
            events.append(event.getevent(pspace))
        else:
            events.append(pspace.sample_space_event)
    return ProductEvent(*events)

class UnionEvent(Event):
    """
    Union of events, possibly across different ProbabilitySpaces
    Represents events like "die1 is odd or die2 is even"

    Creates ProductEvents on the fly if necessary
    """
    def __new__(cls, *events):
        events = list(events)
        def flatten(arg):
            if not arg: # Empty Event
                return []
            if isinstance(arg, Event) and not arg.is_union:
                return [arg]
            if isinstance(arg, Event) and arg.is_union:
                return sum(map(flatten, arg.events), [])
            if is_flattenable(arg):
                return sum(map(flatten, arg), [])
            raise TypeError("Inputs must be (iterable of) Event")
        events = flatten(events)

        # Clear out any emptyset events
        events = [event for event in events if event.set]
        if not events:
            return None
        if len(events)==1:
            return events[0]

        return Basic.__new__(cls, *events)

    @property
    def events(self):
        return self.args

    @property
    def pspace(self):
        # If events have homogenous ProbabilitySpace then return that
        first_pspace = self.events[0].pspace
        if all(e.pspace == first_pspace for e in self.events):
            return first_pspace
        # Otherwise return ProductProbabilitySpace
        return ProductProbabilitySpace(e.pspace for e in self.events)

    @property
    def set(self):
        if not self.pspace.is_product: # Homogenous ProbSpace Events
            return Union(e.set for e in self.events)
        else: # Heterogeneous ProductSpace Events
            return Union(cast_event(e, self.pspace).set for e in self.events)

    @property
    def measure(self):
        # Measure of a union is the sum of the measures of the events minus
        # the sum of their pairwise intersections plus the sum of their
        # triple-wise intersections minus ... etc...

        # Sets is a collection of intersections and a set of elementary
        # sets which made up those interections (called "sos" for set of sets)
        # An example element might of this list might be:
        #    ( {A,B,C}, A.intersect(B).intersect(C) )

        # Start with just elementary sets (  ({A}, A), ({B}, B), ... )
        # Then get and subtract (  ({A,B}, (A int B), ... ) while non-zero
        sets = [(FiniteSet(s), s) for s in self.events]
        measure = 0
        parity = 1
        while sets:
            # Add up the measure of these sets and add or subtract it to total
            measure += parity * sum(inter.measure for sos, inter in sets)

            # For each intersection in sets, compute the intersection with every
            # other set not already part of the intersection.
            sets = ((sos + FiniteSet(newset), newset.intersect(intersection))
                    for sos, intersection in sets for newset in self.args
                    if newset not in sos)

            # Clear out sets with no measure
            sets = [(sos,inter) for sos,inter in sets if inter.measure != 0]

            # Clear out duplicates
            sos_list = []
            sets_list = []
            for set in sets:
                if set[0] in sos_list:
                    continue
                else:
                    sos_list.append(set[0])
                    sets_list.append(set)
            sets = sets_list

            # Flip Parity - next time subtract/add if we added/subtracted here
            parity *= -1
        return measure

    @property
    def is_union(self):
        return True

    def intersect(self, other):
        return self.__class__(event.intersect(other) for event in self.events)

#====================================================================
#=== Finite Example Spaces ==========================================
#====================================================================

class FiniteProbabilityMeasure(ProbabilityMeasure):
    """
    An easy to create probability Measure
    Constructor takes either a dict or a function
    """
    def __new__(cls, pdf):

        # Is dict-like
        if hasattr(pdf, '__getitem__') and not hasattr(pdf, '__call__'):
            # Sympify the dict's keys/values
            pdf_input = dict((sympify(key), sympify(value))
                for key, value in pdf.items())

            # Wrap a function around the dict, defaulting to zero
            pdf = lambda x : pdf_input.get(x,0)

            # pdf is either represented as a lambda or a dict.
            # Neither is an effective arg for Basic comparisons
            # We store the key:value pairs as an arg for comparison purposes
            # and set a field pdf to be the lambda
            obj = Basic.__new__(cls,
                    Tuple(*[Tuple(k,v) for k,v in pdf_input.items()]))
            obj.pdf = pdf
            return obj

    def _call(self, event):
        return sum( self.pdf(element) for element in event.set )

class FiniteProbabilitySpace(ProbabilitySpace):
    """
    A ProbabilitySpace on a finite countable sample space

    Creates a ProbabilitySpace given a probability density function encoded in
    a dictionary like {'Heads':1/2 , 'Tails':1/2}

    See also:
        Die
        Bernoulli
        Coin

    >>> from sympy.statistics.randomvariables import FiniteProbabilitySpace
    >>> from sympy.statistics.randomvariables import Event
    >>> from sympy import S, FiniteSet

    >>> sixth = S(1)/6
    >>> pdf = {1:sixth, 2:sixth, 3:sixth, 4:sixth, 5:sixth, 6:sixth}
    >>> die = FiniteProbabilitySpace(pdf, symbol='die')

    >>> print die.sample_space_event
    die in {1, 2, 3, 4, 5, 6}

    >>> print Event(die, FiniteSet(2,4,6)).measure
    1/2
"""

    def __new__(cls, pdf, symbol=None):
        symbol = symbol or cls.create_symbol()
        M = FiniteProbabilityMeasure(pdf)
        sample_space = FiniteSet(pdf.keys())
        return ProbabilitySpace.__new__(cls, symbol, sample_space, M)

    @property
    def is_finite(self):
        return True

class Die(FiniteProbabilitySpace):
    """
    Represents a die of arbitrary number of sides.
    Implementation of FiniteProbabilitySpace

    >>> from sympy.statistics.randomvariables import Die, Event
    >>> from sympy import FiniteSet
    >>> die = Die(6, symbol='die1')

    >>> print die.sample_space_event
    die1 in {1, 2, 3, 4, 5, 6}

    >>> print Event(die, FiniteSet(2,4,6)).measure
    1/2

    """

    _count = 0
    _name = 'die'
    def __new__(cls, sides=6, symbol=None):
        pdf = dict((i,Rational(1,sides)) for i in range(1,sides+1))
        return FiniteProbabilitySpace.__new__(cls, pdf, symbol)

class Bernoulli(FiniteProbabilitySpace):
    """
    Represents a Bernoulli Event with probability p.
    Inherits from FiniteProbabilitySpace

    >>> from sympy.statistics.randomvariables import Bernoulli, Event
    >>> from sympy import S, FiniteSet

    >>> coin = Bernoulli(S.Half, 'H', 'T', symbol='coin')
    >>> print coin.sample_space_event
    coin in {H, T}

    >>> print Event(coin, FiniteSet('T')).measure
    1/2

    """
    _numcount = 0
    _name = 'bernoulli'
    def __new__(cls, p=S.Half, a=0, b=1, symbol=None):
        a, b, p = map(sympify, (a,b,p))
        pdf = {a:p, b:(1-p)}
        return FiniteProbabilitySpace.__new__(cls, pdf, symbol)

class Coin(Bernoulli):
    """
    Represents a coin flip Event with probability p.
    Inherits from Bernoulli

    >>> from sympy.statistics.randomvariables import Coin, Event
    >>> from sympy import FiniteSet

    >>> coin = Coin(symbol='coin')
    >>> print coin.sample_space_event
    coin in {H, T}

    >>> print Event(coin, FiniteSet('T')).measure
    1/2

    >>> print Event(Coin(p=1), FiniteSet('T')).measure # Trick coin
    0
    """
    _numcount = 0
    _name = 'coin'
    def __new__(cls, p=S.Half, symbol=None):
        p = sympify(p)
        return Bernoulli.__new__(cls, p=p, a='H', b='T', symbol=symbol)

#====================================================================
#=== Continuous Example Spaces ======================================
#====================================================================

class ContinuousProbabilityMeasure(ProbabilityMeasure):
    """
    An probability measure over the real line

    Given a symbol and a pdf.
    Can optionally provide a sample_space, (-oo, oo) assumed by default

    A building block of a ContinuousProbabilitySpace
    """
    def __new__(cls, symbol, pdf=None, cdf=None, sample_space=None):
        sample_space = sample_space or Interval(-oo,oo)
        obj = Basic.__new__(cls, symbol, pdf, cdf, sample_space)
        obj.symbol = symbol
        obj._pdf = pdf
        obj._cdf = cdf
        return obj

    @property
    def sample_space(self):
        return self.args[3]

    @property
    def pdf(self):
        if self._pdf:
            return self._pdf
        elif self._cdf:
            return self._cdf.diff(symbol)
        else:
            return None

    @property
    @cacheit
    def cdf(self):
        if self._cdf:
            return self._cdf
        elif self._pdf:
            return integrate(self._pdf, (self.symbol,
                    self.sample_space.intersect(Interval(-oo, self.symbol))))
        else:
            raise ValueError("CDF and PDF not defined")

    def _call(self, event):
        if event.set.is_interval:
            if self._cdf:
                return self.cdf(event.set.end) - self.cdf(event.set.start)
            else:
                return integrate(self.pdf,
                        (self.symbol, self.sample_space.intersect(event.set)))
        elif event.set.is_union:
            intervals = [s for s in event.set.args if s.is_interval]
            assert all(s.measure==0 for s in event.set.args
                    if not s.is_interval)
            return sum(self(Event(event.pspace, i)) for i in intervals)
        # FiniteSet or EmptySet
        elif event.set.measure == 0:
            return 0
        else:
            raise NotImplementedError("Can not integrate over %s."%event.set)

class ContinuousProbabilitySpace(ProbabilitySpace):
    """
    A ProbabilitySpace on the real line

    Defined by a pdf or cdf on a sample space over the symbolic variable

    >>> from sympy import exp, pi, Symbol, sqrt
    >>> from sympy.statistics import ContinuousProbabilitySpace, E

    >>> x = Symbol('x', real=True)
    >>> pdf = exp(-x**2 / 2) / sqrt(2*pi)
    >>> X = ContinuousProbabilitySpace(x, pdf).value

    >>> E(5*X + 10)
    10

    See also:
        NormalProbabilitySpace
        ExponentialProbabilitySpace

        FiniteProbabilitySpace
    """

    def __new__(cls, symbol, pdf=None, cdf=None,
            sample_space = Interval(-oo, oo)):
        M = ContinuousProbabilityMeasure(symbol=symbol, pdf=pdf, cdf=cdf,
                sample_space = sample_space)
        obj = ProbabilitySpace.__new__(cls, symbol, sample_space, M)
        return obj

    @property
    def is_bounded(self):
        # Probability of being at oo or -oo is zero
        e = Event(self, FiniteSet(oo, -oo))
        return self.probability_measure(e) == 0


class UniformProbabilitySpace(ContinuousProbabilitySpace):
    def __new__(cls, start, end, symbol = None):
        x = symbol or Dummy('x', real=True, finite=True)
        pdf = Piecewise( (0, x<start), (0, x>end), (S(1)/(end-start), True) )
        return ContinuousProbabilitySpace.__new__(cls, x, pdf=pdf)

class NormalProbabilitySpace(ContinuousProbabilitySpace):
    """
    Continuous Probability Space with Normal distribution.

    Defined by pdf exp( -(x-mu)**2 / (2*sigma**2) )
    with mean mu and standard deviation sigma

    >>> from sympy.statistics import NormalProbabilitySpace, PDF, Symbol
    >>> sigma = Symbol('sigma', real=True, bounded=True, positive=True)
    >>> mu = Symbol('mu', real=True, finite=True, bounded=True)

    >>> X = NormalProbabilitySpace(mu, sigma).value

    >>> PDF(X)
    (_y, 2**(1/2)*exp(-(_y - mu)**2/(2*sigma**2))/(2*pi**(1/2)*sigma))

    """

    def __new__(cls, mean, std, symbol = None):
        x = symbol or Dummy('x', real=True, finite=True)
        pdf = exp(-(x-mean)**2 / (2*std**2)) / (sqrt(2*pi) * std)
        obj = ContinuousProbabilitySpace.__new__(cls, x, pdf=pdf)
        obj.mean = mean
        obj.variance = variance
        return obj

    @property
    def is_bounded(self):
        return self.variance != S.Zero

class ParetoProbabilitySpace(ContinuousProbabilitySpace):
    def __new__(cls, xm, alpha, symbol=None):
        assert xm>0, "Xm must be positive"
        assert alpha>0, "Alpha must be positive"

        x = symbol or Dummy('x', real=True, finite=True)
        pdf = alpha * xm**alpha / x**(alpha+1)
        obj = ContinuousProbabilitySpace.__new__(cls, x, pdf=pdf,
                sample_space=Interval(xm, oo))
        obj.xm = xm
        obj.alpha = alpha
        return obj

class ExponentialProbabilitySpace(ContinuousProbabilitySpace):
    def __new__(cls, rate, symbol=None):
        x = symbol or Dummy('x', real=True, finite=True, positive=True)
        pdf = rate * exp(-rate*x)
        obj = ContinuousProbabilitySpace.__new__(cls, x, pdf=pdf,
                sample_space = Interval(0, oo))
        obj.rate = rate
        return obj

class BetaProbabilitySpace(ContinuousProbabilitySpace):
    def __new__(cls, alpha, beta, symbol=None):
        assert alpha>0, "Alpha must be positive"
        assert beta>0, "Beta must be positive"
        x = symbol or Dummy('x', real=True, finite=True, positive=True)
        pdf = x**(alpha-1) * (1-x)**(beta-1)
        pdf = pdf / integrate(pdf, (x, 0,1))
        obj = ContinuousProbabilitySpace.__new__(cls, x, pdf=pdf,
                sample_space = Interval(0, 1))
        obj.alpha = alpha
        obj.beta = beta
        return obj

class GammaProbabilitySpace(ContinuousProbabilitySpace):
    def __new__(cls, k, theta, symbol=None):
        assert k>0, "k must be positive"
        assert theta>0, "theta must be positive"
        x = symbol or Dummy('x', real=True, finite=True, positive=True)
        pdf = x**(k-1) * exp(-x/theta) / (gamma(k)*theta**k)

        obj = ContinuousProbabilitySpace.__new__(cls, x, pdf=pdf,
                sample_space = Interval(0, oo))
        obj.k = k
        obj.theta = theta
        return obj


#=========================================
#=== Random Expression Functions =========
#=========================================

def det(expr):
    crvs = [s for s in expr.free_symbols if is_random(s) and not s.is_finite]
    dexprdrv = [expr.diff(rv) for rv in crvs]
    return Mul(*dexprdrv)

def symbol_subs(expr):
    return expr.subs({rv:rv.symbol for rv in random_symbols(expr)})

def marginalize(expr, *rvs):
    for rv in rvs:
        if rv not in random_symbols(expr):
            continue
        expr = expr.subs(rv, rv.symbol)
        if rv.is_finite:
            expr = Add(*[expr.subs(atomic_event.dict())*P(atomic_event)
                for atomic_event in rv.pspace.sample_space_event])
        if not rv.is_finite:
            expr = integrate(expr * rv.pdf, (rv.symbol, rv.pspace.sample_space))
    return expr

@cacheit
def PDF(expr):
    rvs = random_symbols(expr)
    if rvs and all(rv.is_finite for rv in rvs):
        raise ValueError("Probability Density Function not defined for finite random variables. Try Probability Mass Funciton (PMF) instead")

    var, pdf = _continuous_pdf(expr)
    # If finite RVs occur in the pdf marginalize them out (Mixed case)
    pdf = marginalize(pdf, *[rv for rv in random_symbols(pdf) if rv.is_finite])
    return var, pdf

def _continuous_pdf(expr, val=None):
    # Gather continuous random variables from expression
    crvs = [rv for rv in random_symbols(expr) if not rv.is_finite]

    if not crvs:
        return expr, 1

    # Handle Continuous Random Variables

    # We allow all but one crv to float.
    # The last crv needs to enforce the condition that expr = Q
    # for some value Q. We then compute the probability density
    # Around of expr around Q
    head, tail = crvs[0], crvs[1:]
    # We'll use head to select Q given all other crvs
    val = val or Dummy('y', real=True, finite=True)
    # Solve expr == Q for a variable in expr
    constraint = solve(expr - val, head)
    # Convert the expression in RVs to one in RV.symbols
    constraint = [cons.subs({rv:rv.symbol for rv in crvs})
            for cons in constraint]
    # Compute PDF of the full vector space (X,Y,Z, ...) of the crvs
    fullpdf = Mul(*[crv.pdf for crv in crvs])
    # Divide by the determinant to rescale the dxdydz
    fullpdf = fullpdf / det(expr).subs({rv:rv.symbol for rv in crvs})
    # Substitute the constraint computed above into the PDF
    # Make sure to account for multiple solutions if they exist
    newpdf = (Add(*[fullpdf.subs(head.symbol, cons)
        for cons in constraint]))
    # Marginalize over extra symbols in tail
    if tail:
        newpdf= integrate(newpdf,
                *[(crv.symbol, crv.pspace.sample_space)
                    for crv in tail])
    return val, newpdf

@cacheit
def PMF(expr):
    """Probability Mass Function of a Finite Random Variable

    Compute the probability mass of each of the possible values
    of a random expression.

    Returns a dict that maps expression values to probabilities.

    >>> from sympy.statistics import Die, PMF
    >>> X = Die(6).value
    >>> PMF(2*X)
    {2: 1/6, 4: 1/6, 6: 1/6, 8: 1/6, 10: 1/6, 12: 1/6}

    >>> PMF(X>4)
    {False: 2/3, True: 1/3}
    """

    # Handle Discrete Random Variables
    frvs = [rv for rv in random_symbols(expr) if rv.is_finite]

    if not frvs:
        return {expr:1}

    d = {}
    rv = frvs[0] # Take first random variable

    # For each possibility of the value of rv
    for elem in rv.pspace.sample_space_event:
        # Compute the pdf of the rest of the expression recursively
        sub_pdf = PMF(expr.subs(rv, tuple(elem.set)[0]))
        # For each value with probability P in that pdf
        for val, prob in sub_pdf.items():
            # Add the probability of P * probability of this value of rv
            d[val] = d.get(val,0) + elem.measure * prob

    return d

@cacheit
def expectation(expr):
    """Expected value of a random expression.

    Marginalizes over all random_symbols within the expression

    >>> from sympy.statistics import Die, E, Bernoulli, Symbol
    >>> X = Die(6).value
    >>> p = Symbol('p')
    >>> B = Bernoulli(p, 1, 0).value

    >>> E(2*X)
    7

    >>> E(B)
    p
    """
    return marginalize(expr, *random_symbols(expr))
E = expectation

def variance(X):
    """Variance of a random expression.

    Expectation of (X-E(X))**2

    >>> from sympy.statistics import Die, E, Bernoulli, Symbol, var, simplify
    >>> X = Die(6).value
    >>> p = Symbol('p')
    >>> B = Bernoulli(p, 1, 0).value

    >>> var(2*X)
    35/3

    >>> simplify(var(B))
    p*(-p + 1)

    """
    return E(X**2) - E(X)**2
var = variance


def standard_deviation(X):
    """Standard Deviation of a random expression.

    Square root of the Expectation of (X-E(X))**2

    >>> from sympy.statistics import Bernoulli, std, Symbol
    >>> p = Symbol('p')
    >>> B = Bernoulli(p, 1, 0).value

    >>> std(B)
    (-p**2 + p)**(1/2)

    """
    return sqrt(variance(X))
std = standard_deviation

def covariance(X, Y):
    """Covariance of two random expressions.

    The expectation that the two variables will rise and fall together

    Covariance(X,Y) = E( (X-E(X)) * (Y-E(Y)) )


    >>> from sympy.statistics import ExponentialProbabilitySpace, covar, Symbol
    >>> rate = Symbol('lambda', positive=True, real=True, bounded = True)
    >>> X = ExponentialProbabilitySpace(rate).value
    >>> Y = ExponentialProbabilitySpace(rate).value

    >>> covar(X, X)
    lambda**(-2)
    >>> covar(X, Y)
    0
    >>> covar(X, Y + rate*X)
    1/lambda
    """

    return E( (X-E(X)) * (Y-E(Y)) )
covar = covariance

def skewness(X):
    mu = E(X)
    sigma = std(X)
    return E( ((X-mu)/sigma) ** 3 )

def pspace(expr):
    """
    The underlying ProbabilitySpace of an expression which holds RandomSymbols
    """
    rvs = random_symbols(expr)
    return ProductProbabilitySpace(*[rv.pspace for rv in rvs])

def independent(X,Y):
    # Independent if their ProbabilitySpaces do not overlap
    return len( set(pspace(X).spaces) & set(pspace(Y).spaces) ) == 0
def dependent(X,Y):
    return not independent(X,Y)

#===================================
#===== Event Creation ==============
#===================================

def P(arg):
    """
    The Probability of an event

    from sympy.statistics import P, Die
    >>> from sympy import Eq, cos, pi
    >>> from sympy.statistics import P, Die

    >>> X = Die(6).value

    >>> P(X>4)
    1/3

    >>> P( Eq(cos(pi*X), 1) )
    1/2
    """
    if arg.is_Relational or arg.is_Boolean:
        return P(_rel_to_event(arg))
    if isinstance(arg, Event):
        return arg.measure

def _rel_to_event(rel):
    rvs = random_symbols(rel)
    if not rvs:
        raise ValueError("No random variables present in %s"%rel)

    if rel.is_Boolean:
        if rel.__class__ == Or:
            return UnionEvent(_rel_to_event(arg) for arg in rel.args)
        if rel.__class__ == And:
            intersection = _rel_to_event(rel.args[0])
            for arg in rel.args[1:]:
                intersection = intersection & _rel_to_event(arg)
            return intersection


    if all(rv.is_finite for rv in rvs):
        return _rel_to_event_finite(rel)
    elif not any(rv.is_finite for rv in rvs):
        return _rel_to_event_continuous(rel)
    raise NotImplementedError("Events of complex Relationals not implemented")

def eventify(expr):
    """
    Converts a relation on an expression into an event if the expression
    contains a RandomSymbol

    from sympy.statistics import P, Die

    >>> from sympy.statistics import eventify, Die

    >>> X = Die(6).value
    >>> eventify(X>4)
    die1 in {5, 6}

    """

    return _rel_to_event(expr)

def _rel_to_event_finite(rel):
    ps = pspace(rel)
    if ps.is_product and len(ps.spaces)==1:
        ps = ps.spaces[0]
    testrel = rel.subs({rv:rv.symbol for rv in random_symbols(rel)})
    if ps.is_finite:
        events = []
        for event in ps.sample_space_event:
            if ps.is_product:
                subsdict={e.pspace.symbol:tuple(e.set)[0] for e in event.events}
            else:
                subsdict = {event.pspace.symbol:tuple(event.set)[0]}
            val = testrel.subs(subsdict)
            if val == True:
                events.append(event)
            elif val != False:
                raise ValueError("Relational did not evaluate to True/False")
        return UnionEvent(events)

def _rel_to_event_continuous(rel):
    rvs = random_symbols(rel)
    if len(rvs)!=1 or rvs[0].is_finite:
        raise NotImplementedError(
                "Not implemented for multiple continuous random variables")
    if not rel.is_Relational:
        raise "Argument is not a Relational"

    interval = reduce_poly_inequalities([[rel]], rvs[0], relational=False)
    event = Event(rvs[0].pspace, interval)

    return event

