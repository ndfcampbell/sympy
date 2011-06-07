from sympy import Basic, S, Rational, Expr, integrate
from sympy.core.sympify import sympify
from sympy.core.sets import Set, ProductSet, FiniteSet, Union, Interval

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

    def __new__(cls, name, sample_space, probability_measure):
        name = sympify(name)
        return Basic.__new__(cls, name, sample_space, probability_measure)

    @property
    def name(self):
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

    def __mul__(self, other):
        return ProductProbabilitySpace(self, other)

    @property
    def is_product(self):
        return False
    _count = 0
    _name = 'space'
    @classmethod
    def create_name(cls):
        cls._count += 1
        return '%s%d'%(cls._name, cls._count)

class ProbabilityMeasure(Basic):
    """
    A Probability Measure is a function from Events to Real numbers.
    For events A,B in the SampleSpace the measure has the following properties
    P(A) >= 0 , P(SampleSpace) == 1 , P(A + B) == P(A) + P(B) - P(A intersect B)
    """
    def __call__(self, event):
        return self._call(event)

class Event(Basic):
    """
    A subset of the possible outcomes of a random process.
    """
    def __new__(cls, pspace, a_set):
        obj = Basic.__new__(cls, pspace, a_set)

        return obj

    @property
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

    def _iter_events(self):
        if self.is_iterable:
            return (Event(self.pspace, FiniteSet(item)) for item in self.set)
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

class RandomVariable(Expr):
    """
    Represents a Random Variable.
    A function on a ProbabilitySpace's Sample Space
    """

    def __new__(cls, pspace, fn):
        args = [pspace, fn]
        return Basic.__new__(cls, *args)

    @property
    def pspace(self):
        return args[0]

    @property
    def fn(self):
        return args[1]

    def _add(self, other):
        if self.pspace == other.pspace:
            pspace = self.pspace
        else:
            pspace = self.pspace * other.space
        return RandomVariable(pspace,
                lambda *args : self.fn(*args) + other.fn(*args))



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

        return Basic.__new__(cls, pdf)

    @property
    def pdf(self):
        return self.args[0]

    def _call(self, event):
        return sum( self.pdf(element) for element in event.set )

#====================================================================
#=== Product Spaces =================================================
#====================================================================

class ProductProbabilitySpace(ProbabilitySpace):
    """
    Cartesian Product of ProbabilitySpaces

    >>> from sympy.statistics.randomvariables import Coin
    >>> print (Coin(name='coin1') * Coin(name='coin2')).sample_space_event
    {coin1, coin2} in {H, T} x {H, T}

    """

    def __new__(cls, *args):
        args = list(args)
        def flatten(arg):
            if isinstance(arg, ProbabilitySpace) and not arg.is_product:
                return [arg]
            if isinstance(arg, ProbabilitySpace) and arg.is_product:
                return sum(map(flatten, arg), [])
            if hasattr(arg, '__iter__') and not isinstance(arg, ProbabilitySpace):
                return sum(map(flatten, arg), [])
            raise TypeError("Inputs must be (iterable of) ProductSpace")
        spaces = set(flatten(args))

        return Basic.__new__(cls, *spaces)

    @property
    def constituent_spaces(self):
        return FiniteSet(self.args)

    @property
    def name(self):
        return FiniteSet(pspace.name for pspace in self.constituent_spaces)

    @property
    def sample_space(self):
        return ProductSet(pspace.sample_space
                for pspace in self.constituent_spaces)

    @property
    def sample_space_event(self):
        return ProductEvent(pspace.sample_space_event
                for pspace in self.constituent_spaces)

    @property
    def probability_measure(self):
        return ProductProbabilityMeasure()

    def __iter__(self):
        return self.constituent_spaces.__iter__()

    def __contains__(self, other):
        return other in self.constituent_spaces

    @property
    def is_product(self):
        return True

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
                raise TypeError("Inputs must be (iterable of) ProductSpace")
            if hasattr(arg, '__iter__') and not isinstance(arg, Event):
                return sum(map(flatten, arg), [])
        events = flatten(events)

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
            return (ProductEvent(atomic_events)
                    for atomic_events in itertools.product(*event_iterators))
        else:
            raise TypeError("Not all constituent sets are iterable")


    @property
    def is_product(self):
        return True

def cast_event(event, productspace):
    """
    Casts an event into a particular ProductProbabilitySpace
    If ProbabilitySpaces are requested which are not in the event then the
    entire sample space is taken instead

    For internal use only.
    """

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
            if isinstance(arg, Event) and not arg.is_union:
                return [arg]
            if isinstance(arg, Event) and arg.is_union:
                return sum(map(flatten, arg.events), [])
            if hasattr(arg, '__iter__') and not isinstance(arg, Event):
                return sum(map(flatten, arg), [])
            raise TypeError("Inputs must be (iterable of) Event")
        events = flatten(events)

        return Basic.__new__(cls, *events)

    @property
    def events(self):
        return self.args

    @property
    def pspace(self):
        return ProductProbabilitySpace(e.pspace for e in self.events)

    @property
    def set(self):
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
        sets = [(FiniteSet((s,)), s) for s in self.events]
        measure = 0
        parity = 1
        while sets:
            # Add up the measure of these sets and add or subtract it to total
            measure += parity * sum(inter.measure for sos, inter in sets)

            # For each intersection in sets, compute the intersection with every
            # other set not already part of the intersection.
            sets = ((sos + FiniteSet((newset,)), newset.intersect(intersection))
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


#====================================================================
#=== Finite Example Spaces ==========================================
#====================================================================
class FiniteProbabilitySpace(ProbabilitySpace):
    """
    A ProbabilitySpace on a finite countable sample space

    Creates a ProbabilitySpace given a probability density function encoded in
    a dictionary like {'H':1/2 , 'T':1/2}

    See also:
        Die
        Bernoulli
        Coin

    >>> from sympy.statistics.randomvariables import FiniteProbabilitySpace
    >>> from sympy.statistics.randomvariables import Event
    >>> from sympy import S, FiniteSet

    >>> sixth = S(1)/6
    >>> pdf = {1:sixth, 2:sixth, 3:sixth, 4:sixth, 5:sixth, 6:sixth}
    >>> die = FiniteProbabilitySpace(pdf, name='die')

    >>> print die.sample_space_event
    die in {1, 2, 3, 4, 5, 6}

    >>> print Event(die, FiniteSet(2,4,6)).measure
    1/2
"""

    def __new__(cls, pdf, name=None):
        if not name:
            name = cls.create_name()
        M = FiniteProbabilityMeasure(pdf)
        sample_space = FiniteSet(pdf.keys())
        return Basic.__new__(cls, name, sample_space, M)

class Die(FiniteProbabilitySpace):
    """
    Represents a die of arbitrary number of sides.
    Implementation of FiniteProbabilitySpace

    >>> from sympy.statistics.randomvariables import Die, Event
    >>> from sympy import FiniteSet
    >>> die = Die(6)

    >>> print die.sample_space_event
    die1 in {1, 2, 3, 4, 5, 6}

    >>> print Event(die, FiniteSet(2,4,6)).measure
    1/2

    """

    _count = 0
    _name = 'die'
    def __new__(cls, sides=6, name=None):
        pdf = dict((i,Rational(1,sides)) for i in range(1,sides+1))
        return FiniteProbabilitySpace.__new__(cls, pdf, name)

class Bernoulli(FiniteProbabilitySpace):
    """
    Represents a Bernoulli Event with probability p.
    Inherits from FiniteProbabilitySpace

    >>> from sympy.statistics.randomvariables import Bernoulli, Event
    >>> from sympy import S, FiniteSet

    >>> coin = Bernoulli(S.Half, 'H', 'T', name='coin')
    >>> print coin.sample_space_event
    coin in {H, T}

    >>> print Event(coin, FiniteSet('T')).measure
    1/2

    """
    _numcount = 0
    _name = 'bernoulli'
    def __new__(cls, p=S.Half, a=0, b=1, name=None):
        pdf = {a:p, b:(1-p)}
        return FiniteProbabilitySpace.__new__(cls, pdf, name)

class Coin(Bernoulli):
    """
    Represents a coin flip Event with probability p.
    Inherits from Bernoulli

    >>> from sympy.statistics.randomvariables import Coin, Event
    >>> from sympy import FiniteSet

    >>> coin = Coin(name='coin')
    >>> print coin.sample_space_event
    coin in {H, T}

    >>> print Event(coin, FiniteSet('T')).measure
    1/2

    >>> print Event(Coin(p=1), FiniteSet('T')).measure # Trick coin
    0
    """
    _numcount = 0
    _name = 'coin'
    def __new__(cls, p=S.Half, name=None):
        return Bernoulli.__new__(cls, p=p, a='H', b='T', name=name)

#====================================================================
#=== Continuous Example Spaces ======================================
#====================================================================


class ContinuousProbabilityMeasure(ProbabilityMeasure):
    """
    An easy to create probability Measure
    Constructor takes either a dict or a function
    """
    def __new__(cls, symbol, pdf=None, cdf=None):
        obj = Basic.__new__(cls, symbol, pdf, cdf)
        obj.symbol = symbol
        obj._pdf = pdf
        obj._cdf = cdf
        return obj

    @property
    def pdf(self):
        if self._pdf:
            return self._pdf
        elif self._cdf:
            return self._cdf.diff(symbol)
        else:
            return None

    @property
    def cdf(self):
        if self._cdf:
            return self._cdf
        elif self._pdf:
            return integrate(self._pdf,
                    (self.symbol, Interval(-oo, self.symbol)))
        else:
            raise ValueError("CDF and PDF not defined")

    def _call(self, event):
        if event.set.is_interval:
            if self._cdf:
                return self._cdf(event.set.end) - self._cdf(event.set.start)
            else:
                return integrate(self._pdf, (self.symbol, event.set))
        elif event.set.is_union:
            intervals = [s for s in event.set.args if s.is_interval]
            assert all(s.measure==0 for s in event.set.args
                    if not s.is_interval)
            return sum(self(Event(event.pspace, i)) for i in intervals)
        else:
            raise NotImplementedError(
            "Measuing sets other than unions or intervals is not yet supported")

class ContinuousProbabilitySpace(ProbabilitySpace):
    """
    A ProbabilitySpace on the real line

    Creates a ProbabilitySpace given ...

    See also:
        FiniteProbabilitySpace

    """

    def __new__(cls, symbol, pdf=None, cdf=None, name=None):
        if not name:
            name = cls.create_name()
        M = ContinuousProbabilityMeasure(symbol=symbol, pdf=pdf, cdf=cdf)
        sample_space = Interval(-oo, oo)
        return Basic.__new__(cls, name, sample_space, M)
