from sympy import Basic, S
from sympy.core.sympify import sympify
from sympy.core.sets import Set, ProductSet, FiniteSet, Union


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

class ProbabilityMeasure(Basic):
    """
    A Probability Measure is a function from Events to Real numbers.
    For events A,B in the SampleSpace the measure has the following properties
    P(A) >= 0 , P(SampleSpace) == 1 , P(A + B) == P(A) + P(B) - P(A intersect B)
    """
    def __call__(self, event):
        return self._call(event)

class Event(Set):
    """
    A subset of the possible outcomes of a random process.
    """
    def __new__(cls, pspace, a_set):
        obj = Basic.__new__(cls, pspace, a_set)

        #setattrs = dir(a_set)
        #selfattrs = dir(obj)
        #new_attr_names = (name for name in setattrs if name not in selfattrs)
        #for attr_name in new_attr_names:
        #    setattr(obj, attr_name, getattr(obj.set, attr_name))
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
        if other.pspace == self.pspace:
            return Event(self.pspace, self.set.intersect(other.set))
        else:
            return ProductEvent(self).intersect(ProductEvent(other))

    def union(self, other):
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
        return self.set.__iter__()

    @property
    def is_product(self):
        return False

    @property
    def is_union(self):
        return False

class RandomVariable(Basic):
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

    def __add__(self, other):
        if self.pspace == other.pspace:
            pspace = self.pspace
        else:
            pspace = self.pspace * other.space
        return RandomVariable(pspace,
                lambda *args : self.fn(*args) + other.fn(*args))



class FiniteProbabilityMeasure(ProbabilityMeasure):

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
        return sum( self.pdf(element) for element in event )

#====================================================================
#=== Product Spaces =================================================
#====================================================================

class ProductProbabilitySpace(ProbabilitySpace):
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
        return [pspace.name for pspace in self.constituent_spaces]

    @property
    def sample_space(self):
        return ProductSet(pspace.sample_space
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
    """ProductProbabilityMeasures measure the probability of a ProductEvent.
    In essense they are very simple. It simply calls the measures of the
    constituent events.

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
        # Consider all probability spaces in either of the events
        pspace = self.pspace * other.pspace
        # Cast each event up to the product subspace
        A = cast_event(self, pspace)
        B = cast_event(other, pspace)
        # intersect all of the constituent events
        return ProductEvent(a.intersect(b)
                for a,b in zip(A.events, B.events))

    def getevent(self, pspace):
        """Get event associated with particular probability space"""
        for event in self.events:
            if event.pspace == pspace:
                return event
        raise ValueError("Probability Space not found in Event")

    @property
    def is_product(self):
        return True

def cast_event(event, productspace):
    """Casts an event into a particular ProductProbabilitySpace
    If ProbabilitySpaces are requested which are not in the event then the
    entire sample space is taken instead"""

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
        pspace = self.pspace
        return Union(cast_event(e, pspace).set for e in self.events)

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

