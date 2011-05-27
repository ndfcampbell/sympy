from sympy import Basic
from sympy.core.sympify import sympify

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
    def probability_measure(self):
        return self.args[2]

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

        setattrs = dir(a_set)
        selfattrs = dir(obj)
        new_attr_names = (name for name in setattrs if name not in selfattrs)
        for attr_name in new_attr_names:
            setattr(obj, attr_name, getattr(obj.set, attr_name))
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
        if other.pspace != self.pspace:
            raise ValueError(
                    "Cannot intersect two events from different spaces")
        return Event(self.pspace, self.set.intersect(other.set))
    @property
    def complement(self):
        return Event(self.pspace, self.pspace.sample_space - self.set)

    def __add__(self, other):
        if other.pspace != self.pspace:
            raise ValueError("Cannot add two events from different spaces")
        return Event(self.pspace, self.set + other.set)

    def __iter__(self):
        return self.set.__iter__()

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



