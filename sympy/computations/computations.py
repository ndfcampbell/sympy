from sympy import Basic, Tuple, Dict
from sympy.rules.tools import subs
from sympy.rules import chain

class Computation(Basic):
    """ Represents a computation graph """

    def __new__(cls, inputs, outputs):
        return Basic.__new__(cls, Tuple(*inputs),
                                  Tuple(*outputs))

    @property
    def inputs(self):
        return self.args[0]

    @property
    def outputs(self):
        return self.args[1]

    @classmethod
    def purify(cls):
        return type(cls.__name__+"Pure", (Pure, cls), {})

def intersect(a, b):
    return len(set(a).intersection(set(b))) != 0

class CompositeComputation(Computation):
    """ Represents a computation of many parts """
    def __new__(cls, computations, inputs=None, outputs=None):
        # TODO: flatten composites of composites?
        allinputs  = set([i for c in computations for i in c.inputs])
        alloutputs = set([o for c in computations for o in c.outputs])
        if inputs is None:
            inputs = Tuple(*sorted(allinputs - alloutputs, key=str))
        if outputs is None:
            outputs = Tuple(*sorted(alloutputs - allinputs, key=str))

        return Basic.__new__(cls, Tuple(*computations),
                                  Tuple(*inputs),
                                  Tuple(*outputs))

    @property
    def computations(self):
        return self.args[0]

    @property
    def inputs(self):
        return self.args[1]

    @property
    def outputs(self):
        return self.args[2]

    def dag_io(self):
        """ Return a dag of computations from inputs to outputs

        returns {A: {Bs}} such that A must occur before each of the Bs
        """
        return {A: set([B for B in self.computations
                          if intersect(A.outputs, B.inputs)])
                    for A in self.computations}

    def dag_oi(self):
        """ Return a dag of computations from outputs to inputs

        returns {A: {Bs}} such that A requires each of the Bs before it runs
        """
        return {A: set([B for B in self.computations
                          if intersect(A.inputs, B.outputs)])
                    for A in self.computations}

    def toposort(self):
        from sympy.utilities.iterables import _toposort
        return _toposort(self.dag_io())


class InplaceComputation(Computation):
    def inplace(self):
        return not self.view_map
    def replacements(self):
        return {self.outputs[k]: self.inputs[v] for k, v in
                self.view_map.items()}
    def inplace_fn(self, seen = set([])):
        """ Return a version of self with all inplace variables replaced """
        replacements = Tuple(*[Tuple(k, v) for k, v in
            self.replacements().items()])
        seen = set([])
        unseen = set(replacements) - seen
        rl = lambda x: x
        while unseen:
            k, v = unseen.pop()
            newrl = subs({k: v})
            replacements = newrl(replacements)
            seen.add(Tuple(k, k))
            unseen = set(replacements) - seen
            rl = chain(rl, newrl) # Build up rl as we go
        return rl

class Pure(InplaceComputation):
    view_map = {}
