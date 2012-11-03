from sympy import Basic, Tuple, Dict
from sympy.rules.tools import subs
from sympy.rules import chain

class Computation(Basic):
    """ Computation graph - stores inputs and outputs """

    def __new__(cls, inputs, outputs):
        return Basic.__new__(cls, Tuple(*inputs),
                                  Tuple(*outputs))

    @property
    def inputs(self):
        return self.args[0]

    @property
    def outputs(self):
        return self.args[1]

    def __add__(self, other):
        return self._composite((self, other))

    @property
    def _composite(self):
        return CompositeComputation


def intersect(a, b):
    return len(set(a).intersection(set(b))) != 0

class CompositeComputation(Computation):
    """ Computation of many parts

    Constituent computations are stored in a set.
    Dependence is inferred from their inputs and outputs.
    """
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
    """ Computation where some inputs are overwritten """
    def inplace(self):
        return not self.view_map

    def replacements(self):
        return {self.outputs[k]: self.inputs[v] for k, v in
                self.view_map.items()}

    def inplace_fn(self, seen = set([])):
        """ Return a function to substitute outputs with overwritten inputs """
        return subs(self.replacements())

    @property
    def inplace_outputs(self):
        d = self.replacements()
        return tuple(d.get(o, o) for o in self.outputs)

    @property
    def inplace_variables(self):
        d = self.replacements()
        return tuple(d.get(v, v) for v in self.variables)
