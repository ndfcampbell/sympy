from sympy import Basic, Tuple

class Computation(Basic):
    """ Represents a computation graph """

    def __new__(cls, inputs, outputs, varins, varouts):
        return Basic.__new__(cls, Tuple(*inputs),
                                  Tuple(*outputs),
                                  Tuple(*varins),
                                  Tuple(*varouts))

    @property
    def inputs(self):
        return set(self.args[0])

    @property
    def outputs(self):
        return set(self.args[1])

    @property
    def varins(self):
        return set(self.args[2])

    @property
    def varouts(self):
        return set(self.args[3])

    def inplace(self):
        return len(set(self.inputs).intersection(self.varouts)) != 0

def intersect(a, b):
    return len(set(a).intersection(set(b))) != 0

class CompositeComputation(Computation):
    """ Represents a computation of many parts """
    def __new__(cls, *computations):
        # TODO: flatten composites of composites?
        return Basic.__new__(cls, *computations)

    def allinputs(self):
        return set([i for c in self.args for i in c.inputs])

    def alloutputs(self):
        return set([o for c in self.args for o in c.outputs])

    @property
    def inputs(self):
        return self.allinputs() - self.alloutputs()

    @property
    def outputs(self):
        return self.alloutputs() - self.allinputs()

    @property
    def varouts(self):
        raise NotImplemented()

    def dag_io(self):
        """ Return a dag of computations from inputs to outputs

        returns {A: {Bs}} such that A must occur before each of the Bs
        """
        return {A: set([B for B in self.args
                          if intersect(A.outputs, B.inputs)])
                   for A in self.args}

    def dag_oi(self):
        """ Return a dag of computations from outputs to inputs

        returns {A: {Bs}} such that A requires each of the Bs before it runs
        """
        return {A: set([B for B in self.args
                          if intersect(A.inputs, B.outputs)])
                   for A in self.args}

    def toposort(self):
        from sympy.utilities.iterables import _toposort
        return _toposort(self.dag_io())
