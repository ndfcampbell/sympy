from sympy import Basic, Tuple
from itertools import chain

def unique(seq):
    seen = set()
    for item in seq:
        if item not in seen:
            seen.add(item)
            yield item

def intersect(a, b):
    return not not set(a).intersection(set(b))

class Computation(Basic):
    """ An interface for a Computation

    Computations have inputs and outputs
    """

    inputs  = None
    outputs = None
    raw_inputs = property(lambda self: self.inputs)

    def edges(self):
        """ A sequence of edges """
        inedges  = ((i, self) for i in self.inputs)
        outedges = ((self, o) for o in self.outputs)
        return chain(inedges, outedges)

    @property
    def variables(self):
        return chain(self.inputs, self.outputs)

    def __add__(self, other):
        return CompositeComputation(self, other)

    def __str__(self):
        ins  = "["+', '.join(map(str, self.inputs)) +"]"
        outs = "["+', '.join(map(str, self.outputs))+"]"
        return "%s -> %s -> %s"%(ins, str(self.__class__.__name__), outs)

    def dot_nodes(self):
        return ['"%s" [shape=box, label=%s]' % (
                str(self), str(self.__class__.__name__))]

    def dot_edges(self):
        return ['"%s" -> "%s"' % tuple(map(str, edge)) for edge in self.edges()]

    def dot(self, orientation='TD'):
        """ A DOT language representation of the graph """
        nodes = "\n\t".join(self.dot_nodes())
        edges = "\n\t".join(self.dot_edges())
        return ("digraph{\n\trankdir=" + orientation +
                "\n\t" + nodes + "\n\n\t" + edges + "\n}")

    def writepdf(self, filename='comp', extension='pdf'):
        import os
        dotfile = open(filename+'.dot', 'w')
        dotfile.write(self.dot())
        dotfile.close()

        os.system('dot -T%s %s.dot -o %s.%s' % (
                        extension, filename, filename, extension))

    def show(self, filename='comp'):
        self.writepdf(filename)
        import os
        os.system('evince %s.pdf' % filename)

    def toposort(self):
        """ Order computations in an executable order """
        return [self]


class CompositeComputation(Computation):
    """ A computation composed of other computations """

    def __new__(cls, *args):
        obj = Basic.__new__(cls, *args)
        obj = obj.canonicalize()
        return obj

    computations = property(lambda self: self.args)

    def _input_outputs(self):
        """ Find the inputs and outputs of the complete computation """
        allin = tuple(unique(chain(
                        *[c.inputs  for c in self.computations])))
        allout = tuple(unique(chain(
                        *[c.outputs for c in self.computations])))
        inputs  = [i for i in allin  if i not in allout]
        outputs = [o for o in allout if o not in allin]
        ident_inputs  = [i for c in self.computations if isinstance(c, Identity)
                           for i in c.inputs]
        ident_outputs = [o for c in self.computations if isinstance(c, Identity)
                           for o in c.outputs]
        return tuple(inputs + ident_inputs), tuple(outputs + ident_outputs)

    @property
    def inputs(self):
        return self._input_outputs()[0]

    @property
    def outputs(self):
        return self._input_outputs()[1]

    @property
    def variables(self):
        return unique(chain(
                        *[c.variables for c in self.computations]))

    def __str__(self):
        return "[[" + ", ".join(map(str, self.toposort())) + "]]"

    def edges(self):
        return chain(*[c.edges() for c in self.computations])

    def dot_nodes(self):
        return (n for c in self.computations for n in c.dot_nodes())

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
        """ Order computations in an executable order """
        from sympy.utilities.iterables import _toposort
        return _toposort(self.dag_io())

    def canonicalize(self):
        from sympy.rules import exhaust, do_one, flatten, unpack, typed, sort
        rl = do_one(rm_identity, flatten, unpack, canon_unique, sort(str))
        return exhaust(typed({CompositeComputation: rl}))(self)

def canon_unique(comp):
    """ Remove repeat computations

    TODO: Why do these exist?
    """
    if len(comp.computations) != len(set(comp.computations)):
        return type(comp)(*tuple(unique(comp.computations)))
    else:
        return comp

def rm_identity(comp):
    """ Remove or reduce unnecessary identities """
    for c in comp.computations:
        if isinstance(c, Identity):
            others = [x for x in comp.computations if x != c]
            other_vars = set([v for other in others
                                for v in chain(other.outputs, other.inputs)])
            vars = [v for v in c.outputs if v not in other_vars]
            if not vars:
                return type(comp)(*others)
            if tuple(vars) != c.outputs:
                newident = Identity(*vars)
                return type(comp)(newident, *others)
    return comp

class Identity(Computation):
    """ An Identity computation """
    inputs = property(lambda self: self.args)
    outputs = inputs


class OpComp(Computation):
    """ A Computation represented by (Operation, inputs, outputs)

    Analagous to theano.Apply"""
    def __new__(cls, op, inputs, outputs):
        return Basic.__new__(cls, op, Tuple(*inputs), Tuple(*outputs))

    op = property(lambda self: self.args[0])
    inputs = property(lambda self: self.args[1])
    outputs = property(lambda self: self.args[2])

    def __str__(self):
        ins  = "["+', '.join(map(str, self.inputs)) +"]"
        outs = "["+', '.join(map(str, self.outputs))+"]"
        opstr = self.op.__name__ if isinstance(self.op, type) else str(self.op)
        return "%s -> %s -> %s"%(ins, opstr, outs)

    def dot_nodes(self):
        oname = self.op.__name__ if isinstance(self.op, type) else str(self.op)
        return ['"%s" [shape=box, label=%s]' % (str(self), oname)]
