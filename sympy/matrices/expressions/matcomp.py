
from sympy.computations import InplaceComputation, CompositeComputation
from sympy import Symbol
from sympy.matrices.expressions import MatrixSymbol
from sympy.utilities.iterables import merge

class MatrixComputation(InplaceComputation):
    """ A Computation for Matrix operations

    Adds methods for matrices like shapes, dimensions.
    Also adds methods for Fortran code generation
    """
    name = 'f'

    def shapes(self):
        return {x: x.shape for x in self.variables if hasattr(x, 'shape')}

    def dimensions(self):
        return set(d for x, shape in self.shapes().items()
                     for d in shape if isinstance(d, Symbol))

    def declarations(self, namefn):
        inplace = self.inplace_fn()(self)
        def declaration(x):
            s = "%s" % inplace.types()[x]
            if x in inplace.intents():
                s += ", intent(%s)" % inplace.intents()[x]
            s += " :: "
            s += "%s" % namefn(x)
            if x in inplace.shapes():
                s += "%s" % str(inplace.shapes()[x])
            return s
        return map(declaration,
                sorted(set(inplace.variables) | set(inplace.dimensions()), key=str))

    def intents(self):
        def intent(x):
            if x in self.inputs and x in self.replacements().values():
                return 'inout'
            if x in self.inputs or x in self.dimensions():
                return 'in'
            if x in self.outputs:
                return 'out'

        return {x: intent(x) for x in set(self.variables) | self.dimensions()
                             if intent(x)}

    def header(self, namefn):
        return "subroutine %(name)s(%(inputs)s)" % {
            'name': self.name,
            'inputs': ', '.join(map(namefn, self.inputs+tuple(self.dimensions())))}

    def footer(self):
        return "RETURN\nEND\n"

    def print_Fortran(self, namefn, assumptions=True):
        return '\n\n'.join([self.header(namefn),
                            '\n'.join(self.declarations(namefn)),
                            '\n'.join(self.calls(namefn, assumptions)),
                            self.footer()])

    def write(self, filename, *args, **kwargs):
        file = open(filename, 'w')
        file.write(self.print_Fortran(*args, **kwargs))
        file.close()

    def build(self, *args, **kwargs):
        import os
        _id = abs(hash(args))
        src = kwargs.pop('src', 'tmp.f90')
        mod = kwargs.pop('mod', 'blasmod'+str(_id))
        self.write(src, *args, **kwargs)

        command = 'f2py -c %(src)s -m %(mod)s -lblas' % locals()
        file = os.popen(command); file.read()
        module = __import__(mod)
        return module.__dict__[self.name]

    @property
    def _composite(self):
        return MatrixRoutine


class CopyMatrix(InplaceComputation):
    _id = 0
    view_map = {}
    def __new__(cls, x):
        name = '_%d'%cls._id
        cls._id += 1
        cp = MatrixSymbol(name, x.rows, x.cols)
        return Computation.__new__((x,), (cp,))

def basic_names(x):
    """ A function to map MatrixExprs to strings

    if X is a MatrixSymbol return X.name
    Otherwise return a name like "_123" consistently for repeated inputs """
    if x in basic_names._cache:
        return basic_names._cache[x]
    if isinstance(x, (MatrixSymbol, Symbol)):
        result = x.name
    else:
        result = "_%d"%basic_names._id
        basic_names._id += 1
    basic_names._cache[x] = result
    return result
basic_names._cache = {}
basic_names._id = 1

# TODO: Rename
class MatrixRoutine(CompositeComputation, MatrixComputation):
    """ A Composite MatrixComputation """
    def calls(self, namefn, assumptions=True):
        computations = map(self.inplace_fn(), self.toposort())
        return [call for c in computations
                     for call in c.calls(namefn, assumptions)]

    def types(self):
        return merge(*map(lambda x: x.types(), self.computations))

    @property
    def variables(self):
        return set([x for c in self.computations for x in c.variables])

    def replacements(self):
        return merge(*[c.replacements() for c in self.computations])
