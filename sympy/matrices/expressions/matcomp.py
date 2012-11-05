
from sympy.computations import InplaceComputation, CompositeComputation
from sympy import Symbol, Expr
from sympy.matrices.expressions import MatrixSymbol
from sympy.utilities.iterables import merge

def is_number(x):
    return (isinstance(x, (int, float)) or
            isinstance(x, Expr) and x.is_Number)

def remove_numbers(coll):
    return filter(lambda x: not is_number(x), coll)

def shape_str(shape):
    if shape[0] == 1:
        return "(%d)"%shape[1]
    elif shape[1] == 1:
        return "(%d)"%shape[0]
    else:
        return "(%d, %d)"%shape

class MatrixComputation(InplaceComputation):
    """ A Computation for Matrix operations

    Adds methods for matrices like shapes, dimensions.
    Also adds methods for Fortran code generation
    """
    name = 'f'

    def shapes(self):
        return {x: x.shape for x in self.variables if hasattr(x, 'shape')}

    def dimensions(self):
        return set(d for x, shape in self.shapes().items() for d in shape)

    def declarations(self, namefn):
        def declaration(x):
            s = "%s" % self.types()[x]
            if x in self.intents():
                s += ", intent(%s)" % self.intents()[x]
            s += " :: "
            s += "%s" % namefn(x)
            if x in self.shapes():
                s += "%s" % shape_str(self.shapes()[x])
            return s
        return map(declaration,
                sorted(remove_numbers(set(self.inplace_variables) |
                                      set(self.dimensions())), key=str))

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
            'inputs': ', '.join(map(namefn,
                remove_numbers(self.inputs+tuple(self.dimensions()))))}

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

    if is_number(x):
        return str(x)
    if x in basic_names._cache:
        return basic_names._cache[x]
    if isinstance(x, (MatrixSymbol, Symbol)):
        result = x.name
    else:
        result = "_%d"%basic_names._id
        basic_names._id += 1

    # TODO: replace this with a collection-wise name function
    lower = lambda x: x.lower()
    while(result.lower() in map(lower, basic_names._cache.values())):
        result += '_'

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
