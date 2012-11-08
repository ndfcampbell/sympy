
from sympy import Basic, Symbol, Q, symbols, ask, Tuple
from sympy.matrices.expressions import MatrixSymbol, Transpose
from sympy.utilities.iterables import merge
from matcomp import MatrixComputation
from sympy.rules.tools import subs

# Pattern variables
alpha = Symbol('alpha')
beta  = Symbol('beta')
n,m,k = symbols('n,m,k')
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, k)
C = MatrixSymbol('C', n, k)
S = MatrixSymbol('S', n, n)
x = MatrixSymbol('x', n, 1)
a = MatrixSymbol('a', m, 1)
b = MatrixSymbol('b', k, 1)

basetypes = {'S': 'real*4', 'D': 'real*8', 'C': 'complex*8', 'Z': 'complex*16'}

class MatrixCall(MatrixComputation):
    """ An atomic call, superclass for BLAS and LAPACK """
    def __new__(cls, *args):
        if args[-1] not in basetypes:
            typecode = 'D'
            args = args + (typecode,)
        return Basic.__new__(cls, *args)

    @property
    def inputs(self):
        return tuple(self.args[:-1])

    @property
    def outputs(self):
        cls = self.__class__
        mapping = dict(zip(cls._inputs, self.inputs))
        return tuple(map(lambda x: x.canonicalize(),
                         subs(mapping)(Tuple(*cls._outputs))))

    @property
    def typecode(self):
        return self.args[-1]

    def types(self):
        return merge({v: basetypes[self.typecode] for v in self.variables},
                     {d: 'integer' for d in self.dimensions()})

    def calls(self, namefn, assumptions=True):
        return [self.fortran_template % self.codemap(namefn, assumptions)]

    @property
    def variables(self):
        return filter(lambda x: isinstance(x, Basic) and not x.is_number,
                      self.inputs + self.outputs)

    def fnname(self):
        """ GEMM(...).fnname -> dgemm """
        return (self.typecode+self.__class__.__name__).lower()

    @classmethod
    def valid(cls, inputs, assumptions):
        d = dict(zip(cls._inputs, inputs))
        if cls.condition is True:
            return True
        return ask(cls.condition.subs(d), assumptions)

class SV(MatrixCall):
    """ Matrix Vector Solve """
    _inputs   = (S, x)
    _outputs  = (S.I*x,)
    view_map  = {0: 1}
    condition = True

