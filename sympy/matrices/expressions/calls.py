from sympy import Basic, Symbol, Q, symbols, ask, Tuple, Expr
from sympy.matrices.expressions import MatrixSymbol, Transpose, MatrixExpr
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
        def canonicalize(x):
            if isinstance(x, MatrixExpr):
                return x.canonicalize()
            if isinstance(x, Symbol):
                return x
            if isinstance(x, Expr):
                return type(x)(*x.args)
        return tuple(map(canonicalize, subs(mapping)(Tuple(*cls._outputs))))

    @property
    def typecode(self):
        return self.args[-1]

    def types(self):
        return merge(dict(zip(self.inputs, self.in_types)),
                     dict(zip(self.outputs, self.out_types)),
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

    basetype = property(lambda self:  basetypes[self.typecode])
    _in_types = property(lambda self: (None,)*len(self._inputs))
    _out_types = property(lambda self: (None,)*len(self._outputs))

    @property
    def in_types(self):
        return tuple(it or self.basetype for it in self._in_types)

    @property
    def out_types(self):
        return tuple(ot or self.basetype for ot in self._out_types)

    flags = []

def trans(A):
    """ Return 'T' if A is a transpose, else 'N' """
    if isinstance(A, Transpose):
        return 'T'
    else:
        return 'N'

def uplo(A, assumptions):
    """ Return 'U' if A is stored in the upper Triangular 'U' if lower """
    if ask(Q.upper_triangular(A), assumptions):
        return 'U'
    if ask(Q.lower_triangular(A), assumptions):
        return 'L'

def LD(A):
    """ Leading dimension of matrix """
    # TODO make sure we don't use transposed matrices in untransposable slots
    return str(detranspose(A).shape[0])

def left_or_right(A, B, predicate, assumptions):
    """ Return 'L' if predicate is true of A, 'R' if predicate is true of B """
    if ask(predicate(A), assumptions):
        return 'L'
    if ask(predicate(B), assumptions):
        return 'R'

def diag(A, assumptions):
    """ Return 'U' if A is unit_triangular """
    if ask(Q.unit_triangular(A), assumptions):
        return 'U'
    else:
        return 'N'

def detranspose(A):
    """ Unpack a transposed matrix """
    if isinstance(A, Transpose):
        return A.arg
    else:
        return A
