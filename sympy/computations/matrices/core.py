from sympy.computations import Computation
from sympy.computations.core import unique
from sympy import Symbol, Expr, Basic, ask, Tuple
from sympy.matrices.expressions import MatrixExpr, MatMul
from sympy.rules.tools import subs
from sympy.rules import exhaust
from sympy.rules.traverse import bottom_up

def is_number(x):
    return (isinstance(x, (int, float)) or
            isinstance(x, Expr) and x.is_Number)

def remove_numbers(coll):
    return filter(lambda x: not is_number(x), coll)

basetypes = {'S': 'real*4',
             'D': 'real*8',
             'C': 'complex*8',
             'Z': 'complex*16'}


class MatrixCall(Computation):
    """ An atomic call, superclass for BLAS and LAPACK """
    def __new__(cls, *args):
        if args[-1] not in basetypes:
            typecode = 'D'
            args = args + (typecode,)
        return Basic.__new__(cls, *args)

    raw_inputs = property(lambda self: tuple(self.args[:-1]))
    typecode   = property(lambda self: self.args[-1])

    inputs = property(lambda self:
                    tuple(map(exhaust(bottom_up(canonicalize)),
                              unique(remove_numbers(self.raw_inputs)))))

    @property
    def outputs(self):
        cls = self.__class__
        mapping = dict(zip(cls._inputs, self.raw_inputs))
        return tuple(map(exhaust(bottom_up(canonicalize)),
                         subs(mapping)(Tuple(*cls._outputs))))

    basetype = property(lambda self:  basetypes[self.typecode])
    _in_types = property(lambda self: (None,)*len(self._inputs))
    _out_types = property(lambda self: (None,)*len(self._outputs))

    in_types  = property(lambda self:
                          tuple(it or self.basetype for it in self._in_types))
    out_types = property(lambda self:
                          tuple(ot or self.basetype for ot in self._out_types))

    @property
    def inplace(self):
        rv = {}
        for outind, inind in self.view_map.items():
            rawinput = self.raw_inputs[inind]
            inputind = self.inputs.index(canonicalize(rawinput))
            rv[outind] = inputind
        return rv

    @classmethod
    def valid(cls, inputs, assumptions):
        d = dict(zip(cls._inputs, inputs))
        if cls.condition is True:
            return True
        return ask(cls.condition.subs(d), assumptions)

def prepend_ones_to_matmuls(expr):
    factor, matrices = expr.as_coeff_matrices()
    return MatMul(factor, *matrices, **{'evaluate': False})

def canonicalize(x):
    if isinstance(x, MatrixExpr):
        return x.canonicalize()
    if isinstance(x, Symbol):
        return x
    if isinstance(x, Expr):
        try:
            return type(x)(*x.args)
        except:
            return x
    else:
        return x
