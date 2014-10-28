from sympy import Basic, Expr
from matexpr import ShapeError

import pdb

class Trace(Expr):
    """Matrix Trace

    Represents the trace of a matrix expression.

    >>> from sympy import MatrixSymbol, Trace, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Trace(A)
    Trace(A)

    >>> Trace(eye(3)).doit()
    3
    """

    def __new__(cls, mat):
        if not mat.is_Matrix:
            raise TypeError("input to Trace, %s, is not a matrix" % str(mat))

        if not mat.is_square:
            raise ShapeError("Trace of a non-square matrix")

        #try:
        #    return mat._eval_trace()
        #except (AttributeError, NotImplementedError):
        #    return Basic.__new__(cls, mat)

        return Basic.__new__(cls, mat)

    def _eval_derivative(self, x):
        return Trace(self.arg.diff(x))

    def _eval_transpose(self):
        return self

    @property
    def arg(self):
        return self.args[0]

    def doit(self, **hints):
        arg = self.arg
        if hints.get('deep', True) and isinstance(arg, Basic):
            arg = arg.doit(**hints)
        try:
            #pdb.set_trace()
            result = arg._eval_trace()
            return result if result is not None else Trace(arg)
        except (AttributeError, NotImplementedError):
            return Trace(arg)

        #from sympy import Add
        #return Add(*[self.arg[i, i] for i in range(self.arg.rows)])

        #try:
        #    return self.arg._eval_trace()
        #except (AttributeError, NotImplementedError):
        #    return self


def trace(expr):
    """ Matrix Trace

    >>> from sympy import MatrixSymbol, trace, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> trace(A)
    Trace(A)

    >>> trace(eye(3))
    3
    """
    print ('So this is actually called..')
    return Trace(expr).doit()