from sympy import Basic, Expr
from matexpr import ShapeError

class Trace(Expr):
    """Matrix Trace

    Represents the trace of a matrix expression.

    >>> from sympy import MatrixSymbol, Trace, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Trace(A)
    Trace(A)

    >>> Trace(eye(3))
    3
    """
    is_Trace = True
    def __new__(cls, mat):

        if not mat.is_square:
            raise ShapeError("Trace of a non-square matrix")

        if not mat.is_Matrix:
            return mat

        try:
            return mat._eval_trace()
        except (AttributeError, NotImplementedError):
            return Basic.__new__(cls, mat)

    @property
    def arg(self):
        return self.args[0]

    def doit(self):
        from sympy import Add
        return Add(*[self.arg[i,i] for i in xrange(self.arg.rows)])