from matexpr import MatrixExpr
from sympy import Basic, ask, Q

class Transpose(MatrixExpr):
    """Matrix Transpose

    Represents the transpose of a matrix expression.

    Use .T as shorthand

    >>> from sympy import MatrixSymbol, Transpose
    >>> A = MatrixSymbol('A', 3, 5)
    >>> B = MatrixSymbol('B', 5, 3)
    >>> Transpose(A)
    A'
    >>> A.T
    A'
    >>> Transpose(A*B)
    B'*A'
    """
    is_Transpose = True
    def __new__(cls, mat, evaluate=True):

        obj = Basic.__new__(cls, mat)
        if evaluate:
            return obj.simplify()
        else:
            return obj

    def simplify(self):
        mat = self.arg

        if not mat.is_Matrix:
            return mat

        if isinstance(mat, Transpose):
            return mat.arg

        if hasattr(mat, 'transpose'):
            return mat.transpose()

        if mat.is_Mul:
            return MatMul(*[Transpose(arg) for arg in mat.args[::-1]])

        if mat.is_Add:
            return MatAdd(*[Transpose(arg) for arg in mat.args])

        if mat.is_Pow:
            return MatPow(Transpose(mat.base), mat.exp)

        if ask(Q.symmetric(mat)):
            return mat

        return Transpose(mat, evaluate=False)

    @property
    def arg(self):
        return self.args[0]

    @property
    def shape(self):
        return self.arg.shape[::-1]

    def _entry(self, i, j):
        return self.arg._entry(j, i)

    def equals(self, other):
        return self.simplify() == other.simplify()

from matmul import MatMul
from matadd import MatAdd
