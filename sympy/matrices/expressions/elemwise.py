from sympy.matrices.expressions.matexpr import MatrixExpr, ShapeError
from sympy.core.basic import Basic

class ElemwiseMatrix(MatrixExpr):
    """ Apply Elementwise operation to list of matrices """
    fn      = property(lambda self: self.args[0])
    parents = property(lambda self: self.args[1:])
    shape = property(lambda self: self.parents[0].shape)

    def _entry(self, i, j):
        return self.fn(*[p._entry(i, j) for p in self.parents])

    def __new__(cls, fn, *parents, **kwargs):
        if kwargs.get('check'):
            validate(fn, *parents)
        return Basic.__new__(cls, fn, *parents)


def validate(fn, *matrices):
    if not all(m.shape == matrices[0].shape for m in matrices[1:]):
        raise ShapeError()
