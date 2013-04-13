from sympy.matrices.expressions import MatrixExpr
from sympy.core import S

class EigenVectors(MatrixExpr):
    arg = property(lambda self: self.args[0])
    shape = property(lambda self: self.arg.shape)

class EigenValues(MatrixExpr):
    arg = property(lambda self: self.args[0])
    shape = property(lambda self: (self.arg.shape[0], S.One))

def eig(X):
    return EigenValues(X), EigenVectors(X)
