from sympy.matrices.expressions.gen import blas_rule
from sympy.matrices.expressions.blas import TRSV
from sympy.matrices.expressions import MatrixSymbol

def test_blas_rule():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    x = MatrixSymbol('x', 3, 1)

    expr = (3*X*Y + 2*Z).I*x
    assert isinstance(blas_rule(expr).next(), TRSV)
