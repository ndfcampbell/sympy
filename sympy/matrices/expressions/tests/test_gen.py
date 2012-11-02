from sympy.matrices.expressions.gen import blas_rule
from sympy.matrices.expressions.blas import TRSV, GEMM
from sympy.matrices.expressions import MatrixSymbol

def test_blas_rule():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    x = MatrixSymbol('x', 3, 1)

    expr = (3*X*Y + 2*Z).I*x
    comp = blas_rule(expr).next()
    assert isinstance(comp, TRSV)
    comp2 = blas_rule(comp.inputs[0]).next()
    assert isinstance(comp2, GEMM)
