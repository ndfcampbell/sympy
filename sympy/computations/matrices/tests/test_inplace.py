from sympy.matrices.expressions import MatrixSymbol
from sympy import Q, Symbol, assuming
from sympy.computations.inplace import inplace_compile
from sympy.computations.core import Identity
from sympy.computations.matrices.compile import compile
from sympy.utilities.pytest import slow

@slow
def test_inplace():
    a,b,c, = map(Symbol, 'abc')
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    W = MatrixSymbol('X', 3, 3)
    expr = (Y.I*Z*Y + b*Z*Y + c*W*W).I*Z*W
    comp = Identity(expr)
    assumptions = Q.symmetric(Y), Q.positive_definite(Y), Q.symmetric(X)

    with assuming(*assumptions):
        mathcomp = next(compile(comp))

    icomp = inplace_compile(mathcomp)
    return rule(comp)
