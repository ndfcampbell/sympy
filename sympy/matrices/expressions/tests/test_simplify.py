from sympy.matrices.expressions.simplify import simplify_one

from sympy import Q, Abs
from sympy.matrices.expressions import MatrixSymbol, BlockMatrix, det, Inverse
from sympy.abc import n
from sympy.utilities.pytest import XFAIL

X = MatrixSymbol('X', n, n)
A, B, C, D, E, F, G, H, I = [MatrixSymbol(a, n, n) for a in 'ABCDEFGKI']

def test_simplify_one():
    assert simplify_one(X) is X
    assert simplify_one(X.T, Q.symmetric(X)) == X
    assert simplify_one(BlockMatrix([[X]])) == X

def test_simplify_block():
    X = BlockMatrix([[A, B],
                     [C, D]])
    assert simplify_one(det(X), Q.invertible(A)) == det(A)*det(D - C*A.I*B)

    assert not isinstance(simplify_one(X.I, Q.invertible(A), Q.invertible(D)),
                          Inverse)
    assert simplify_one(X.I) == X.I

def test_simplify_determinants():
    assert simplify_one(det(X), Q.singular(X)) == 0
    assert simplify_one(det(X), Q.orthogonal(X)) == 1
    assert simplify_one(Abs(det(X)), Q.unitary(X)) == 1

@XFAIL
def test_MatrixElement():
    assert simplify_one(A[2, 1], Q.upper_triangular(A)) == 0
