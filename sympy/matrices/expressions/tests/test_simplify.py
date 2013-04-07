from sympy.matrices.expressions.simplify import simplify_one

from sympy import MatrixSymbol, Q, det, BlockMatrix
from sympy.abc import n

X = MatrixSymbol('X', n, n)
A, B, C, D, E, F, G, H, I = [MatrixSymbol(a, n, n) for a in 'ABCDEFGKI']

def test_simplify_one():
    assert simplify_one(X) is X
    assert simplify_one(X.T, Q.symmetric(X)) == X
    assert simplify_one(det(X), Q.singular(X)) == 0
    assert simplify_one(BlockMatrix([[X]])) == X

def test_block_simplify():
    X = BlockMatrix([[A, B],
                     [C, D]])
    assert simplify_one(det(X), Q.invertible(A)) == det(A)*det(D - C*A.I*B)
