from sympy.matrices.expressions import MatrixSymbol, det, Inverse, BlockMatrix
from sympy import Dummy, Symbol, Tuple, Q, S, Abs


# Dummy variables
n, m, l, k = map(Dummy, 'nmlk')
A = MatrixSymbol('_A', n, m)
B = MatrixSymbol('_B', n, k)
C = MatrixSymbol('_A', l, m)
D = MatrixSymbol('_B', l, k)
SA = MatrixSymbol('_SA', n, n)
SB = MatrixSymbol('_SB', n, n)
SC = MatrixSymbol('_SC', n, n)
SD = MatrixSymbol('_SD', n, n)
slice1, slice2 = Tuple(1, 223432, 3), Tuple(4, 252345, 6)

Sq = MatrixSymbol('_Sq', n, n)
vars = [A, B, n, m, Sq, SA, SB, SC, SD]

known_relations = [
    (A.T, A, Q.symmetric(A)),
    (Sq.I, Sq.T, Q.orthogonal(Sq)),

    # Determinants
    (det(Sq), S.Zero, Q.singular(Sq)),
    (det(BlockMatrix([[SA,SB],[SC,SD]])), det(SA)*det(SD - SC*SA.I*SB),
        Q.invertible(SA)),
    (det(BlockMatrix([[SA,SB],[SC,SD]])), det(SD)*det(SA - SB*SD.I*SA),
        Q.invertible(SD)),
    (det(SA), S.One, Q.orthogonal(SA)),
    (Abs(det(SA)), S.One, Q.unitary(SA)),

    # BlockMatrices
    (BlockMatrix([[A]]), A, True),
    (Inverse(BlockMatrix([[SA, SB],
                          [SC, SD]])),
     BlockMatrix([[ (SA - SB*SD.I*SC).I,  (-SA).I*SB*(SD - SC*SA.I*SB).I],
                  [-(SD - SC*SA.I*SB).I*SC*SA.I,     (SD - SC*SA.I*SB).I]]),
     Q.invertible(SA) & Q.invertible(SD)),
]
