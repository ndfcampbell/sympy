import logpy
from logpy import facts, Relation, var, eq
from logpy.unify import unify_dispatch, reify_dispatch, reify, unify_seq
from logpy.variables import variables
from logpy.unifymore import register_object_attrs

from sympy import S, Q, Dummy, assuming, ask, Tuple
from sympy.assumptions import AppliedPredicate

from sympy import Mul, Add, Pow, Abs, And, Or, Not
from sympy.matrices import (MatrixSymbol, Transpose, Inverse,
        Trace, Determinant, MatMul, MatAdd, BlockMatrix, BlockDiagMatrix,
        Identity, ZeroMatrix, Adjoint, HadamardProduct, ImmutableMatrix, det)

classes = (MatrixSymbol, Transpose, Inverse,
        Trace, Determinant, MatMul, MatAdd, BlockMatrix, BlockDiagMatrix,
        Identity, ZeroMatrix, Adjoint, HadamardProduct, AppliedPredicate,
        ImmutableMatrix, Tuple, Mul, Add, Pow, Abs, And, Or, Not)

def unify_Basic(u, v, s):
    return unify_seq((type(u),) + u.args,
                     (type(v),) + v.args,
                     s)
def reify_Basic(u, s):
    args = [reify(arg, s) for arg in u.args]
    return u.func(*args)
def unify_Predicate(u, v, s):
    return unify_seq((type(u), u.func, u.args[0]),
                     (type(v), v.func, v.args[0]))


for cls in classes:
    unify_dispatch[(cls, cls)] = unify_Basic
    reify_dispatch[cls]        = reify_Basic
unify_dispatch[(AppliedPredicate, AppliedPredicate)] = unify_Predicate


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

reduces = Relation('reduces')
facts(reduces, (A.T, A, Q.symmetric(A)),
               (Sq.I, Sq.T, Q.orthogonal(Sq)),

               (det(Sq), S.Zero, Q.singular(Sq)),
               (det(BlockMatrix([[SA,SB],[SC,SD]])), det(SA)*det(SD - SC*SA.I*SB),
                   Q.invertible(SA)),
               (det(BlockMatrix([[SA,SB],[SC,SD]])), det(SD)*det(SA - SB*SD.I*SA),
                   Q.invertible(SD)),
               (det(SA), S.One, Q.orthogonal(SA)),
               (Abs(det(SA)), S.One, Q.unitary(SA)),


               (BlockMatrix([[A]]), A, True),
               (Inverse(BlockMatrix([[SA, SB],
                                     [SC, SD]])),
                BlockMatrix([[ (SA - SB*SD.I*SC).I,  (-SA).I*SB*(SD - SC*SA.I*SB).I],
                             [-(SD - SC*SA.I*SB).I*SC*SA.I,     (SD - SC*SA.I*SB).I]]),
                Q.invertible(SA) & Q.invertible(SD)),



               )


def asko(predicate, truth):
    return (eq, ask(predicate), truth)


def simplify_one(expr, *assumptions, **kwargs):
    with assuming(*assumptions):
        with variables(*vars):
            source, target, condition = var(), var(), var()
            result = logpy.run(1, target, (reduces, source, target, condition),
                                          (eq, source, expr),
                                          (asko, condition, True))
    return result[0] if result else expr
