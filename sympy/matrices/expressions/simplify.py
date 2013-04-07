

# Teach LogPy how to manipulate SymPy
import logpy
from logpy import facts, Relation, var, eq
from logpy.unify import unify_dispatch, reify_dispatch, reify, unify_seq
from logpy.variables import variables
from logpy.unifymore import register_object_attrs

from sympy.assumptions import AppliedPredicate
from sympy import Mul, Add, Pow, Abs, And, Or, Not, Tuple
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

# Gather and assert known relations

from simplifydata import known_relations, vars
reduces = Relation('reduces')
facts(reduces, *known_relations)


# Simplification code
from sympy import assuming, ask

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
