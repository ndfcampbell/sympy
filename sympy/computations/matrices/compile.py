from sympy.computations.matrices.blas import GEMM, SYMM, AXPY
from sympy.computations.matrices.lapack import GESV, POSV, IPIV, LASWP
from sympy.computations.matrices.shared import (alpha, beta, n, m, k, A, B, C,
        x, a, b, X, Y, Z)
from sympy import Q, S, ask, Expr, Symbol
from sympy.matrices.expressions import (MatrixExpr, PermutationMatrix,
        MatrixSymbol, ZeroMatrix)
from sympy.computations.compile import input_crunch, multi_output_rule
from sympy.unify import unify, rewriterule
from sympy.rules.branch import multiplex, exhaust, debug, sfilter
from sympy.rules.tools import subs
import functools

basetypes = (Expr, MatrixExpr)
def basetype(var):
    for bt in basetypes:
        if isinstance(var, bt):
            return bt

def typecheck(wilds):
    def check(*variables):
        return all(basetype(v) == basetype(w)
                   for v, w in zip(variables, wilds))
    return check

def expr_to_comp_rule(source, target, wilds, condition, assumptions):
    """
    source - a mathematical expression to trigger this transformation
    target - the resulting pattern of the transformation
    wilds  - the wilds of the source pattern
    condition - A boolean on the wilds that must hold true
    assumptions - assumptions under which the condition must hold true
    """
    # pattern = patternify(source, *wilds, types=wildtypes(wilds))
    def matrix_expr_to_comp_brule(expr):
        for match in unify(expr, source, variables=wilds):
            if (condition is True):
                yield subs(match)(target)
            elif ask(condition.subs(match), assumptions):
                yield subs(match)(target)
    return matrix_expr_to_comp_brule

# pattern is (source expression, target expression, wilds, condition)
blas_patterns = [
    (alpha*A*B + beta*C, SYMM(*SYMM._inputs), SYMM._inputs, SYMM.condition),
    (alpha*A*B + C, SYMM(alpha, A, B, S.One, C), (alpha, A, B, C), SYMM.condition),
    (A*B + beta*C, SYMM(S.One, A, B, beta, C), (A, B, beta, C), SYMM.condition),
    (A*B + C, SYMM(S.One, A, B, S.One, C), (A, B, C), SYMM.condition),
    (alpha*A*B, SYMM(alpha, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (alpha, A, B), SYMM.condition),
    (A*B, SYMM(S.One, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (A, B), SYMM.condition),

    (alpha*A*B + beta*C, GEMM(*GEMM._inputs), GEMM._inputs, GEMM.condition),
    (alpha*A*B + C, GEMM(alpha, A, B, S.One, C), (alpha, A, B, C), True),
    (A*B + beta*C, GEMM(S.One, A, B, beta, C), (A, B, beta, C), True),
    (A*B + C, GEMM(S.One, A, B, S.One, C), (A, B, C), True),
    (alpha*A*B, GEMM(alpha, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (alpha, A, B), True),
    (A*B, GEMM(S.One, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (A, B), True),

    (alpha*X + Y, AXPY(*AXPY._inputs), AXPY._inputs, AXPY.condition),
    (X + Y, AXPY(S.One, X, Y), (X, Y), True)
]
lapack_patterns = [
    (POSV._outputs[0], POSV(*POSV._inputs), POSV._inputs, POSV.condition),
    (Z.I*X, GESV(Z, X), (Z, X), True),
]

multi_out_patterns = [
    ((IPIV(A), PermutationMatrix(IPIV(A))*A),
      LASWP(PermutationMatrix(IPIV(A))*A, IPIV(A)), (A,), True)
]

patterns = lapack_patterns + blas_patterns

def good_computation(c):
    """ Our definition of an acceptable computation

    Must:
        contain only symbols and matrix symbols as inputs
    """
    if all(isinstance(inp, (Symbol, MatrixSymbol)) for inp in c.inputs):
        return True
    else:
        return False

def make_inrule(pattern, assumptions):
    src, target, wilds, conds = pattern
    brl = rewriterule(src, target, wilds, condition=typecheck(wilds),
                                          assume=conds)
    brl = functools.partial(brl, assumptions=assumptions)
    return input_crunch(brl)

def make_rule(patterns, assumptions):
    input_brules = [make_inrule(pattern, assumptions) for pattern in patterns]

    output_brules = [multi_output_rule(sources, target, *wilds)
            for sources, target, wilds, condition in multi_out_patterns]
    return sfilter(good_computation,
                   (exhaust(multiplex(*(output_brules + input_brules)))))
