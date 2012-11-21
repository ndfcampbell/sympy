from sympy.computations.matrices.blas import GEMM, SYMM
from sympy.computations.matrices.lapack import GESV, POSV

from sympy.computations.matrices.shared import (alpha, beta, n, m, k, A, B, C,
        x, a, b)
from sympy import Q, S, ask
from sympy.computations.compile import input_crunch, brulify
from sympy.unify import patternify, unify
from sympy.rules.branch import multiplex, exhaust, debug
from sympy.rules.tools import subs

def expr_to_comp_rule(source, target, wilds, condition, assumptions):
    """
    source - a mathematical expression to trigger this transformation
    target - the resulting pattern of the transformation
    wilds  - the wilds of the source pattern
    condition - A boolean on the wilds that must hold true
    assumptions - assumptions under which the condition must hold true
    """
    pattern = patternify(source, *wilds)
    def matrix_expr_to_comp_brule(expr):
        for match in unify(expr, pattern):
            if (condition is True):
                yield subs(match)(target)
            elif ask(condition.subs(match), assumptions):
                yield subs(match)(target)
    return matrix_expr_to_comp_brule

# pattern is (source expression, target expression, wilds, condition)
blas_patterns = [
    (GEMM._outputs[0], GEMM(*GEMM._inputs), GEMM._inputs, GEMM.condition),
    (alpha*A*B, GEMM(alpha, A, B, S.Zero, B), (alpha, A, B), True),
    (A*B, GEMM(S.One, A, B, S.Zero, B), (A, B), True),
    (SYMM._outputs[0], SYMM(*SYMM._inputs), SYMM._inputs, SYMM.condition),
    (alpha*A*B, SYMM(alpha, A, B, S.Zero, B), (alpha, A, B), SYMM.condition),
    (A*B, SYMM(S.One, A, B, S.Zero, B), (A, B), SYMM.condition),
]
lapack_patterns = [
    (POSV._outputs[0], POSV(*POSV._inputs), POSV._inputs, POSV.condition),
    (GESV._outputs[0], GESV(*GESV._inputs), GESV._inputs, GESV.condition),
]

patterns = lapack_patterns + blas_patterns

def make_rule(patterns, assumptions):
    rules = [expr_to_comp_rule(src, target, wilds, cond, assumptions)
            for src, target, wilds, cond in patterns]
    return exhaust(multiplex(*map(input_crunch, rules)))
